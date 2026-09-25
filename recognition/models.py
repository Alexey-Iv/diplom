"""HardNet и AG-Net из глав 4 и 5 диплома, а также функции потерь."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import roi_align


class HardNet(nn.Module):
    """Серый патч 32 x 32 -> нормированный дескриптор из 128 чисел."""

    def __init__(self):
        super().__init__()
        layers = []
        channels = [1, 32, 32, 64, 64, 128, 128]
        for index, (source, target) in enumerate(
            zip(channels[:-1], channels[1:])
        ):
            stride = 2 if index in (2, 4) else 1
            layers.extend([
                nn.Conv2d(source, target, 3, stride, 1, bias=False),
                nn.BatchNorm2d(target, affine=False),
                nn.ReLU(inplace=True),
            ])
        layers.extend([
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, 8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        ])
        self.features = nn.Sequential(*layers)

    def forward(self, patches):
        if patches.shape[1:] != (1, 32, 32):
            raise ValueError("HardNet expects [batch, 1, 32, 32]")
        mean = patches.mean(dim=(2, 3), keepdim=True).detach()
        std = patches.std(dim=(2, 3), keepdim=True).detach()
        normalized = (patches - mean) / (std + 1e-7)
        return F.normalize(self.features(normalized).flatten(1), dim=1)


class SpatialAttention(nn.Module):
    """Self-attention связывает разные позиции карты признаков."""

    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.delta = nn.Parameter(torch.zeros(()))

    def forward(self, features):
        query = self.query(features).flatten(2).transpose(1, 2)
        key = self.key(features).flatten(2)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        value = self.value(features).flatten(2).transpose(1, 2)
        context = torch.bmm(attention, value).transpose(1, 2)
        return features + self.delta * context.reshape_as(features)


class SEResidual(nn.Module):
    """Один общий SE-блок для всех регионов размером 7 x 7."""

    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        residual = self.body(features)
        result = F.relu(features + residual * self.gate(residual))
        return result.mean(dim=(2, 3))


class AGNet(nn.Module):
    """ResNet-50 + KeyNet/GMM regions + spatial and region attention.

    Boxes are normalized [x1, y1, x2, y2], including the full image.
    Width reduction before region pooling bounds memory at kappa=22.
    """

    def __init__(self, embedding_dim=256, channels=128, pretrained=False):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.spatial_attention = SpatialAttention(2048)
        self.reduce = nn.Conv2d(2048, channels, 1)
        self.region_encoder = SEResidual(channels)
        attention_dim = 32
        self.target = nn.Linear(channels, attention_dim)
        self.source = nn.Linear(channels, attention_dim, bias=False)
        self.relation = nn.Linear(attention_dim, 1)
        self.importance = nn.Linear(channels, 1)
        self.embedding = nn.Linear(channels, embedding_dim)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )

    def forward(self, images, boxes):
        rgb = images.expand(-1, 3, -1, -1)
        features = self.backbone((rgb - self.mean) / self.std)
        features = self.reduce(self.spatial_attention(features))
        batch, _, height, width = features.shape
        scaled = boxes * boxes.new_tensor([width, height, width, height])
        regions = roi_align(
            features, list(scaled.unbind()), output_size=7,
            sampling_ratio=2, aligned=True,
        )
        # Chunking prevents a single huge convolution over all 254 regions.
        vectors = torch.cat([
            self.region_encoder(part) for part in regions.split(64)
        ]).reshape(batch, boxes.shape[1], -1)
        target = self.target(vectors)
        source = self.source(vectors)
        contexts = []
        for chunk in target.split(32, dim=1):
            relation = torch.tanh(chunk[:, :, None] + source[:, None])
            weights = torch.sigmoid(self.relation(relation).squeeze(-1))
            contexts.append(torch.bmm(weights, vectors))
        context = torch.cat(contexts, dim=1)
        importance = self.importance(context).softmax(dim=1)
        combined = (importance * context).sum(dim=1)
        return F.normalize(self.embedding(combined), dim=1)


class ArcFace(nn.Module):
    """ArcFace добавляет угловой отступ только к правильному классу."""

    def __init__(self, embedding_dim, classes, margin=0.5, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale

    def forward(self, embeddings, labels):
        cosine = F.linear(embeddings, F.normalize(self.weight, dim=1))
        cosine = cosine.clamp(-1 + 1e-6, 1 - 1e-6)
        angle = torch.acos(cosine)
        target = torch.cos(angle + self.margin)
        # Monotonic continuation for theta + margin beyond pi.
        boundary = torch.cos(cosine.new_tensor(torch.pi - self.margin))
        correction = self.margin * torch.sin(
            cosine.new_tensor(torch.pi - self.margin)
        )
        target = torch.where(cosine > boundary, target, cosine - correction)
        one_hot = F.one_hot(labels, cosine.shape[1]).to(cosine.dtype)
        return self.scale * (cosine + one_hot * (target - cosine))


def hardnet_loss(anchor, positive, labels, margin=1.0):
    """Ближайший отрицательный пример; та же радужка исключается."""
    distances = torch.cdist(anchor, positive)
    negatives = labels[:, None] != labels[None, :]
    if not negatives.any(dim=1).all():
        raise ValueError("HardNet batch needs at least two iris identities")
    masked = distances.masked_fill(~negatives, float("inf"))
    hardest = torch.minimum(masked.min(0).values, masked.min(1).values)
    return F.relu(distances.diag() - hardest + margin).mean()


def batch_hard_triplet(embeddings, labels, margin=0.3):
    """Самый далёкий свой и ближайший чужой пример в батче P x K."""
    distances = torch.cdist(embeddings, embeddings)
    same = labels[:, None] == labels[None, :]
    positive = same & ~torch.eye(
        len(labels), dtype=torch.bool, device=labels.device
    )
    if not positive.any(1).all() or not (~same).any(1).all():
        raise ValueError("Triplet batches need P >= 2 and K >= 2")
    farthest = distances.masked_fill(~positive, -float("inf")).max(1).values
    nearest = distances.masked_fill(same, float("inf")).min(1).values
    return F.relu(farthest - nearest + margin).mean()
