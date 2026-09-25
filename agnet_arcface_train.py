import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from sklearn.metrics import roc_curve
import yaml

# Импорт вашей архитектуры
from keyNet.model.keynet_architecture import keynet
from keyNet.model.hybrid_net import IrisIdentificationNet
import keyNet.config as config

# ==========================================
# 1. ARCFACE LAYER
# ==========================================
class ArcFace(nn.Module):
    def __init__(self, in_features, num_classes, s=30.0, m=0.5):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_normal_(self.W)

    def forward(self, x, labels):
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.W, p=2, dim=1)
        cos_theta = torch.matmul(x, w.T)
        sin_theta = torch.sqrt(1.0 - torch.clamp(cos_theta, -1.0, 1.0) ** 2)
        phi = cos_theta * torch.cos(torch.tensor(self.m)) - sin_theta * torch.sin(torch.tensor(self.m))
        one_hot = torch.zeros_like(cos_theta).scatter_(1, labels.view(-1, 1), 1)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cos_theta)
        logits *= self.s
        return logits

# ==========================================
# 2. КОМБИНИРОВАННАЯ МОДЕЛЬ (BACKBONE + ARCFACE)
# ==========================================
class CombinedIrisModel(nn.Module):
    def __init__(self, backbone, embedding_dim, num_classes, s=30.0, m=0.5):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(p=0.15)
        self.arcface = ArcFace(embedding_dim, num_classes, s, m)

    def forward(self, x, labels=None, masks=None):
        emb = self.backbone(x, masks)

        # Применяем Dropout только при обучении!
        if self.training:
            emb_dropped = self.dropout(emb)
        else:
            emb_dropped = emb

        if labels is not None:
            logits = self.arcface(emb_dropped, labels)
            return logits, emb
        else:
            return emb

# ==========================================
# ONLINE HARD TRIPLET LOSS
# ==========================================

def filter_valid_labels(embeddings, labels, min_samples=4):
    unique, counts = torch.unique(labels, return_counts=True)
    valid_classes = unique[counts >= min_samples]
    if valid_classes.numel() == 0:
        return None, None
    mask = torch.isin(labels, valid_classes)
    return embeddings[mask], labels[mask]


class OnlineHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3, min_samples_per_class=4):
        super().__init__()
        self.margin = margin
        self.min_samples = min_samples_per_class

    def forward(self, embeddings, labels):
        # Нормализуем ВНУТРИ — класс самодостаточен
        embeddings = F.normalize(embeddings, p=2, dim=1)
        device = embeddings.device

        # Фильтрация классов с < min_samples примеров
        embeddings, labels = filter_valid_labels(embeddings, labels, self.min_samples)
        if embeddings is None:
            # ✅ device известен до фильтрации — используем замыкание
            return torch.tensor(0.0, device=device, requires_grad=True)

        n = embeddings.size(0)
        dist_mat = torch.cdist(embeddings, embeddings, p=2)

        same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(n, dtype=torch.bool, device=device)

        mask_pos = same_class & ~eye
        mask_neg = ~same_class & ~eye

        # Hard Positive
        pos_dist = dist_mat.clone()
        pos_dist[~mask_pos] = -1.0
        hard_pos, _ = pos_dist.max(dim=1)

        # Hard Negative
        neg_dist = dist_mat.clone()
        neg_dist[~mask_neg] = float('inf')
        hard_neg, _ = neg_dist.min(dim=1)

        valid = mask_pos.any(dim=1) & mask_neg.any(dim=1)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = F.relu(hard_pos[valid] - hard_neg[valid] + self.margin)
        active = loss[loss > 1e-16]
        return active.mean() if active.numel() > 0 else torch.tensor(0.0, device=device, requires_grad=True)


from torch.utils.data.sampler import Sampler

# ==========================================
# 4.5 PK BATCH SAMPLER (Для Hard Triplet Loss)
# ==========================================
class PKBatchSampler(Sampler):
    """
    Формирует батчи размера P * K, где P - количество уникальных классов (людей) в батче,
    а K - количество изображений для каждого класса.
    """
    def __init__(self, labels, p, k):
        self.p = p
        self.k = k
        self.batch_size = p * k
        self.labels = np.array(labels)
        self.classes = np.unique(self.labels)

        # Заранее группируем индексы картинок по классам
        self.label_to_indices = {
            c: np.where(self.labels == c)[0] for c in self.classes
        }

    def __iter__(self):
        # В начале каждой эпохи перемешиваем классы (людей)
        np.random.shuffle(self.classes)

        batch = []
        for c in self.classes:
            indices = self.label_to_indices[c]

            # Берем ровно K случайных фоток этого человека
            # Если у человека фоток меньше, чем K, берем с повторением (replace=True)
            replace = len(indices) < self.k
            sampled_indices = np.random.choice(indices, size=self.k, replace=replace)

            batch.extend(sampled_indices)

            # Если батч заполнился (P классов по K фоток = P*K)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        # Мы специально отбрасываем последний неполный батч (drop_last),
        # чтобы Triplet Loss всегда работал с идеальными пропорциями.

    def __len__(self):
        return len(self.classes) // self.p


# ==========================================
# 4. ЕДИНЫЙ ДАТАСЕТ (Для Train и Val)
# ==========================================
class IrisDataset(Dataset):
    def __init__(self, txt_file, masks_dir, transform=None):
        self.transform = transform
        self.masks_dir = masks_dir  # ПУТЬ К ВАШИМ МАСКАМ
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        with open(txt_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        for full_path in lines:
            filename = os.path.basename(full_path)
            parts = filename.split('_')
            if len(parts) >= 2:
                class_id = f"{parts[0]}_{parts[1]}"
                self.image_paths.append(full_path)
                if class_id not in self.class_to_idx:
                    self.class_to_idx[class_id] = len(self.class_to_idx)
                self.labels.append(self.class_to_idx[class_id])

        self.num_classes = len(self.class_to_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        filename = os.path.basename(img_path)

        # 1. ЗАГРУЖАЕМ ИЗОБРАЖЕНИЕ
        img = Image.open(img_path).convert('RGB')

        # 2. ЗАГРУЖАЕМ МАСКУ (ВНИМАНИЕ: укажите правильное имя файла маски!)
        # Если маска называется так же, как картинка:
        mask_path = os.path.join(self.masks_dir, filename)
        # Если у маски есть суффикс, например _mask.jpg, раскомментируйте строку ниже:
        # mask_path = os.path.join(self.masks_dir, filename.replace('.jpg', '_mask.jpg'))

        # Загружаем маску в ЧБ, ресайзим строго до 512x64 (W x H)
        mask_img = Image.open(mask_path).convert('L').resize((512, 64), Image.NEAREST)

        # Переводим маску в тензор [H, W], где 1.0 - радужка, 0.0 - веки
        mask_tensor = torch.from_numpy(np.array(mask_img)).float() / 255.0

        if self.transform:
            img = self.transform(img)

        return img, label, mask_tensor

# ==========================================
# 5. ФУНКЦИЯ ВЫДЕЛЕНИЯ KAPPA-ОБЛАСТЕЙ
# ==========================================
def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    device = tensor.device
    mean_t = torch.tensor(mean, device=device).view(3, 1, 1)
    std_t  = torch.tensor(std,  device=device).view(3, 1, 1)
    tensor = tensor * std_t + mean_t        # обратная нормализация
    tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor.permute(1, 2, 0).cpu().numpy()  # (H,W,3)

def visualize_kappa_regions(image_vis, SRs, keypoints, region_counts, save_path):
    # image_vis — tensor_vis БЕЗ нормализации, [1, C, H, W] в [0,1]
    img = image_vis[0].permute(1, 2, 0).cpu().numpy()  # (H,W,3)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = np.ascontiguousarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for kp in keypoints[0]:
        x, y = int(kp[0]), int(kp[1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), 2, (0, 255, 255), -1)

    n_regions = int(region_counts[0])
    for box in SRs[:n_regions]:
        _, x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imwrite(save_path, img)

def apply_vertical_fade(tensor_images, fade_pixels=8):
    """Слегка затушевывает черные границы обрезки век, заставляя сеть смотреть в центр"""
    b, c, h, w = tensor_images.shape
    mask = torch.ones_like(tensor_images)
    fade_in = (1 - torch.cos(torch.linspace(0, torch.pi, fade_pixels, device=tensor_images.device))) / 2
    for i in range(fade_pixels):
        mask[:, :, i, :] = fade_in[i]
        mask[:, :, h - 1 - i, :] = fade_in[i]
    return tensor_images * mask

# ==========================================
# 6. ВЫЧИСЛЕНИЕ EER И ГРАФИКИ
# ==========================================
def compute_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings, all_labels, all_paths = [], [], []
    with torch.no_grad():
        idx = 0
        for images, labels, masks in tqdm(dataloader, desc="Extracting embeddings", leave=False):
            images, masks = images.to(device).float(), masks.to(device)
            #images = apply_vertical_fade(images) # Применяем защиту от век
            emb = model(images, masks=masks)
            all_embeddings.append(emb.cpu())
            all_labels.extend(labels.numpy())

            batch_size = labels.size(0)
            all_paths.extend(dataloader.dataset.image_paths[idx:idx+batch_size])
            idx += batch_size

    return torch.cat(all_embeddings, dim=0), np.array(all_labels), all_paths


class CircleLoss(nn.Module):
    """
    Circle Loss (Sun et al., 2020)
    Работает с нормализованными эмбеддингами.
    m  — margin (рекомендуется 0.25)
    γ  — масштаб (рекомендуется 80-128 для iris)
    """
    def __init__(self, m: float = 0.25, gamma: float = 80.0):
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        device = embeddings.device

        # Матрица косинусных схожестей
        sim = torch.matmul(embeddings, embeddings.T)   # [N, N]

        n = embeddings.size(0)
        eye = torch.eye(n, dtype=torch.bool, device=device)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)

        mask_pos = same & ~eye   # истинные позитивные пары
        mask_neg = ~same & ~eye  # истинные негативные пары

        # Адаптивные весовые коэффициенты
        # Op = оптимальная схожесть для позитивов = 1, On = 0
        alpha_pos = torch.clamp(-sim.detach() + 1 + self.m, min=0.0)
        alpha_neg = torch.clamp( sim.detach()      + self.m, min=0.0)

        # Для каждого анкора суммируем взвешенные вклады
        # Сдвиги: позитивы тянутся к 1, негативы — к 0
        delta_pos = 1 - self.m
        delta_neg = self.m

        losses = []

        for i in range(n):
            pos_idx = mask_pos[i]
            neg_idx = mask_neg[i]
            if not pos_idx.any() or not neg_idx.any():
                continue

            logit_pos = -self.gamma * alpha_pos[i][pos_idx] * (sim[i][pos_idx] - delta_pos)
            logit_neg =  self.gamma * alpha_neg[i][neg_idx] * (sim[i][neg_idx] - delta_neg)

            loss_i = self.soft_plus(
                torch.logsumexp(logit_neg, dim=0) +
                torch.logsumexp(logit_pos, dim=0)
            )
            losses.append(loss_i)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(losses).mean()

def compute_eer(embeddings, labels, save_plot=False, save_dir=None, epoch=None, prefix="val"):
    embeddings = F.normalize(embeddings, p=2, dim=1)
    num_images = embeddings.shape[0]
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    upper_tri = torch.triu_indices(num_images, num_images, offset=1)
    distances = dist_matrix[upper_tri[0], upper_tri[1]].numpy()
    matches = (labels[upper_tri[0]] == labels[upper_tri[1]])

    genuine_dists = distances[matches]
    imposter_dists = distances[~matches]

    if len(genuine_dists) == 0 or len(imposter_dists) == 0: return None

    y_true = matches.astype(int)
    y_score = -distances
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr

    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_idx]
    eer_threshold = -thresholds[eer_idx]

    actual_thresholds = -thresholds[1:]
    fpr = fpr[1:]
    fnr = fnr[1:]

    # Сохранение тех самых красивых графиков ROC, Hist, FAR/FRR
    if save_plot and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        # 1. Распределение дистанций
        plt.figure(figsize=(10, 6))
        plt.hist(genuine_dists, bins=50, alpha=0.6, color='green', density=True, label=f'Genuine (mean={np.mean(genuine_dists):.2f})')
        plt.hist(imposter_dists, bins=50, alpha=0.6, color='red', density=True, label=f'Imposter (mean={np.mean(imposter_dists):.2f})')
        plt.axvline(x=eer_threshold, color='blue', linestyle='--', label=f'Threshold={eer_threshold:.3f}')
        plt.title(f'[{prefix.upper()}] L2 Distance (EER = {eer*100:.2f}%)')
        plt.xlabel('L2 Distance')                    # ← ДОБАВЛЕНО: подпись оси X
        plt.ylabel('Probability Density')            # ← ДОБАВЛЕНО: подпись оси Y
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'{prefix}_hist_epoch_{epoch}.png'), dpi=300)
        plt.close()

        # 2. ROC Curve
        plt.figure(figsize=(8, 6))
        _, tpr_full, _ = roc_curve(y_true, y_score)
        fpr_full = np.concatenate(([0.0], fpr))
        tpr_full = np.concatenate(([0.0], tpr_full[1:]))
        plt.plot(fpr_full, tpr_full, label=f'ROC (EER={eer*100:.2f}%)', color='darkorange')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.scatter(eer, 1-eer, color='red', zorder=5)
        plt.title(f'[{prefix.upper()}] ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'{prefix}_roc_epoch_{epoch}.png'), dpi=300)
        plt.close()

        # 3. FAR & FRR
        plt.figure(figsize=(8, 6))
        plt.plot(actual_thresholds, fpr, label='FAR (False Accept Rate)', color='red', linewidth=2)
        plt.plot(actual_thresholds, fnr, label='FRR (False Reject Rate)', color='blue', linewidth=2)
        plt.axvline(x=eer_threshold, color='green', linestyle='--', label=f'EER Threshold ({eer_threshold:.3f})')
        plt.plot(eer_threshold, eer, 'ko', markersize=6, label=f'EER = {eer*100:.2f}%', zorder=5)
        plt.xlabel('Distance Threshold')
        plt.ylabel('Error Rate')
        plt.title(f'[{prefix.upper()}] FAR and FRR Curves')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(os.path.join(save_dir, f'{prefix}_far_frr_epoch_{epoch}.png'), dpi=300)
        plt.close()

    return eer, eer_threshold

def save_hard_examples(embeddings, labels, image_paths, eer_threshold, save_dir, prefix="val"):
    os.makedirs(save_dir, exist_ok=True)

    embeddings = F.normalize(embeddings, p=2, dim=1)
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)

    n = len(labels)

    bad_genuine = []
    #bad_imposter = []

    for i in range(n):
        for j in range(i+1, n):
            dist = dist_matrix[i, j].item()
            same = labels[i] == labels[j]

            # Genuine ошибка — слишком далеко
            if same and dist > eer_threshold:
                bad_genuine.append((image_paths[i], image_paths[j], dist))

            # Imposter ошибка — слишком близко
            # if not same and dist < eer_threshold:
            #     bad_imposter.append((image_paths[i], image_paths[j], dist))

    # Сортируем по "плохости"
    bad_genuine.sort(key=lambda x: -x[2])
    #bad_imposter.sort(key=lambda x: x[2])

    # Сохраняем
    with open(os.path.join(save_dir, f"{prefix}_bad_genuine.txt"), "w") as f:
        for p1, p2, d in bad_genuine:
            f.write(f"{p1} {p2} dist={d:.4f}\n")

    # with open(os.path.join(save_dir, f"{prefix}_bad_imposter.txt"), "w") as f:
    #     for p1, p2, d in bad_imposter:
    #         f.write(f"{p1} {p2} dist={d:.4f}\n")

    print(f"Saved {len(bad_genuine)} bad genuine pairs")
    #print(f"Saved {len(bad_imposter)} bad imposter pairs")

# ==========================================
# 7. РАЗДЕЛЬНЫЕ ГРАФИКИ ОБУЧЕНИЯ
# ==========================================
def plot_training_history(train_losses, train_eers_tuples, val_eers_tuples, save_path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    epochs = np.arange(1, len(train_losses)+1)

    # 1. График Loss
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Training Loss')
    ax1.grid(True, linestyle='--', alpha=0.4); ax1.legend()

    # 2. График VAL EER (Квадратики)
    if len(val_eers_tuples) > 0:
        val_ep, val_vals = zip(*val_eers_tuples)
        val_vals_percent = [v * 100 for v in val_vals]
        ax2.plot(val_ep, val_vals_percent, 'g-', marker='s', label='Val EER (%)')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('EER (%)'); ax2.set_title('Validation EER')
    ax2.grid(True, linestyle='--', alpha=0.4); ax2.legend()

    # 3. График TRAIN EER (Оранжевые точки)
    if len(train_eers_tuples) > 0:
        tr_ep, tr_vals = zip(*train_eers_tuples)
        tr_vals_percent = [v * 100 for v in tr_vals]
        ax3.plot(tr_ep, tr_vals_percent, linestyle='--', color='darkorange', marker='o', label='Train EER (%)')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('EER (%)'); ax3.set_title('Train EER')
    ax3.grid(True, linestyle='--', alpha=0.4); ax3.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==========================================
# 8. НАСТРОЙКИ И ПУТИ
# ==========================================
def get_next_run_dir(base_dir="arg-net"):
    os.makedirs(base_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_dir) if d.startswith("train") and os.path.isdir(os.path.join(base_dir, d))]
    nums = []
    for d in existing_dirs:
        try: nums.append(int(d.replace("train", "")))
        except ValueError: pass
    next_num = max(nums) + 1 if nums else 1
    return os.path.join(base_dir, f"train{next_num}")

args_dict = {
    'device': 'cuda:5' if torch.cuda.is_available() else 'cpu',
    'batch_size': 50,               # Увеличили для эффективности Hard Mining!
    'num_epochs': 120,
    'learning_rate': 0.0005,       # Безопасный шаг для ArcFace
    'eval_val_every': 5,            # Val каждый раз
    'eval_train_every': 10,         # Train каждые 10 раз
    'triplet_margin': 0.6,          # Снизили margin для Hard Mining
    'triplet_weight': 2.0,          # ТЕПЕРЬ ОН ВКЛЮЧЕН
    'arcface_weight': 0.3,
    'arcface_s': 55.0,
    'arcface_m': 0.50,               # Снизили до 0.4 для лучшей конвергенции
    'circle_weight': 0,      # ← новый
    'circle_m': 0.5,          # ← margin
    'circle_gamma': 100.0,      # ← масштаб (80-128 для iris)
    'kappa': 1,
    'keynet_weights_path': '/home/a.ivanov/keynet-main/keyNet/weights/KeyNet_default_640_480_YOLO_best:)_0328_173133.log/best_model.pt',

    #'resume_weights_path': "/home/a.ivanov/keynet-main/arg-net/train110/weights/best_combined_model.pth",
    #'resume_weights_path': "/home/a.ivanov/keynet-main/arg-net/train112/weights/best_combined_model.pth",

    'resume_weights_path': "/home/a.ivanov/keynet-main/arg-net/train80/weights/best_combined_model.pth",

    #'resume_weights_path': '/home/a.ivanov/keynet-main/arg-net/train32/weights/best_combined_model.pth',
    #'resume_weights_path': "/home/a.ivanov/keynet-main/arg-net/train80/weights/best_combined_model.pth",
    'path_file_train': '/home/a.ivanov/keynet-main/datasets/synth_10_small_aug(10angle)/train_data.txt',
    #'path_file_train': '/home/a.ivanov/keynet-main/datasets/synth_10_small_aug(10angle)/train_data_norm_HOUGH.txt',
    'path_file_val': '/home/a.ivanov/keynet-main/datasets/synth_10_small_aug(10angle)/val_data_SMALL.txt',
    #'path_file_val': '/home/a.ivanov/keynet-main/datasets/synth_10_small_aug(10angle)/val_data_norm_HOUGH.txt',
    'transformations': "Resize((64, 512)), ToTensor(), Normalize(ImageNet)",
    'path_file_other' : "../normalized_all_Norm_Hough_YOLO_2"
}

device = torch.device(args_dict['device'])

run_dir = get_next_run_dir()
tracker_save_dir = os.path.join(run_dir, "kappa_evolution")
eer_save_dir = os.path.join(run_dir, "eer_plots")
model_save_dir = os.path.join(run_dir, "weights")

os.makedirs(tracker_save_dir, exist_ok=True)
os.makedirs(eer_save_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

print(f"\n============================================")
print(f"🚀 ЗАПУСК СОХРАНЯЕТСЯ В ДИРЕКТОРИЮ: {run_dir}")
print(f"============================================\n")

with open(os.path.join(run_dir, "args.yaml"), "w", encoding="utf-8") as f:
    yaml.dump(args_dict, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

csv_path = os.path.join(run_dir, "results.csv")
with open(csv_path, "w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train_Loss", "Train_Acc", "Train_EER", "Val_EER"])

transform = transforms.Compose([
    transforms.Resize((64, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


transform_vis = transforms.Compose([
    transforms.Resize((64, 512)),
    transforms.ToTensor(),
    # БЕЗ Normalize — только для визуализации!
])
# ==========================================
# 9. ПОДГОТОВКА ДАННЫХ
# ==========================================
print("--- Инициализация Train Датасета ---")
train_dataset = IrisDataset(txt_file=args_dict['path_file_train'], transform=transform, masks_dir="../normalized_all_Norm_Hough_YOLO_2")

# 🔥 МАГИЯ PK SAMPLER 🔥
# Нам нужно по 4 картинки на человека. Если batch_size=64, значит P=16 (16 * 4 = 64)
K = 4
P = args_dict['batch_size'] // K
train_sampler = PKBatchSampler(train_dataset.labels, p=P, k=K)

# ВНИМАНИЕ: Убрали batch_size и shuffle, вместо них передаем batch_sampler!
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)

#train_loader = DataLoader(train_dataset, batch_size=args_dict['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

print("--- Инициализация Train/Val Eval Датасетов ---")
train_eval_dataset = IrisDataset(txt_file=args_dict['path_file_train'], transform=transform, masks_dir="../normalized_all_Norm_Hough_YOLO_2")
train_eval_loader = DataLoader(train_eval_dataset, batch_size=args_dict['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

val_dataset = IrisDataset(txt_file=args_dict['path_file_val'], transform=transform, masks_dir="../normalized_all_Norm_Hough_YOLO_2")
val_loader = DataLoader(val_dataset, batch_size=args_dict['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

# ==========================================
# 10. ИНИЦИАЛИЗАЦИЯ СЕТИ И ЛОССОВ
# ==========================================
print("\nЗагрузка KeyNet...")
prog_args = config.get_config()
MSIP_sizes = [8, 16, 24, 32, 40]
pretrained_keynet = keynet(prog_args, device, MSIP_sizes).to(device)
pretrained_keynet.load_state_dict(torch.load(args_dict['keynet_weights_path'], map_location=device))

backbone = IrisIdentificationNet(keynet_model=pretrained_keynet, kappa=args_dict['kappa'],
                                 batch_size_loc=args_dict['batch_size'], device_loc=device).to(device)

dummy = torch.randn(1, 3, 64, 512).to(device)
with torch.no_grad(): embedding_dim = backbone(dummy).shape[-1]
num_classes = len(train_dataset.class_to_idx)
model = CombinedIrisModel(backbone, embedding_dim, num_classes,
                          s=args_dict['arcface_s'], m=args_dict['arcface_m']).to(device)

# ==========================================
# 🔥 ЗАГРУЗКА ПРЕДОБУЧЕННЫХ ВЕСОВ ДЛЯ FINE-TUNING 🔥
# ==========================================
if 'resume_weights_path' in args_dict and os.path.exists(args_dict['resume_weights_path']):
    print(f"\n🔄 ВОЗОБНОВЛЕНИЕ ОБУЧЕНИЯ: Загрузка весов из {args_dict['resume_weights_path']}")
    try:
        # 💥 ДОБАВИЛИ strict=False 💥
        # Это разрешает загружать только совпадающие ключи и игнорировать недостающие/лишние
        state_dict = torch.load(args_dict['resume_weights_path'], map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Веса успешно загружены (strict=False)! Начинаем Fine-Tuning.")
    except Exception as e:
        print(f"❌ Ошибка загрузки весов: {e}")
        exit()
else:
    print("\n⚠️ Базовые веса не найдены, обучение начнется с нуля!")
# ==========================================


# НОВЫЙ Triplet Loss с Online Hard Mining + Weight Decay
triplet_criterion = OnlineHardTripletLoss(
    margin=args_dict['triplet_margin'],
    min_samples_per_class=4   # ← теперь явно
)

circle_criterion = CircleLoss(
    m=args_dict['circle_m'],
    gamma=args_dict['circle_gamma']
)

ce_criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=args_dict['learning_rate'], weight_decay=1e-4)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args_dict['num_epochs'], eta_min=1e-8)

# ==========================================
# ПОДГОТОВКА ТРЕКЕРОВ (Без изменений)
# ==========================================
# ==========================================
# ПОДГОТОВКА ТРЕКЕРОВ (ИСПРАВЛЕНО)
# ==========================================
val_class_to_paths = defaultdict(list)
with open(args_dict['path_file_val'], 'r') as f:
    eval_paths = [line.strip() for line in f if line.strip()]
for path in eval_paths:
    parts = os.path.basename(path).split('_')
    if len(parts) >= 2:
        val_class_to_paths[f"{parts[0]}_{parts[1]}"].append(path)

tracked_classes = random.sample(list(val_class_to_paths.keys()), min(3, len(val_class_to_paths)))
tracker_images = []  # общий список для обеих аудиторий

# --- ВАЛИДАЦИОННЫЕ ИЗОБРАЖЕНИЯ (исправлено) ---
for cls in tracked_classes:
    os.makedirs(os.path.join(tracker_save_dir, str(cls)), exist_ok=True)
    for path in val_class_to_paths[cls]:
        # Изображение
        pil_img = Image.open(path).convert('RGB')
        tensor_img = transform(pil_img).unsqueeze(0).to(device)

        # Маска
        mask_path = os.path.join("../normalized_all_Norm_Hough_YOLO_2", os.path.basename(path))
        if not os.path.exists(mask_path):
            mask_path = mask_path.replace('.jpg', '_mask.jpg')
        pil_mask = Image.open(mask_path).convert('L').resize((512, 64), Image.NEAREST)
        tensor_mask = torch.from_numpy(np.array(pil_mask)).float().unsqueeze(0).to(device) / 255.0

        tracker_images.append({
            "class_id": cls,
            "filename": os.path.splitext(os.path.basename(path))[0],
            "tensor": tensor_img,   # ← добавлен ключ "tensor"
            "tensor_vis": transform_vis(pil_img).unsqueeze(0).to(device),
            "mask": tensor_mask
        })

# --- ДРУГАЯ АУДИТОРИЯ ---

# other_images_dir = "../normalized_all_Norm_Hough_YOLO_2/"  # папка с другими изображениями
# other_class_to_paths = defaultdict(list)
# for cls in tracked_classes:
#     for val_path in val_class_to_paths[cls]:
#         fname = os.path.basename(val_path)
#         other_path = os.path.join(other_images_dir, fname)
#         if os.path.exists(other_path):
#             other_class_to_paths[cls].append(other_path)

# for cls in tracked_classes:
#     if cls not in other_class_to_paths:
#         print(f"Предупреждение: класс {cls} отсутствует в другой аудитории")
#         continue
#     os.makedirs(os.path.join(tracker_save_dir, str(cls)), exist_ok=True)
#     for path in other_class_to_paths[cls]:
#         pil_img = Image.open(path).convert('RGB')
#         tensor_img = transform(pil_img).unsqueeze(0).to(device)

#         # Маска (из той же папки, что и для валидации)
#         mask_path = os.path.join("../normalized_all_Norm_Hough_YOLO_2", os.path.basename(path))
#         if not os.path.exists(mask_path):
#             mask_path = mask_path.replace('.jpg', '_mask.jpg')
#         pil_mask = Image.open(mask_path).convert('L').resize((512, 64), Image.NEAREST)
#         tensor_mask = torch.from_numpy(np.array(pil_mask)).float().unsqueeze(0).to(device) / 255.0

#         tracker_images.append({
#             "class_id": cls,
#             "filename": os.path.splitext(os.path.basename(path))[0],
#             "tensor": tensor_img,
#             "tensor_vis": transform_vis(pil_img).unsqueeze(0).to(device),
#             "mask": tensor_mask,
#             "source": "other"   # опционально
#         })

# ==========================================
# 13. ОБУЧЕНИЕ
# ==========================================
best_val_eer = 1.0
train_losses = []
train_eers_tuples = [] # Формат: [(epoch, eer)]
val_eers_tuples = []


model.eval()
with torch.no_grad():
    #print("He")
    for tracker in tracker_images:
        #img = apply_vertical_fade(tracker["tensor"]) # Рисуем уже поверх Fade
        img = tracker["tensor_vis"]

        mask = tracker["mask"]
        SRs, region_counts, keypoints = backbone.get_attention_data(tracker["tensor"], mask)
        save_path = os.path.join(tracker_save_dir, str(tracker["class_id"]), f"epoch_{0+1:03d}_{tracker['filename']}.png")
        visualize_kappa_regions(img, SRs, keypoints, region_counts, save_path)
        #cv2.imwrite(img, save_path)


print("  Evaluating Val EER... [-1]")
val_embeddings, val_labels_np, val_paths = compute_embeddings(model, val_loader, device)
v_eer, eer_threshold = compute_eer(val_embeddings, val_labels_np, save_plot=True, save_dir=eer_save_dir, epoch=0, prefix="val")

save_hard_examples(
    val_embeddings,
    val_labels_np,
    val_paths,
    eer_threshold,
    eer_save_dir,
    prefix=f"epoch_{0}"
)
print(f"    Val EER:   {v_eer*100:.3f}%")

for epoch in range(args_dict['num_epochs']):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args_dict['num_epochs']}]", leave=False)

    # 💥 ТОЛЬКО 1 КАРТИНКА И 1 МЕТКА НА ВХОДЕ В БАТЧ
    for images, labels, masks in train_pbar:
        images, labels, masks = images.to(device).float(), labels.to(device), masks.to(device)

        # Защита от переобучения на черных краях!
        #images = apply_vertical_fade(images)

        optimizer.zero_grad()

        # Получаем и логиты, и эмбеддинги за 1 прогон!
        logits, embeddings = model(images, labels, masks=masks)

        #embeddings = F.normalize(embeddings, p=2, dim=1)
        # Считаем ArcFace и Online Hard Triplet Loss
        loss_arc = ce_criterion(logits, labels)
        loss_triplet = triplet_criterion(embeddings, labels)
        loss_circle  = circle_criterion(embeddings, labels)

        loss = (args_dict['arcface_weight'] * loss_arc
            + args_dict['triplet_weight'] * loss_triplet
            + args_dict['circle_weight']  * loss_circle)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        train_pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'arc':  f'{loss_arc.item():.3f}',
            'tri':  f'{loss_triplet.item():.3f}',
            'cir':  f'{loss_circle.item():.3f}',   # ← для мониторинга
            'acc':  f'{correct/total:.3f}'
        })

    avg_train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(avg_train_loss)

    print(f"Epoch [{epoch+1}/{args_dict['num_epochs']}] Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")

    # ---- ВЫЧИСЛЕНИЕ EER ----
    current_train_eer = ""
    current_val_eer = ""

    eer_loss = 0

    # TRAIN EER (Каждые 10 эпох)
    if (epoch + 1) % args_dict['eval_train_every'] == 0:
        print("  Evaluating Train EER...")
        train_embeddings, train_labels_np, train_paths = compute_embeddings(model, train_eval_loader, device)
        t_eer, train_threshold = compute_eer(train_embeddings, train_labels_np, save_plot=True, save_dir=eer_save_dir, epoch=epoch+1, prefix="train")
        save_hard_examples(
            train_embeddings,
            train_labels_np,
            train_paths,
            train_threshold,
            eer_save_dir,
            prefix=f"epoch_{0}_train"
        )

        if t_eer is not None:
            train_eers_tuples.append((epoch + 1, t_eer))
            current_train_eer = f"{t_eer:.6f}"
            print(f"    Train EER: {t_eer*100:.3f}%")


    # VAL EER (Каждую эпоху)
    if (epoch + 1) % args_dict['eval_val_every'] == 0:
        print("  Evaluating Val EER...")
        val_embeddings, val_labels_np, val_paths = compute_embeddings(model, val_loader, device)
        v_eer, eer_threshold = compute_eer(val_embeddings, val_labels_np, save_plot=True, save_dir=eer_save_dir, epoch=epoch+1, prefix="val")
        save_hard_examples(
            val_embeddings,
            val_labels_np,
            val_paths,
            eer_threshold,
            eer_save_dir,
            prefix=f"epoch_{epoch+1}"
        )
        if v_eer is not None:
            val_eers_tuples.append((epoch + 1, v_eer))
            current_val_eer = f"{v_eer:.6f}"
            print(f"    Val EER:   {v_eer*100:.3f}%")

            if v_eer < best_val_eer:
                best_val_eer = v_eer
                torch.save(model.state_dict(), os.path.join(model_save_dir, 'best_combined_model.pth'))
                print(f"    🔥 Сохранена лучшая модель (Val EER: {best_val_eer*100:.3f}%)")
            eer_loss = v_eer

    # ---- ЗАПИСЬ СТАТИСТИКИ ----
    with open(csv_path, "a", newline='', encoding="utf-8") as f:
        csv.writer(f).writerow([epoch + 1, f"{avg_train_loss:.6f}", f"{train_acc:.6f}", current_train_eer, current_val_eer])

    # ---- ВИЗУАЛИЗАЦИЯ KAPPA СЕТОК ----
    model.eval()
    with torch.no_grad():
        if epoch <= 1:
            for tracker in tracker_images:
                #img = apply_vertical_fade(tracker["tensor"]) # Рисуем уже поверх Fade
                img = tracker["tensor_vis"]
                mask = tracker["mask"]
                SRs, region_counts, keypoints = backbone.get_attention_data(tracker["tensor"], mask)
                save_path = os.path.join(tracker_save_dir, str(tracker["class_id"]), f"epoch_{epoch+1:03d}_{tracker['filename']}.png")
                visualize_kappa_regions(img, SRs, keypoints, region_counts, save_path)

    # ---- ГРАФИКИ -----
    plot_training_history(train_losses, train_eers_tuples, val_eers_tuples, os.path.join(run_dir, 'training_history.png'))
    scheduler.step(eer_loss)

torch.save(model.state_dict(), os.path.join(model_save_dir, 'last_combined_model.pth'))
print(f"✅ Обучение завершено! Все данные и графики сохранены в папку {run_dir}")
