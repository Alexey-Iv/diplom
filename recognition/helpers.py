"""Reproducible image pairs and shared preprocessing for train/evaluation."""

from itertools import combinations

import numpy as np
import torch
from torch.nn import functional as F

from . import config
from .data import read_image
from .keypoints import KeyNetDetector, semantic_regions


def pair_indices(labels, max_impostors=100000, seed=42):
    """All genuine pairs and a uniform sample of unique impostor pairs.

    Sampling uses image pairs, not class-balanced impostor pairs. The seed
    and limit are set in config.py. No self-pairs are included.
    """
    labels = np.asarray(labels)
    if max_impostors < 1:
        raise ValueError("max_impostors must be positive")
    genuine = []
    for label in np.unique(labels):
        genuine.extend(combinations(np.flatnonzero(labels == label), 2))
    count = len(labels)
    total = count * (count - 1) // 2 - len(genuine)
    if not genuine or total < 1:
        raise ValueError("Need genuine and impostor pairs")
    requested = min(total, max_impostors)
    if total <= max_impostors * 2:
        impostors = [
            (i, j) for i in range(count) for j in range(i + 1, count)
            if labels[i] != labels[j]
        ]
        if len(impostors) > requested:
            rng = np.random.default_rng(seed)
            selected = rng.choice(len(impostors), requested, replace=False)
            impostors = [impostors[int(i)] for i in selected]
    else:
        rng = np.random.default_rng(seed)
        chosen = set()
        while len(chosen) < requested:
            candidates = rng.integers(0, count, size=(4096, 2))
            for first, second in candidates:
                if labels[first] != labels[second]:
                    chosen.add(tuple(sorted((int(first), int(second)))))
                if len(chosen) == requested:
                    break
        impostors = sorted(chosen)
    pairs = np.asarray(genuine + impostors, dtype=np.int64)
    targets = np.r_[
        np.ones(len(genuine)), np.zeros(len(impostors))
    ].astype(int)
    return pairs, targets


def global_image(image, mask, size):
    """Neutral fill for invalid pixels, then fixed rectangular resizing."""
    filled = image * mask + 0.5 * (1 - mask)
    return F.interpolate(
        filled[None], size=size, mode="bilinear", align_corners=False,
        antialias=True,
    )[0]


def augment_patches(patches):
    """Independently rotate and change brightness/noise in each patch."""
    count = len(patches)
    angles = (torch.rand(count, device=patches.device) - 0.5) * 0.10
    matrices = patches.new_zeros(count, 2, 3)
    matrices[:, 0, 0] = torch.cos(angles)
    matrices[:, 0, 1] = -torch.sin(angles)
    matrices[:, 1, 0] = torch.sin(angles)
    matrices[:, 1, 1] = torch.cos(angles)
    grid = F.affine_grid(matrices, patches.shape, align_corners=False)
    result = F.grid_sample(
        patches, grid, padding_mode="reflection", align_corners=False
    )
    gain = 0.85 + 0.3 * torch.rand(count, 1, 1, 1, device=patches.device)
    noise = torch.randn_like(result) * 0.01
    return (result * gain + noise).clamp(0, 1)


def setup():
    """Повторяемые начальные веса и число потоков CPU."""

    torch.set_num_threads(config.THREADS)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)


class Features:
    """Точки и регионы считаются один раз и остаются в памяти процесса."""

    def __init__(self, use_regions=False):

        self.detector = KeyNetDetector(
            config.KEYNET_DIR, config.KEYNET_WEIGHTS, config.DEVICE
        )
        self.use_regions = use_regions
        self.cache = {}

    def load(self, row):

        image, mask = read_image(row)
        if row["path"] not in self.cache:
            points = self.detector.detect(
                image, mask, config.PATCH_SIZE, config.TOP_K, config.NMS_SIZE
            )
            boxes = None
            if self.use_regions:
                try:
                    boxes = semantic_regions(
                        points.numpy(), *image.shape[-2:],
                        config.KAPPA, config.SEED,
                    )
                except ValueError as error:
                    raise ValueError(f"{row['path']}: {error}") from error
                boxes = torch.from_numpy(boxes)
            self.cache[row["path"]] = points, boxes
        points, boxes = self.cache[row["path"]]
        return image, mask, points, boxes


def agnet_batch(rows, features, augment=False):
    """Готовим изображения и прямоугольники для AG-Net."""

    images, regions = [], []
    for row in rows:
        image, mask, _, boxes = features.load(row)
        if augment:
            gain = float(torch.empty(()).uniform_(0.9, 1.1))
            image = (gain * image + 0.01 * torch.randn_like(image)).clamp(0, 1)
        images.append(global_image(image, mask, config.IMAGE_SIZE))
        regions.append(boxes)
    return (
        torch.stack(images).to(config.DEVICE),
        torch.stack(regions).to(config.DEVICE),
    )


@torch.inference_mode()
def agnet_embeddings(model, rows, features):
    """По одному эмбеддингу на изображение; порядок совпадает с rows."""
    model.eval()
    vectors = []
    for row in rows:
        images, boxes = agnet_batch([row], features)
        vectors.append(model(images, boxes)[0].cpu())
    return torch.stack(vectors)
