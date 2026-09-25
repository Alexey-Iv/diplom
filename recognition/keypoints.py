"""Frozen KeyNet adapter, differentiable patch extraction and matching."""

from itertools import combinations
import sys

import numpy as np
from sklearn.mixture import GaussianMixture
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F


class KeyNetDetector:
    """Use the exact KeyNet code and checkpoint format from this repository."""

    def __init__(self, repository, checkpoint, device):
        repository = Path(repository).resolve()
        if not (repository / "keyNet/model/keynet_architecture.py").is_file():
            raise ValueError("Проверьте путь KEYNET_DIR в config.py")
        sys.path.insert(0, str(repository))
        from checkpoints import initialize
        from keyNet.model.keynet_architecture import keynet
        from train_utils import forward_scores, points

        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format") == "keynet-training-1":
            config = saved["config"]
            self.activation = config["score_activation"]
            model = keynet(SimpleNamespace(**config), torch.device(device))
            model.load_state_dict(saved["model"], strict=True)
        else:
            # Architecture defaults of the supplied legacy 10-channel model.
            config = dict(
                num_filters=8, num_learnable_blocks=3,
                num_levels_within_net=3, factor_scaling_pyramid=1.5,
                conv_kernel_size=5, hermite=True, nms_size=5,
                batch_size=1, patch_size=64,
            )
            model = keynet(SimpleNamespace(**config), torch.device(device))
            state = torch.load(
                checkpoint,
                map_location=device,
                weights_only=True,
            )

            model.learner.load_state_dict(
                state["learner"],
                strict=True,
            )

            model.last_layer_learner.load_state_dict(
                state["last_layer_learner"],
                strict=True,
            )
            
        self.activation = "relu"
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        self.forward_scores = forward_scores
        self.select_points = points
        self.device = device

    @torch.inference_mode()
    def detect(self, image, mask, patch_size=32, topk=200, nms_size=5):
        image = image[None].to(self.device)
        mask = mask[None].to(self.device)
        _, scores = self.forward_scores(
            self.model, image, mask, self.activation
        )
        # Reject a point if any part of its patch touches a masked pixel.
        radius = (patch_size + 1) // 2
        kernel = 2 * radius + 1
        invalid = F.max_pool2d(1 - mask, kernel, 1, radius)
        valid = (invalid == 0).float()[0, 0]
        valid[:radius] = 0
        valid[-radius:] = 0
        valid[:, :radius] = 0
        valid[:, -radius:] = 0
        return self.select_points(
            scores[0, 0], valid, topk, nms_size
        ).cpu()


def extract_patches(image, points, patch_size=32, output_size=32):
    """Extract centered patches; point coordinates are (x, y) in pixels."""
    if len(points) == 0:
        return image.new_empty((0, 1, output_size, output_size))
    height, width = image.shape[-2:]
    offsets = torch.linspace(
        -(patch_size - 1) / 2, (patch_size - 1) / 2, output_size,
        dtype=image.dtype, device=image.device,
    )
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)[None] + points[:, None, None]
    grid = 2 * grid / grid.new_tensor([width - 1, height - 1]) - 1
    return F.grid_sample(
        image[None].expand(len(points), -1, -1, -1), grid,
        align_corners=True, padding_mode="zeros",
    )


def match_count(first, second, threshold=0.8, ratio=0.8):
    """Mutual nearest neighbours, distance threshold and Lowe ratio test."""
    if not 0 < ratio < 1 or threshold <= 0:
        raise ValueError("Need threshold > 0 and 0 < ratio < 1")
    if len(first) < 2 or len(second) < 2:
        return 0
    distances = torch.cdist(first, second)
    values, indices = distances.topk(2, largest=False, dim=1)
    nearest = indices[:, 0]
    reverse = distances.argmin(dim=0)
    mutual = reverse[nearest] == torch.arange(
        len(first), device=first.device
    )
    valid = (
        mutual & (values[:, 0] <= threshold)
        & (values[:, 0] < ratio * values[:, 1])
    )
    return int(valid.sum())


def semantic_regions(points, height, width, kappa=22, seed=42, padding=8):
    """Primary boxes, every pairwise union, and one full-image region."""
    if kappa < 1 or len(points) < kappa:
        raise ValueError(
            f"Need >= kappa={kappa} valid KeyNet points, got {len(points)}. "
            "Inspect masks/detector or explicitly reduce KAPPA."
        )
    coordinates = np.asarray(points, dtype=np.float64)
    normalized = coordinates / [width, height]
    mixture = GaussianMixture(
        n_components=kappa, covariance_type="full", random_state=seed,
        reg_covar=1e-5, n_init=1,
    )
    assignment = mixture.fit_predict(normalized)
    if not mixture.converged_:
        raise ValueError("GMM did not converge; inspect detected points")
    boxes = []
    for cluster in range(kappa):
        members = coordinates[assignment == cluster]
        if not len(members):
            raise ValueError("Empty GMM component; reduce KAPPA")
        minimum = np.maximum(members.min(axis=0) - padding, 0)
        maximum = np.minimum(
            members.max(axis=0) + padding + 1, [width, height]
        )
        boxes.append(np.r_[minimum, maximum])
    primary = list(boxes)
    for first, second in combinations(primary, 2):
        boxes.append(np.r_[
            np.minimum(first[:2], second[:2]),
            np.maximum(first[2:], second[2:]),
        ])
    boxes.append(np.array([0, 0, width, height]))
    return np.asarray(boxes, dtype=np.float32) / np.array(
        [width, height, width, height], dtype=np.float32
    )
