"""MSIP with the archive's exp(score / window_max)-1 proposal.

Visibility handling is repaired.

No spatial-softmax replacement: IP keeps its original positive-map formulation.
Loss normalization is changed to per-image valid windows and documented in README.
"""

import torch
import torch.nn.functional as F

from geometry import erode, warp


def proposals(scores, valid, size):
    height, width = scores.shape[-2:]
    height -= height % size
    width -= width % size

    if height == 0 or width == 0:
        return None

    score_windows = F.unfold(
        scores[..., :height, :width],
        size,
        stride=size,
    )

    mask_windows = F.unfold(
        valid[..., :height, :width],
        size,
        stride=size,
    )

    visible = mask_windows.amin(1) > 0.999
    maximum = score_windows.amax(
        1,
        keepdim=True,
    )

    # expm1 is stable close to zero; scores are nonnegative by contract.
    mass = (
        torch.expm1(
            score_windows
            / (maximum.detach() + 1e-6)
        )
        + 1e-6
    )

    denominator = (
        mass.sum(
            1,
            keepdim=True,
        )
        + 1e-6
    )

    yy, xx = torch.meshgrid(
        torch.arange(
            1,
            size + 1,
            device=score_windows.device,
        ),
        torch.arange(
            1,
            size + 1,
            device=score_windows.device,
        ),
        indexing="ij",
    )

    coordinates = torch.stack(
        (
            yy.flatten(),
            xx.flatten(),
        ),
        dim=1,
    ).to(score_windows.dtype)

    soft = torch.einsum(
        "bpl,pc->blc",
        mass / denominator,
        coordinates,
    )

    hard = coordinates[
        score_windows.detach().argmax(1)
    ]

    confidence = (
        (
            mass * score_windows
        ).sum(1)
        / denominator[:, 0]
    ).detach()

    usable = (
        visible
        & (
            maximum[:, 0].detach()
            > 1e-8
        )
    )

    return (
        soft,
        hard,
        confidence,
        usable,
    )


def one_direction(
    src,
    target,
    valid,
    size,
    coordinate_weighting,
):
    src_proposals = proposals(
        src,
        valid,
        size,
    )

    target_proposals = proposals(
        target.detach(),
        valid,
        size,
    )

    if (
        src_proposals is None
        or target_proposals is None
    ):
        return src.sum() * 0, 0

    soft, _, confidence, good = src_proposals
    good = good & target_proposals[3]

    errors = (
        (
            soft
            - target_proposals[1]
        )
        / size
    ).square().sum(-1)

    rows = good.any(1)

    if not rows.any():
        return src.sum() * 0, 0

    if coordinate_weighting:
        logits = confidence.masked_fill(
            ~good,
            -1e9,
        )

        weights = (
            logits.softmax(1)
            * good
        )

        weights = (
            weights
            / weights.sum(
                1,
                keepdim=True,
            ).clamp_min(1e-8)
        )
    else:
        weights = (
            good.to(errors)
            / good.sum(
                1,
                keepdim=True,
            ).clamp_min(1)
        )

    loss = (
        (
            errors * weights
        ).sum(1)[rows].mean()
        * 1000
    )

    return loss, int(good.sum())


def msip_loss(
    src,
    dst,
    src_mask,
    dst_mask,
    H,
    windows=(8, 16, 24),
    factors=(256.0, 64.0, 16.0),
    coordinate_weighting=True,
):
    if (
        len(windows) != len(factors)
        or not windows
        or any(size < 2 for size in windows)
        or any(weight <= 0 for weight in factors)
    ):
        raise ValueError(
            "Invalid MSIP scales/weights"
        )

    if (
        src.shape != dst.shape
        or src.shape != src_mask.shape
        or dst.shape != dst_mask.shape
    ):
        raise ValueError(
            "Score and mask shapes differ"
        )

    src_eroded = erode(src_mask)
    dst_eroded = erode(dst_mask)

    inverse = torch.linalg.inv(H)

    # Warp FULL maps first; cropping before warping can remove
    # a visible correspondence.
    src_valid = (
        src_eroded
        * (
            warp(
                dst_eroded,
                inverse,
            )
            > 0.999
        )
    )

    dst_valid = (
        dst_eroded
        * (
            warp(
                src_eroded,
                H,
            )
            > 0.999
        )
    )

    src_target = warp(
        dst.detach(),
        inverse,
    )

    dst_target = warp(
        src.detach(),
        H,
    )

    total = src.sum() * 0
    total_weight = 0.0
    diagnostics = {}

    for size, weight in zip(
        windows,
        factors,
    ):
        src_loss, src_count = one_direction(
            src,
            src_target,
            src_valid,
            size,
            coordinate_weighting,
        )

        dst_loss, dst_count = one_direction(
            dst,
            dst_target,
            dst_valid,
            size,
            coordinate_weighting,
        )

        count = src_count + dst_count
        diagnostics[str(size)] = count

        if count:
            # Symmetric directions. Ignore an entirely empty
            # direction rather than diluting it.
            directions = (
                int(src_count > 0)
                + int(dst_count > 0)
            )

            total += (
                weight
                * (
                    src_loss
                    + dst_loss
                )
                / directions
            )

            total_weight += weight

    if not total_weight:
        raise ValueError(
            "No valid MSIP windows: inspect masks, "
            "score activity, crop size and borders"
        )

    return (
        total / total_weight,
        diagnostics,
    )