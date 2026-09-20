import contextlib
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from tqdm import tqdm

from geometry import erode, transform, warp
from keyNet.loss.score_loss_function import msip_loss


def fix_randseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@contextlib.contextmanager
def validation_mode(model):
    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            yield
    finally:
        model.train(was_training)


def forward_scores(model, image, mask, activation):
    # The underlying original architecture expects NHWC, not BCHW.
    image = image * mask + 0.5 * (1 - mask)

    _, raw = model(image.permute(0, 2, 3, 1))

    positive = (
        F.relu(raw)
        if activation == "relu"
        else F.softplus(raw)
    )

    return raw, positive


def move(batch, device):
    return {
        key: value.to(device)
        for key, value in batch.items()
    }


def train_epoch(loader, model, optimizer, args):
    model.train()

    total = 0.0
    n = 0
    active = 0.0
    windows = {str(size): 0 for size in args.windows}

    for step, batch in enumerate(loader):
        if args.max_steps and step >= args.max_steps:
            break

        b = move(batch, args.device)
        optimizer.zero_grad(set_to_none=True)

        raw, a = forward_scores(
            model,
            b["src"],
            b["src_input_mask"],
            args.score_activation,
        )

        _, p = forward_scores(
            model,
            b["dst"],
            b["dst_input_mask"],
            args.score_activation,
        )

        loss, counts = msip_loss(
            a,
            p,
            b["src_mask"],
            b["dst_mask"],
            b["H"],
            args.windows,
            args.factors,
            args.coordinate_weighting,
        )

        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite MSIP loss")

        loss.backward()

        grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
            error_if_nonfinite=True,
        )

        if float(grad) == 0:
            raise RuntimeError(
                "Zero gradient: inspect dead ReLU maps or use an explicit "
                "softplus experiment"
            )

        optimizer.step()

        total += float(loss.detach())
        n += 1

        valid = b["src_mask"] > 0.999

        if valid.any():
            active += float(
                (raw[valid] > 0).float().mean()
            )

        for key, value in counts.items():
            windows[key] += value

    if not n:
        raise ValueError("No training batches")

    return {
        "loss": total / n,
        "positive_logit_fraction": active / n,
        "gradient_norm": float(grad),
        "valid_windows": windows,
        "train_batches": n,
    }


def points(score, mask, k=25, nms=5):
    if k < 1 or nms < 1 or nms % 2 != 1:
        raise ValueError(
            "Need positive top-k and odd NMS size"
        )

    maxima = F.max_pool2d(
        score[None, None],
        nms,
        1,
        nms // 2,
    )[0, 0]

    yy, xx = torch.where(
        (score == maxima)
        & (mask > 0.999)
        & (score > 0)
        & torch.isfinite(score)
    )

    order = torch.argsort(
        score[yy, xx],
        descending=True,
        stable=True,
    )

    accepted = []

    for index in order.tolist():
        point = torch.stack(
            (xx[index], yy[index])
        ).float()

        if accepted:
            distances = (
                torch.stack(accepted) - point
            ).abs().amax(1)

            if (distances <= nms // 2).any():
                continue

        accepted.append(point)

        if len(accepted) == k:
            break

    if accepted:
        return torch.stack(accepted)

    return score.new_empty((0, 2))


def repeatability(a, b, H, tolerance=3.0):
    if min(len(a), len(b)) == 0:
        return 0.0

    mapped = transform(
        a[None],
        H[None],
    )[0]

    edges = (
        torch.cdist(mapped, b) <= tolerance
    ).cpu().numpy()

    match = maximum_bipartite_matching(
        csr_matrix(edges),
        perm_type="column",
    )

    return float(
        (match >= 0).sum()
        / min(len(a), len(b))
    )


def validate(loader, model, args):
    values = []
    counts = []
    losses = []
    empty = 0

    with validation_mode(model):
        for step, batch in enumerate(loader):
            if args.max_steps and step >= args.max_steps:
                break

            b = move(batch, args.device)

            _, a = forward_scores(
                model,
                b["src"],
                b["src_input_mask"],
                args.score_activation,
            )

            _, p = forward_scores(
                model,
                b["dst"],
                b["dst_input_mask"],
                args.score_activation,
            )

            loss, _ = msip_loss(
                a,
                p,
                b["src_mask"],
                b["dst_mask"],
                b["H"],
                args.windows,
                args.factors,
                args.coordinate_weighting,
            )

            losses.extend(
                [float(loss)] * len(b["src"])
            )

            ma = erode(b["src_mask"])
            mb = erode(b["dst_mask"])

            va = ma * (
                warp(
                    mb,
                    torch.linalg.inv(b["H"]),
                )
                > 0.999
            )

            vb = mb * (
                warp(
                    ma,
                    b["H"],
                )
                > 0.999
            )

            for index in range(len(a)):
                aa = points(
                    a[index, 0],
                    va[index, 0],
                    args.topk,
                    args.nms_size,
                )

                bb = points(
                    p[index, 0],
                    vb[index, 0],
                    args.topk,
                    args.nms_size,
                )

                empty += int(
                    min(len(aa), len(bb)) == 0
                )

                counts.extend(
                    [len(aa), len(bb)]
                )

                values.append(
                    repeatability(
                        aa,
                        bb,
                        b["H"][index],
                        args.pixel_threshold,
                    )
                )

    if not values:
        raise ValueError("No validation images")

    return {
        "val_loss": float(np.mean(losses)),
        "val_repeatability_px": float(np.mean(values)),
        "pixel_threshold": args.pixel_threshold,
        "mean_points": float(np.mean(counts)),
        "empty_pairs": empty,
        "val_pairs": len(values),
    }