"""HardNet: считаем пропуски, а лучшие веса выбираем по image-pair EER."""

import csv
import math
from pathlib import Path

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

import torch
from torch.nn import functional as F

from . import config
from .data import get_split, training_batches
from .evaluate import calculate_scores, extract_features
from .helpers import Features, augment_patches, setup
from .keypoints import extract_patches
from .metrics import error_rates, verification_metrics
from .models import HardNet, hardnet_loss


def validation_positive(patches):
    """Фиксированный поворот на 3 градуса. Случайности в validation нет."""
    angle = math.radians(3)
    matrix = patches.new_tensor([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle), math.cos(angle), 0],
    ])
    matrices = matrix[None].expand(len(patches), -1, -1)
    grid = F.affine_grid(matrices, patches.shape, align_corners=False)
    return F.grid_sample(
        patches, grid, padding_mode="reflection", align_corners=False
    )


def make_batch(rows, features, training):
    patches, labels = [], []
    skipped_images = 0
    for row in rows:
        image, _, points, _ = features.load(row)
        if len(points) == 0:
            skipped_images += 1
            continue
        index = int(torch.randint(len(points), (1,))) if training else 0
        patch = extract_patches(
            image, points[index:index + 1], config.PATCH_SIZE
        )[0]
        patches.append(patch)
        labels.append(row["label"])
    if not patches:
        return None, None, None, skipped_images
    anchor = torch.stack(patches).to(config.DEVICE)
    if training:
        positive = augment_patches(anchor)
        anchor = augment_patches(anchor)
    else:
        positive = validation_positive(anchor)
    labels = torch.tensor(labels, device=config.DEVICE)
    return anchor, positive, labels, skipped_images


def run_epoch(model, rows, features, seed, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, samples = 0.0, 0
    used_batches, skipped_batches, skipped_images = 0, 0, 0
    for batch in training_batches(rows, seed):
        anchor, positive, labels, missing = make_batch(
            batch, features, training
        )
        skipped_images += missing
        if anchor is None or len(torch.unique(labels)) < 2:
            skipped_batches += 1
            continue
        with torch.set_grad_enabled(training):
            first, second = model(torch.cat([anchor, positive])).chunk(2)
            loss = hardnet_loss(first, second, labels, margin=1.0)
            if not torch.isfinite(loss):
                raise ValueError("Loss HardNet стал NaN или inf")
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * len(labels)
        samples += len(labels)
        used_batches += 1
    if samples == 0:
        raise ValueError(
            "Все батчи пропущены. Проверьте точки KeyNet и классы. "
            "Пустую эпоху нельзя считать эпохой с loss=0."
        )
    return {
        "loss": loss_sum / samples,
        "used_batches": used_batches,
        "skipped_batches": skipped_batches,
        "skipped_images": skipped_images,
    }


def plot_eer(scores, output_dir, epoch):
    """Сохраняем распределения расстояний и FAR/FRR в один PNG.

    Гистограммы нормированы отдельно: сумма долей каждого класса — 100%.
    Поэтому разное число своих и чужих пар не искажает сравнение.
    FAR/FRR считаются по исходным расстояниям, не по столбцам гистограммы.
    """
    labels = np.array([row["label"] for row in scores], dtype=int)
    distances = np.array([row["hardnet_distance"] for row in scores])
    report = verification_metrics(labels, distances)
    thresholds, far, frr = error_rates(labels, distances)
    threshold = report["discrete_threshold"]
    eer_percent = report["eer_percent"]
    far_percent = 100 * report["far_at_discrete_threshold"]
    frr_percent = 100 * report["frr_at_discrete_threshold"]
    genuine = distances[labels == 1]
    impostor = distances[labels == 0]

    # Для exp(-alpha * matches) диапазон всегда [0, 1].
    # Одинаковые границы и интервалы помогают сравнивать разные эпохи.
    left = min(0.0, float(distances.min()))
    right = max(1.0, float(distances.max()))
    bins = np.linspace(left, right, 41)
    genuine_hist, _ = np.histogram(genuine, bins=bins)
    impostor_hist, _ = np.histogram(impostor, bins=bins)
    genuine_percent = 100 * genuine_hist / len(genuine)
    impostor_percent = 100 * impostor_hist / len(impostor)
    blue, orange, ink = "#2563a6", "#df7540", "#233044"
    stage = "До обучения" if epoch == 0 else f"Эпоха {epoch}"
    folder = Path(output_dir) / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"eer_epoch_{epoch:03d}.png"

    style = {
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelcolor": ink, "text.color": ink,
        "xtick.color": ink, "ytick.color": ink,
        "axes.edgecolor": "#cbd5e1", "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white",
    }
    with rc_context(style):
        # Canvas Agg работает на сервере без окна и без DISPLAY.
        figure = Figure(figsize=(14, 6), facecolor="white")
        FigureCanvasAgg(figure)
        distribution, curves = figure.subplots(1, 2)
        figure.subplots_adjust(
            left=0.065, right=0.975, bottom=0.22, top=0.77, wspace=0.25
        )
        figure.suptitle(
            f"HardNet · validation · {stage} · EER {eer_percent:.3f}%",
            x=0.065, y=0.96, ha="left", fontsize=19, fontweight="bold",
        )
        figure.text(
            0.065, 0.895,
            f"Пар одной радужки: {len(genuine):,}   |   "
            f"Пар разных радужек: {len(impostor):,}   |   "
            "Меньше расстояние — больше сходство",
            color="#64748b", fontsize=10,
        )
        distribution.stairs(
            genuine_percent, bins, fill=True, alpha=0.22, color=blue
        )
        distribution.stairs(
            impostor_percent, bins, fill=True, alpha=0.22, color=orange
        )
        distribution.stairs(
            genuine_percent, bins, color=blue, linewidth=1.8,
            label="Одна радужка",
        )
        distribution.stairs(
            impostor_percent, bins, color=orange, linewidth=1.8,
            label="Разные радужки",
        )
        distribution.axvline(
            threshold, color=ink, linestyle="--", linewidth=1.3,
            label=f"Порог τ = {threshold:.4f}",
        )
        distribution.set_title("Распределения расстояний", loc="left", pad=12)
        distribution.set_xlabel("Расстояние между изображениями")
        distribution.set_ylabel("Доля пар в интервале")
        distribution.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        distribution.legend(frameon=False, fontsize=9, loc="upper right")
        distribution.set_ylim(bottom=0)

        # Ступени отражают правило принятия distance <= threshold,
        # включая все пары с одинаковым расстоянием одновременно.
        curves.step(
            thresholds, 100 * far, where="post", color=orange,
            linewidth=2, label="FAR — ложный допуск",
        )
        curves.step(
            thresholds, 100 * frr, where="post", color=blue,
            linewidth=2, label="FRR — ложный отказ",
        )
        curves.axvline(threshold, color=ink, linestyle="--", linewidth=1.3)
        curves.axhline(
            eer_percent, color="#64748b", linestyle=":", linewidth=1.3,
            label=f"EER = {eer_percent:.3f}% (интерполяция)",
        )
        curves.scatter(
            [threshold, threshold], [far_percent, frr_percent],
            color=[orange, blue], s=42, edgecolors="white", zorder=5,
        )
        curves.set_title("Ошибки при изменении порога", loc="left", pad=12)
        curves.set_xlabel("Порог принятия пары, τ")
        curves.set_ylabel("Доля ошибок")
        curves.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        curves.set_ylim(-2, 102)
        curves.legend(frameon=False, fontsize=9, loc="upper center")
        for axis in (distribution, curves):
            axis.set_xlim(left, right)
            axis.grid(axis="y", color="#e2e8f0", linewidth=0.7)
            axis.set_axisbelow(True)

        figure.text(
            0.065, 0.105,
            f"При τ = {threshold:.4f}:  FAR = {far_percent:.3f}%   |   "
            f"FRR = {frr_percent:.3f}%",
            fontsize=12, fontweight="bold",
        )
        figure.text(
            0.065, 0.055,
            "FAR: чужие пары с distance ≤ τ.  "
            "FRR: свои пары с distance > τ.\n"
            "τ — ближайший дискретный порог; FAR(τ) и FRR(τ) "
            "могут отличаться от интерполированного EER.",
            color="#64748b", fontsize=9,
        )
        try:
            figure.savefig(path, dpi=180)
        finally:
            figure.clear()
    print(f"График EER: {path}", flush=True)
    return report


@torch.inference_mode()
def validation_scores(model, rows, output_dir, epoch):
    """Та же оценка пар изображений, что в recognition.evaluate."""
    model.eval()
    local, _ = extract_features(rows, model, None)
    scores = calculate_scores(rows, local, [])
    report = plot_eer(scores, output_dir, epoch)
    return report["eer"], scores


def main():
    setup()
    # Меняем размер батча только в этом процессе обучения HardNet.
    config.IDENTITIES_PER_BATCH = getattr(
        config, "HARDNET_IDENTITIES_PER_BATCH", 16
    )
    config.IMAGES_PER_IDENTITY = 2
    eer_every = getattr(config, "HARDNET_EER_EVERY", 5)
    if eer_every < 1 or config.EPOCHS < 1:
        raise ValueError("EPOCHS и HARDNET_EER_EVERY должны быть >= 1")
    output = config.RUNS_DIR / "hardnet"
    if (output / "best.pt").exists() or (output / "last.pt").exists():
        raise ValueError("Выберите новый RUNS_DIR: здесь уже есть веса")
    train_rows = get_split("train")
    val_rows = get_split("val")
    features = Features()
    model = HardNet().to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # Scheduler получает EER только после полной проверки изображений.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )
    output.mkdir(parents=True, exist_ok=True)

    def save_best(scores):
        torch.save(model.state_dict(), output / "best.pt")
        with (output / "best_val_scores.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(scores[0]))
            writer.writeheader()
            writer.writerows(scores)

    torch.save(model.state_dict(), output / "initial.pt")
    print("Считаем исходный EER до обучения...", flush=True)
    best_eer, scores = validation_scores(model, val_rows, output, epoch=0)
    best_epoch = 0
    save_best(scores)
    scheduler.step(best_eer)
    print(f"До обучения: EER={100 * best_eer:.3f}%", flush=True)
    with (output / "history.csv").open("w", newline="") as stream:
        fields = [
            "epoch", "train_loss", "val_loss", "val_eer", "lr_used",
            "next_lr", "train_batches", "train_skipped_batches",
            "train_skipped_images", "val_batches", "val_skipped_batches",
            "val_skipped_images",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"epoch": 0, "val_eer": best_eer})
        for epoch in range(1, config.EPOCHS + 1):
            lr_used = optimizer.param_groups[0]["lr"]
            train = run_epoch(
                model, train_rows, features, config.SEED + epoch, optimizer
            )
            val = run_epoch(model, val_rows, features, config.SEED)
            eer = None
            if epoch % eer_every == 0 or epoch == config.EPOCHS:
                eer, scores = validation_scores(model, val_rows, output, epoch)
                scheduler.step(eer)
                if eer < best_eer:
                    best_eer, best_epoch = eer, epoch
                    save_best(scores)
            torch.save(model.state_dict(), output / "last.pt")
            next_lr = optimizer.param_groups[0]["lr"]
            writer.writerow({
                "epoch": epoch, "train_loss": train["loss"],
                "val_loss": val["loss"], "val_eer": eer,
                "lr_used": lr_used, "next_lr": next_lr,
                "train_batches": train["used_batches"],
                "train_skipped_batches": train["skipped_batches"],
                "train_skipped_images": train["skipped_images"],
                "val_batches": val["used_batches"],
                "val_skipped_batches": val["skipped_batches"],
                "val_skipped_images": val["skipped_images"],
            })
            stream.flush()
            eer_text = (
                f"{100 * eer:.3f}%" if eer is not None else "не считался"
            )
            print(
                f"Эпоха {epoch}: train={train['loss']:.4f}, "
                f"val={val['loss']:.4f}, EER={eer_text}, "
                f"lr={lr_used:.2e}, next_lr={next_lr:.2e}; "
                f"батчей={train['used_batches']}, "
                f"пропущено батчей={train['skipped_batches']}, "
                f"снимков={train['skipped_images']}", flush=True,
            )
    print(f"Лучший EER={100 * best_eer:.3f}%, эпоха={best_epoch}")


if __name__ == "__main__":
    main()
