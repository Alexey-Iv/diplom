"""Обучение AG-Net. Запуск: python -m recognition.train_agnet."""

import csv

import torch
from torch.nn import functional as F

from . import config
from .data import get_split, training_batches
from .helpers import Features, agnet_batch, agnet_embeddings, pair_indices
from .helpers import setup
from .metrics import verification_metrics
from .models import AGNet, ArcFace, batch_hard_triplet


def main():
    setup()
    train_rows = get_split("train")
    val_rows = get_split("val")
    features = Features(use_regions=True)
    model = AGNet(
        config.EMBEDDING_DIM, config.REGION_CHANNELS,
        pretrained=config.PRETRAINED_BACKBONE,
    ).to(config.DEVICE)
    class_count = len({row["label"] for row in train_rows})
    head = ArcFace(config.EMBEDDING_DIM, class_count).to(config.DEVICE)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()),
        lr=config.LEARNING_RATE,
    )
    val_pairs, val_labels = pair_indices(
        [row["label"] for row in val_rows],
        config.MAX_IMPOSTOR_PAIRS, config.SEED,
    )
    output = config.RUNS_DIR / "agnet"
    output.mkdir(parents=True, exist_ok=True)
    best_eer = float("inf")
    with (output / "history.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_loss", "val_eer"])
        for epoch in range(config.EPOCHS):
            model.train()
            head.train()
            losses = []
            for rows in training_batches(train_rows, config.SEED + epoch):
                images, boxes = agnet_batch(rows, features, augment=True)
                labels = torch.tensor(
                    [row["label"] for row in rows], device=config.DEVICE
                )
                embeddings = model(images, boxes)
                arc_loss = F.cross_entropy(head(embeddings, labels), labels)
                triplet_loss = batch_hard_triplet(embeddings, labels)
                loss = (
                    config.ARC_WEIGHT * arc_loss
                    + config.TRIPLET_WEIGHT * triplet_loss
                )
                if not torch.isfinite(loss):
                    raise ValueError("Loss AG-Net стал NaN или inf")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            embeddings = agnet_embeddings(model, val_rows, features)
            distances = torch.linalg.vector_norm(
                embeddings[val_pairs[:, 0]] - embeddings[val_pairs[:, 1]],
                dim=1,
            ).numpy()
            eer = verification_metrics(val_labels, distances)["eer"]
            # Сохраняем веса и два размера, нужных для загрузки модели.
            checkpoint = {
                "model": model.state_dict(),
                "embedding_dim": config.EMBEDDING_DIM,
                "channels": config.REGION_CHANNELS,
            }
            torch.save(checkpoint, output / "last.pt")
            if eer < best_eer:
                best_eer = eer
                torch.save(checkpoint, output / "best.pt")
            train_loss = sum(losses) / len(losses)
            writer.writerow([epoch + 1, train_loss, eer])
            stream.flush()
            print(
                f"Эпоха {epoch + 1}: loss={train_loss:.4f}, "
                f"val EER={100 * eer:.2f}%", flush=True,
            )


if __name__ == "__main__":
    main()
