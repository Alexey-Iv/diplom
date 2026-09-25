"""Сравнение изображений и EER для HardNet, AG-Net и их гибрида."""

import argparse
import csv
import json

import numpy as np
import torch

from . import config
from .data import get_split
from .helpers import Features, agnet_batch, pair_indices, setup
from .keypoints import extract_patches, match_count
from .metrics import verification_metrics
from .models import AGNet, HardNet


def load_models(name):
    hardnet, agnet = None, None
    if name in {"hardnet", "hybrid"}:
        hardnet = HardNet().to(config.DEVICE)
        weights = torch.load(
            config.RUNS_DIR / "hardnet/best.pt",
            map_location="cpu", weights_only=True,
        )
        hardnet.load_state_dict(weights)
        hardnet.eval()
    if name in {"agnet", "hybrid"}:
        saved = torch.load(
            config.RUNS_DIR / "agnet/best.pt",
            map_location="cpu", weights_only=True,
        )
        agnet = AGNet(saved["embedding_dim"], saved["channels"])
        agnet.load_state_dict(saved["model"])
        agnet.to(config.DEVICE).eval()
    return hardnet, agnet


@torch.inference_mode()
def extract_features(rows, hardnet, agnet):
    reader = Features(use_regions=agnet is not None)
    local, global_vectors = [], []
    for index, row in enumerate(rows):
        image, _, points, _ = reader.load(row)
        if hardnet is not None:
            patches = extract_patches(image, points, config.PATCH_SIZE)
            if len(patches):
                vectors = [
                    hardnet(batch.to(config.DEVICE)).cpu()
                    for batch in patches.split(128)
                ]
                local.append(torch.cat(vectors))
            else:
                local.append(torch.empty(0, 128))
        if agnet is not None:
            images, boxes = agnet_batch([row], reader)
            global_vectors.append(agnet(images, boxes)[0].cpu())
        print(f"Признаки: {index + 1}/{len(rows)}", flush=True)
    return local, global_vectors


def calculate_scores(rows, local, global_vectors):
    pairs, labels = pair_indices(
        [row["label"] for row in rows],
        config.MAX_IMPOSTOR_PAIRS, config.SEED,
    )
    scores = []
    for (first, second), label in zip(pairs, labels):
        result = {
            "path_a": rows[first]["path"], "path_b": rows[second]["path"],
            "label": int(label),
        }
        if local:
            matches = match_count(
                local[first], local[second], config.MATCH_THRESHOLD,
                config.RATIO,
            )
            result["matches"] = matches
            result["hardnet_distance"] = float(np.exp(
                -config.HYBRID_ALPHA * matches
            ))
        if global_vectors:
            result["agnet_distance"] = float(torch.linalg.vector_norm(
                global_vectors[first] - global_vectors[second]
            ))
        if local and global_vectors:
            result["hybrid_distance"] = (
                config.HYBRID_WEIGHT * result["agnet_distance"]
                + (1 - config.HYBRID_WEIGHT) * result["hardnet_distance"]
            )
        scores.append(result)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["hardnet", "agnet", "hybrid"], default="hybrid"
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()
    if not 0 <= config.HYBRID_WEIGHT <= 1 or config.HYBRID_ALPHA <= 0:
        raise ValueError("Нужно 0 <= HYBRID_WEIGHT <= 1 и HYBRID_ALPHA > 0")
    setup()
    rows = get_split(args.split)
    hardnet, agnet = load_models(args.model)
    local, global_vectors = extract_features(rows, hardnet, agnet)
    scores = calculate_scores(rows, local, global_vectors)
    output = config.RUNS_DIR / "evaluation" / args.model
    output.mkdir(parents=True, exist_ok=True)
    scores_path = output / f"{args.split}_scores.csv"
    with scores_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    threshold_path = output / "thresholds.json"
    thresholds = {}
    if args.split == "test" and threshold_path.exists():
        thresholds = json.loads(threshold_path.read_text())
    report = {}
    labels = [row["label"] for row in scores]
    for name in ("hardnet", "agnet", "hybrid"):
        column = f"{name}_distance"
        if column not in scores[0]:
            continue
        distances = [row[column] for row in scores]
        report[name] = verification_metrics(
            labels, distances, threshold=thresholds.get(name)
        )
        if args.split == "val":
            thresholds[name] = report[name]["discrete_threshold"]
        print(f"{name}: EER = {report[name]['eer_percent']:.3f}%")
    if args.split == "val":
        threshold_path.write_text(json.dumps(thresholds, indent=2))
    (output / f"{args.split}_metrics.json").write_text(
        json.dumps(report, indent=2)
    )
    print(f"Расстояния: {scores_path}")


if __name__ == "__main__":
    main()
