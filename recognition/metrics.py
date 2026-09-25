"""Verification metrics. A smaller distance means a stronger match."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def error_rates(labels, distances):
    """Return every distinct operating point, keeping tied scores together."""
    labels = np.asarray(labels)
    distances = np.asarray(distances, dtype=np.float64)
    if labels.ndim != 1 or distances.shape != labels.shape:
        raise ValueError("Expected two one-dimensional arrays of equal size")
    if not np.isfinite(distances).all():
        raise ValueError("Distances must be finite")
    valid_labels = np.isin(labels, [0, 1]).all()
    if not valid_labels or len(np.unique(labels)) != 2:
        raise ValueError("Need both genuine (1) and impostor (0) pairs")
    order = np.argsort(distances, kind="stable")
    values = distances[order]
    genuine = labels[order] == 1
    ends = np.r_[np.flatnonzero(np.diff(values)), len(values) - 1]
    true_accepts = np.r_[0, np.cumsum(genuine)[ends]]
    false_accepts = np.r_[0, np.cumsum(~genuine)[ends]]
    thresholds = np.r_[np.nextafter(values[0], -np.inf), values[ends]]
    far = false_accepts / (~genuine).sum()
    frr = 1.0 - true_accepts / genuine.sum()
    return thresholds, far, frr


def verification_metrics(labels, distances, threshold=None):
    """Interpolated EER plus a realizable discrete operating threshold.

    Interpolation is on the FAR/FRR curve, not on the score axis. In a
    finite sample, a threshold achieving exactly the EER may not exist.
    """
    thresholds, far, frr = error_rates(labels, distances)
    difference = far - frr
    right = int(np.flatnonzero(difference >= 0)[0])
    left = max(0, right - 1)
    span = difference[right] - difference[left]
    weight = -difference[left] / span if span else 0.0
    eer = far[left] + weight * (far[right] - far[left])
    best = int(np.argmin(np.abs(difference)))
    result = {
        "eer": float(eer),
        "eer_percent": float(100 * eer),
        "discrete_threshold": float(thresholds[best]),
        "far_at_discrete_threshold": float(far[best]),
        "frr_at_discrete_threshold": float(frr[best]),
        "genuine_pairs": int(np.sum(np.asarray(labels) == 1)),
        "impostor_pairs": int(np.sum(np.asarray(labels) == 0)),
    }
    if threshold is not None:
        if not np.isfinite(threshold):
            raise ValueError("Threshold must be finite")
        labels = np.asarray(labels)
        accepted = np.asarray(distances) <= threshold
        result.update({
            "fixed_threshold": float(threshold),
            "far": float(accepted[labels == 0].mean()),
            "frr": float((~accepted[labels == 1]).mean()),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--column", default="distance")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.scores, encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = [int(row["label"]) for row in rows]
    scores = [float(row[args.column]) for row in rows]
    report = verification_metrics(labels, scores, args.threshold)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    thresholds, far, frr = error_rates(labels, scores)
    with (output / "curve.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["threshold", "far", "frr"])
        writer.writerows(zip(thresholds, far, frr))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
