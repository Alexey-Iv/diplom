"""Synthetic functional exercise of KeyNet training only, not a CASIA benchmark."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from data import prepare
from train import parse_args, run


def smoke(out):
    out = Path(out).resolve()

    if out.exists():
        raise ValueError("Choose a fresh smoke output directory")

    out.mkdir(parents=True)
    rng = np.random.default_rng(2026)

    for subject in range(8):
        for image in range(3):
            rel = Path(f"{subject:03}") / "L" / f"{image:02}.png"
            p = out / "images" / rel
            m = out / "masks" / rel

            p.parent.mkdir(parents=True, exist_ok=True)
            m.parent.mkdir(parents=True, exist_ok=True)

            array = np.uint8(rng.uniform(20, 235, (64, 128)))
            mask = np.ones_like(array) * 255
            mask[:3] = 0
            mask[-3:] = 0

            Image.fromarray(array).save(p)
            Image.fromarray(mask).save(m)

    prepare(
        out / "images",
        out / "manifest.json",
        out / "masks",
    )

    common = [
        "--data-dir",
        str(out / "images"),
        "--mask-dir",
        str(out / "masks"),
        "--manifest",
        str(out / "manifest.json"),
        "--threads",
        "2",
        "--batch-size",
        "2",
        "--epochs",
        "1",
        "--max-steps",
        "2",
    ]

    weights = (
        Path(__file__).resolve().parent
        / "keyNet"
        / "pretrained_nets"
        / "keyNet.pt"
    )

    variants = [
        (
            "baseline",
            [
                "--init",
                str(weights),
            ],
        ),
        (
            "hermite",
            [
                "--init",
                str(weights),
                "--hermite",
                "--expand-input",
            ],
        ),
        (
            "shift",
            [
                "--init",
                str(weights),
                "--geometry",
                "iris-shift",
            ],
        ),
        (
            "scratch",
            [
                "--score-activation",
                "softplus",
            ],
        ),
    ]

    for name, extra in variants:
        args = common + ["--out", str(out / name)] + extra
        run(parse_args(args))

    run(
        parse_args(
            common
            + [
                "--out",
                str(out / "baseline"),
                "--epochs",
                "2",
                "--resume",
                str(out / "baseline" / "last.pt"),
            ]
        )
    )

    run(
        parse_args(
            common
            + [
                "--out",
                str(out / "continuous"),
                "--epochs",
                "2",
                "--init",
                str(weights),
            ]
        )
    )

    baseline = torch.load(
        out / "baseline" / "last.pt",
        weights_only=True,
        map_location="cpu",
    )["model"]

    continuous = torch.load(
        out / "continuous" / "last.pt",
        weights_only=True,
        map_location="cpu",
    )["model"]

    error = max(
        (
            float((baseline[key] - continuous[key]).abs().max())
            if baseline[key].numel()
            else 0
        )
        for key in baseline
    )

    if error != 0:
        raise AssertionError(
            f"Resume differs from uninterrupted training: {error}"
        )

    before_training = json.loads(
        (out / "baseline" / "before_training.json").read_text()
    )
    initial = before_training["val_repeatability_px"]

    history = [
        json.loads(line)["val_repeatability_px"]
        for line in (
            out / "baseline" / "history.jsonl"
        ).read_text().splitlines()
    ]

    best = torch.load(
        out / "baseline" / "best.pt",
        weights_only=True,
        map_location="cpu",
    )

    if best["best"] != max([initial] + history):
        raise AssertionError(
            "Initial baseline omitted from model selection"
        )

    result = {
        "status": "passed",
        "casia_trained": False,
        "variants": [
            "baseline",
            "hermite",
            "iris-shift",
            "scratch-softplus",
        ],
        "steps_per_mini_epoch": 2,
        "resume_max_error": error,
    }

    (out / "summary.json").write_text(
        json.dumps(result, indent=2)
    )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/smoke")
    args = parser.parse_args()

    print(json.dumps(smoke(args.out), indent=2))