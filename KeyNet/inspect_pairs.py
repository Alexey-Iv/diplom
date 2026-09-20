"""Inspect image/target/mask pairs BEFORE training, using its identical data loader."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from data import Pairs
from geometry import erode, warp
from train import parse_args


if __name__ == "__main__":
    args = parse_args()
    dataset = Pairs(args, "train")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    pictures = []

    for index in range(min(16, len(dataset))):
        batch = dataset[index]

        src_mask = erode(batch["src_mask"][None])
        dst_mask = erode(batch["dst_mask"][None])

        common = src_mask * (
            warp(
                dst_mask,
                torch.linalg.inv(batch["H"])[None],
            )
            > 0.999
        )

        counts = {}

        for size in args.windows:
            height, width = common.shape[-2:]

            crop = common[
                ...,
                : height - height % size,
                : width - width % size,
            ]

            windows = torch.nn.functional.unfold(
                crop,
                size,
                stride=size,
            )

            counts[str(size)] = int(
                (windows.amin(1) > 0.999).sum()
            )

        rows.append(
            {
                "path": dataset.records[index]["path"],
                "valid_fraction": float(common.mean()),
                "valid_windows": counts,
            }
        )

        panels = []

        for key in (
            "src",
            "dst",
            "src_mask",
            "dst_mask",
        ):
            image = np.uint8(
                batch[key][0].numpy() * 255
            )

            panels.append(
                Image.fromarray(image).convert("RGB")
            )

        line = Image.new(
            "RGB",
            (
                args.patch_size * 4,
                args.patch_size,
            ),
        )

        for panel_index, image in enumerate(panels):
            line.paste(
                image,
                (
                    panel_index * args.patch_size,
                    0,
                ),
            )

        pictures.append(line)

    canvas = Image.new(
        "RGB",
        (
            args.patch_size * 4,
            args.patch_size * len(pictures),
        ),
    )

    for index, image in enumerate(pictures):
        canvas.paste(
            image,
            (
                0,
                index * args.patch_size,
            ),
        )

    canvas.save(out / "pairs.png")

    (out / "pairs.json").write_text(
        json.dumps(
            rows,
            indent=2,
        )
    )

    print(
        "Saved pairs.png: source | transformed | "
        "source valid mask | transformed valid mask"
    )