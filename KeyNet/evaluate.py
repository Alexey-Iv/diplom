"""Detector-only held-out repeatability and diagnostic images."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from data import Pairs, digest
from keyNet.model.keynet_architecture import keynet
from train_utils import forward_scores, points, validate


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--data-dir",
        required=True,
    )
    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument("--mask-dir")
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    parser.add_argument(
        "--out",
        required=True,
    )

    cli_args = parser.parse_args()

    checkpoint = torch.load(
        cli_args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("format") != "keynet-training-1":
        raise ValueError(
            "Use a checkpoint produced by this training project"
        )

    if (
        digest(cli_args.manifest)
        != checkpoint["config"]["manifest_sha256"]
    ):
        raise ValueError(
            "Manifest differs from training"
        )

    config = {
        **checkpoint["config"],
        "data_dir": cli_args.data_dir,
        "manifest": cli_args.manifest,
        "mask_dir": cli_args.mask_dir,
        "device": cli_args.device,
    }

    args = SimpleNamespace(**config)

    torch.set_num_threads(args.threads)

    model = keynet(
        args,
        torch.device(cli_args.device),
    ).to(cli_args.device)

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )
    model.eval()

    dataset = Pairs(
        args,
        cli_args.split,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
    )

    metrics = validate(
        loader,
        model,
        args,
    )

    metrics = {
        key.removeprefix("val_"): value
        for key, value in metrics.items()
    }

    metrics["split"] = cli_args.split
    metrics["smoke_only"] = (
        args.max_steps is not None
    )
    metrics["metric"] = (
        "pixel-distance maximum one-to-one repeatability, "
        "fraction in [0,1]"
    )

    out = Path(cli_args.out)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    (out / "metrics.json").write_text(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    previews = []

    with torch.inference_mode():
        for index in range(min(8, len(dataset))):
            batch = dataset[index]

            image = batch["src"][None].to(
                cli_args.device
            )
            mask = batch["src_mask"][None].to(
                cli_args.device
            )
            input_mask = batch["src_input_mask"][None].to(
                cli_args.device
            )

            _, scores = forward_scores(
                model,
                image,
                input_mask,
                args.score_activation,
            )

            keypoints = points(
                scores[0, 0],
                mask[0, 0],
                args.topk,
                args.nms_size,
            )

            gray = np.uint8(
                batch["src"][0].numpy() * 255
            )

            rgb = np.repeat(
                gray[..., None],
                3,
                axis=-1,
            )

            invalid = (
                batch["src_mask"][0].numpy() == 0
            )

            rgb[invalid] = (
                rgb[invalid] * 0.4
                + np.array([160, 0, 0]) * 0.6
            ).astype("uint8")

            preview = Image.fromarray(
                rgb
            ).resize(
                (
                    args.patch_size * 3,
                    args.patch_size * 3,
                )
            )

            draw = ImageDraw.Draw(preview)

            for x_coord, y_coord in keypoints.tolist():
                draw.ellipse(
                    (
                        x_coord * 3 - 3,
                        y_coord * 3 - 3,
                        x_coord * 3 + 3,
                        y_coord * 3 + 3,
                    ),
                    outline="yellow",
                    width=2,
                )

            previews.append(preview)

    canvas = Image.new(
        "RGB",
        (
            args.patch_size * 3 * len(previews),
            args.patch_size * 3,
        ),
        "white",
    )

    for index, preview in enumerate(previews):
        canvas.paste(
            preview,
            (
                index * args.patch_size * 3,
                0,
            ),
        )

    canvas.save(
        out / "keypoints.png"
    )

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()