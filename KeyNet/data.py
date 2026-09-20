import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from geometry import border_mask, warp


def subject_from_path(path):
    path = Path(path)

    match = re.fullmatch(
        r"S5(\d{3})[LR]\d{2}",
        path.stem,
        re.IGNORECASE,
    )
    if match:
        return match[1]

    match = re.fullmatch(
        r"(\d{3,4})[_-][LR][_-]\d+",
        path.stem,
        re.IGNORECASE,
    )
    if match:
        return match[1]

    for index in range(1, len(path.parts) - 1):
        if (
            path.parts[index].upper() in ("L", "R")
            and path.parts[index - 1].isdigit()
        ):
            return path.parts[index - 1]

    raise ValueError(
        f"Unknown subject for {path}; "
        "supply metadata CSV path,subject"
    )


def prepare(
    root,
    manifest,
    masks=None,
    metadata=None,
    seed=42,
):
    root = Path(root).resolve()
    masks = Path(masks).resolve() if masks else None

    if masks == root:
        raise ValueError(
            "Use a separate mask directory"
        )

    if Path(manifest).exists():
        raise ValueError(
            "Manifest already exists; "
            "use it or choose a new filename"
        )

    if metadata:
        with open(
            metadata,
            encoding="utf-8-sig",
            newline="",
        ) as file:
            entries = list(csv.DictReader(file))
    else:
        entries = [
            {
                "path": path.relative_to(
                    root
                ).as_posix()
            }
            for path in sorted(root.rglob("*"))
            if (
                path.is_file()
                and path.suffix.lower()
                in (
                    ".bmp",
                    ".png",
                    ".jpg",
                    ".jpeg",
                )
                and not (
                    masks
                    and path.is_relative_to(masks)
                )
            )
        ]

    if not entries:
        raise ValueError("No images found")

    records = []

    for entry in entries:
        relative_path = Path(entry["path"])
        image_path = (
            root / relative_path
        ).resolve()

        # Проверка на вложенность пути для Python < 3.9.
        try:
            image_path.resolve().relative_to(
                root.resolve()
            )
        except ValueError as exc:
            raise ValueError(
                "Paths must be relative and contained"
            ) from exc

        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise ValueError(
                "Paths must be relative and contained"
            )

        with Image.open(image_path) as image:
            size = image.size

        mask_path = (
            (masks / relative_path).with_suffix(
                ".png"
            )
            if masks
            else None
        )

        if mask_path:
            with Image.open(mask_path) as image:
                if image.size != size:
                    raise ValueError(
                        "Image/mask size mismatch: "
                        f"{relative_path}"
                    )

                mask_array = np.asarray(
                    image.convert("L")
                )

                if (
                    not set(
                        np.unique(mask_array)
                    ).issubset({0, 1, 255})
                    or not mask_array.any()
                ):
                    raise ValueError(
                        "Invalid binary mask: "
                        f"{relative_path}"
                    )

        records.append(
            {
                "path": relative_path.as_posix(),
                "subject": str(
                    entry.get("subject")
                    or subject_from_path(
                        relative_path
                    )
                ),
                "size": list(size),
            }
        )

    people = sorted(
        {
            record["subject"]
            for record in records
        }
    )

    if len(people) < 6:
        raise ValueError(
            "Need >=6 subjects for three splits"
        )

    np.random.default_rng(seed).shuffle(people)

    split_size = max(
        1,
        int(0.15 * len(people)),
    )

    groups = {
        subject: (
            "test"
            if index < split_size
            else (
                "val"
                if index < 2 * split_size
                else "train"
            )
        )
        for index, subject in enumerate(people)
    }

    for record in records:
        record["split"] = groups[
            record["subject"]
        ]

    manifest = Path(manifest)
    manifest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "format": 1,
        "seed": seed,
        "records": records,
    }

    manifest.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )

    return {
        split: {
            "images": sum(
                record["split"] == split
                for record in records
            ),
            "subjects": len(
                {
                    record["subject"]
                    for record in records
                    if record["split"] == split
                }
            ),
        }
        for split in (
            "train",
            "val",
            "test",
        )
    }


def load_manifest(path):
    data = json.loads(
        Path(path).read_text()
    )

    if (
        data.get("format") != 1
        or not data.get("records")
    ):
        raise ValueError(
            "Invalid manifest"
        )

    people = {}
    paths = set()

    for record in data["records"]:
        if record["split"] not in (
            "train",
            "val",
            "test",
        ):
            raise ValueError(
                "Unknown split"
            )

        if (
            record["subject"] in people
            and people[record["subject"]]
            != record["split"]
        ):
            raise ValueError(
                "Subject leakage"
            )

        people[record["subject"]] = (
            record["split"]
        )

        if record["path"] in paths:
            raise ValueError(
                "Duplicate image path"
            )

        relative_path = Path(
            record["path"]
        )

        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise ValueError(
                "Unsafe path"
            )

        paths.add(record["path"])

    return data


class Pairs(Dataset):
    """Self-supervised pairs with subject-based data splits."""

    def __init__(self, args, split):
        self.args = args
        self.split = split
        self.epoch = 0

        self.root = Path(args.data_dir)

        self.masks = (
            Path(args.mask_dir)
            if args.mask_dir
            else None
        )

        self.records = [
            record
            for record in load_manifest(
                args.manifest
            )["records"]
            if record["split"] == split
        ]

        if not self.records:
            raise ValueError(
                f"Empty {split} split"
            )

        for record in self.records:
            if min(record["size"]) < args.patch_size:
                raise ValueError(
                    "Image smaller than patch_size: "
                    f'{record["path"]} '
                    f'{record["size"]}'
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        args = self.args
        record = self.records[index]
        size = args.patch_size

        epoch = (
            self.epoch
            if self.split == "train"
            else 0
        )

        rng = np.random.default_rng(
            args.seed
            + index
            + 1_000_003 * epoch
        )

        image_path = (
            self.root / record["path"]
        )

        with Image.open(image_path) as image:
            full = (
                np.array(
                    image.convert("L"),
                    dtype=np.float32,
                )
                / 255
            )

        if (
            self.masks
            and not args.ignore_masks
        ):
            mask_path = (
                self.masks / record["path"]
            ).with_suffix(".png")

            with Image.open(mask_path) as image:
                mask = (
                    np.array(
                        image.convert("L")
                    )
                    > 0
                ).astype(np.float32)
        else:
            mask = np.ones_like(full)

        height, width = full.shape

        # Uniform random crop instead of fixed central crop.
        # Equal image/patch size is valid.
        top = int(
            rng.integers(
                0,
                height - size + 1,
            )
        )
        left = int(
            rng.integers(
                0,
                width - size + 1,
            )
        )

        src = torch.from_numpy(
            full[
                top : top + size,
                left : left + size,
            ].copy()
        )[None]

        src_mask = torch.from_numpy(
            mask[
                top : top + size,
                left : left + size,
            ].copy()
        )[None]

        if args.geometry == "affine":
            angle = math.radians(
                rng.uniform(
                    -args.max_angle,
                    args.max_angle,
                )
            )
            scale = rng.uniform(
                1 / args.max_scale,
                args.max_scale,
            )
            shear = rng.uniform(
                -args.max_shear,
                args.max_shear,
            )
        else:
            angle = 0.0
            scale = 1.0
            shear = 0.0

        dx = rng.uniform(
            -args.max_shift,
            args.max_shift,
        )

        dy = (
            rng.uniform(
                -args.max_shift,
                args.max_shift,
            )
            if args.geometry == "affine"
            else 0.0
        )

        rotation = np.array(
            [
                [
                    math.cos(angle),
                    -math.sin(angle),
                ],
                [
                    math.sin(angle),
                    math.cos(angle),
                ],
            ],
            dtype=np.float32,
        )

        linear = (
            np.array(
                [
                    [1.0, shear],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            )
            @ rotation
            * scale
        )

        centre = np.array(
            [
                (size - 1) / 2,
                (size - 1) / 2,
            ],
            dtype=np.float32,
        )

        homography = np.eye(
            3,
            dtype=np.float32,
        )

        homography[:2, :2] = linear
        homography[:2, 2] = (
            centre
            - linear @ centre
            + [dx, dy]
        )

        homography = torch.from_numpy(
            homography
        )

        dst = warp(
            src[None],
            homography[None],
        )[0]

        dst_mask = (
            warp(
                src_mask[None],
                homography[None],
            )[0]
            > 0.999
        ).float()

        # Grayscale photometry; no HSV/channel-shuffling
        # for NIR data.
        dst = (
            dst
            * rng.uniform(0.85, 1.15)
            + rng.uniform(-0.03, 0.03)
        ).clamp(0, 1)

        src_input_mask = src_mask
        dst_input_mask = dst_mask

        src_mask = border_mask(
            src_mask,
            args.border,
        )
        dst_mask = border_mask(
            dst_mask,
            args.border,
        )

        return {
            "src": src,
            "dst": dst,
            "src_mask": src_mask,
            "dst_mask": dst_mask,
            "src_input_mask": src_input_mask,
            "dst_input_mask": dst_input_mask,
            "H": homography,
            "index": index,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        required=True,
    )
    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument("--mask-dir")
    parser.add_argument("--metadata")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    result = prepare(
        args.data_dir,
        args.manifest,
        args.mask_dir,
        args.metadata,
        args.seed,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )