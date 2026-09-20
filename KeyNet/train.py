import argparse
import json
import platform
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from checkpoints import initialize
from data import Pairs
from keyNet.model.keynet_architecture import keynet
from train_utils import fix_randseed, train_epoch, validate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Training-only repair of the uploaded KeyNet project"
    )

    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mask-dir")
    parser.add_argument("--ignore-masks", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--border", type=int, default=4)

    parser.add_argument(
        "--geometry",
        choices=["affine", "iris-shift"],
        default="affine",
    )

    parser.add_argument("--max-angle", type=float, default=3.0)
    parser.add_argument("--max-scale", type=float, default=1.0)
    parser.add_argument("--max-shear", type=float, default=0.0)
    parser.add_argument("--max-shift", type=float, default=3.0)

    parser.add_argument(
        "--windows",
        type=lambda value: [
            int(item)
            for item in value.split(",")
        ],
        default=[8, 16, 24],
    )

    parser.add_argument(
        "--factors",
        type=lambda value: [
            float(item)
            for item in value.split(",")
        ],
        default=[256.0, 64.0, 16.0],
    )

    parser.add_argument(
        "--coordinate-weighting",
        default=True,
    )

    parser.add_argument(
        "--score-activation",
        choices=["relu", "softplus"],
        default="relu",
    )

    parser.add_argument("--hermite", action="store_true")
    parser.add_argument("--init")
    parser.add_argument("--expand-input", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--topk", type=int, default=25)
    parser.add_argument("--nms-size", type=int, default=5)
    parser.add_argument(
        "--pixel-threshold",
        type=float,
        default=3.0,
    )
    parser.add_argument("--num-filters", type=int, default=8)
    parser.add_argument(
        "--num-learnable-blocks",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--num-levels-within-net",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--factor-scaling-pyramid",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--conv-kernel-size",
        type=int,
        default=5,
    )
    parser.add_argument("--max-steps", type=int)

    args = parser.parse_args(argv)

    if args.init and args.resume:
        parser.error(
            "--init and --resume are different operations; choose one"
        )

    if args.expand_input and not (
        args.init and args.hermite
    ):
        parser.error(
            "--expand-input requires --init and --hermite"
        )

    if (
        args.patch_size < 32
        or args.border < 0
        or 2 * args.border >= args.patch_size
    ):
        parser.error(
            "Need patch>=32 and 0<=2*border<patch"
        )

    if (
        min(
            args.max_angle,
            args.max_shear,
            args.max_shift,
        )
        < 0
        or args.max_scale < 1
    ):
        parser.error("Invalid augmentation limits")

    if (
        args.max_steps is not None
        and args.max_steps < 1
    ):
        parser.error("max-steps must be >=1")

    if (
        len(args.windows) != len(args.factors)
        or any(
            size < 2 or size > args.patch_size
            for size in args.windows
        )
        or any(
            weight <= 0
            for weight in args.factors
        )
    ):
        parser.error("Invalid MSIP windows/factors")

    if (
        args.conv_kernel_size % 2 == 0
        or args.conv_kernel_size < 1
    ):
        parser.error("Odd convolution kernel required")

    return args


def run(args):
    torch.set_num_threads(args.threads)
    fix_randseed(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if (out / "last.pt").exists() and not args.resume:
        raise ValueError(
            "Run exists; use --resume or new --out"
        )

    train_data = Pairs(args, "train")
    val_data = Pairs(args, "val")

    generator = torch.Generator()

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = keynet(
        args,
        torch.device(args.device),
    ).to(args.device)

    config = vars(args).copy()

    manifest_path = Path(args.manifest)

    config["manifest_path"] = str(
        manifest_path.resolve()
    )

    config["manifest_mtime"] = (
        manifest_path.stat().st_mtime
        if manifest_path.exists()
        else None
    )

    # Сохраняем путь к init-файлу вместо его хэша.
    config["init_path"] = (
        str(Path(args.init).resolve())
        if args.init
        else None
    )

    if args.init:
        init_path = Path(args.init)

        config["init_mtime"] = (
            init_path.stat().st_mtime
            if init_path.exists()
            else None
        )

        initialization = initialize(
            model,
            args.init,
            args.expand_input,
        )

        print(
            json.dumps(
                {
                    "initialization": initialization,
                }
            ),
            flush=True,
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=5,
            factor=0.5,
        )
    )

    start = 0
    best = -1.0
    best_epoch = -1

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=True,
        )

        ignored = {
            "epochs",
            "resume",
            "init",
            "init_path",
            "init_mtime",
            "expand_input",
            "out",
            "data_dir",
            "mask_dir",
            "manifest",
            "manifest_path",
            "manifest_mtime",
            "device",
            "threads",
        }

        current = {
            key: value
            for key, value in config.items()
            if key not in ignored
        }

        old = {
            key: value
            for key, value in checkpoint["config"].items()
            if key not in ignored
        }

        if current != old:
            raise ValueError(
                "Resume configuration/manifest differs"
            )

        if checkpoint.get("format") != "keynet-training-1":
            raise ValueError(
                "Use --init for legacy or weights-only checkpoint"
            )

        model.load_state_dict(
            checkpoint["model"],
            strict=True,
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

        torch.set_rng_state(
            checkpoint["torch_rng"]
        )

        if (
            torch.cuda.is_available()
            and checkpoint["cuda_rng"]
        ):
            torch.cuda.set_rng_state_all(
                checkpoint["cuda_rng"]
            )

        start = checkpoint["epoch"] + 1
        best = checkpoint["best"]
        best_epoch = checkpoint["best_epoch"]

        config["init_path"] = checkpoint[
            "config"
        ].get("init_path")

    (out / "config.json").write_text(
        json.dumps(
            config,
            indent=2,
        )
    )

    environment = {
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "device": args.device,
    }

    (out / "environment.json").write_text(
        json.dumps(
            environment,
            indent=2,
        )
    )

    def snapshot(epoch):
        return {
            "format": "keynet-training-1",
            "config": config,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best": best,
            "best_epoch": best_epoch,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else []
            ),
        }

    if not args.resume:
        baseline = validate(
            val_loader,
            model,
            args,
        )

        (out / "before_training.json").write_text(
            json.dumps(
                baseline,
                indent=2,
            )
        )

        best = baseline["val_repeatability_px"]
        best_epoch = -1

        torch.save(
            snapshot(-1),
            out / "best.pt",
        )

    for epoch in range(start, args.epochs):
        train_data.epoch = epoch
        generator.manual_seed(
            args.seed + epoch
        )

        stats = train_epoch(
            train_loader,
            model,
            optimizer,
            args,
        )

        val = validate(
            val_loader,
            model,
            args,
        )

        score = val["val_repeatability_px"]
        improved = score > best

        if improved:
            best = score
            best_epoch = epoch

        scheduler.step(score)

        row = {
            "epoch": epoch,
            **stats,
            **val,
            "lr": optimizer.param_groups[0]["lr"],
            "smoke_only": args.max_steps is not None,
        }

        with (out / "history.jsonl").open(
            "a"
        ) as file:
            file.write(
                json.dumps(row) + "\n"
            )

        checkpoint = snapshot(epoch)

        torch.save(
            checkpoint,
            out / "last.pt",
        )

        if improved:
            torch.save(
                checkpoint,
                out / "best.pt",
            )

        print(
            json.dumps(row),
            flush=True,
        )

    return {
        "best_repeatability": best,
        "best_epoch": best_epoch,
        "out": str(out),
        "smoke_only": args.max_steps is not None,
    }


if __name__ == "__main__":
    print(
        json.dumps(
            run(parse_args()),
            indent=2,
        )
    )