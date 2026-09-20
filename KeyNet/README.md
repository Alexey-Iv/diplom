# KeyNet Keypoint Detector Training

This repository contains a research-oriented training pipeline for the **KeyNet** keypoint detector. The project is intentionally limited to detector training and detector evaluation. Identity classification, descriptor extraction, and a complete iris-recognition system are outside its scope.

Training is self-supervised. For every input image, the dataset generates a synthetically transformed version and keeps the known geometry between the two views. That geometry is then used by the MSIP loss and by the repeatability metric.

> Important: the included smoke tests validate the training pipeline on synthetic data. They are not CASIA training runs and should not be reported as a CASIA benchmark.

---

## Contents

- [What the project does](#what-the-project-does)
- [Project layout](#project-layout)
- [Installation](#installation)
- [Preparing the data](#preparing-the-data)
- [Inspecting training pairs](#inspecting-training-pairs)
- [Running the smoke test](#running-the-smoke-test)
- [Training the baseline](#training-the-baseline)
- [Hermite variant](#hermite-variant)
- [Training from scratch](#training-from-scratch)
- [Iris-shift geometry](#iris-shift-geometry)
- [Resuming training](#resuming-training)
- [Evaluating a model](#evaluating-a-model)
- [Output files](#output-files)
- [Main command-line options](#main-command-line-options)
- [How the loss and repeatability are computed](#how-the-loss-and-repeatability-are-computed)
- [Reproducibility](#reproducibility)
- [Common problems](#common-problems)
- [Limitations](#limitations)
- [Compatibility note for evaluate.py](#compatibility-note-for-evaluatepy)

---

## What the project does

The pipeline has five main stages:

1. Build a manifest with subject-disjoint `train`, `val`, and `test` splits.
2. Generate self-supervised pairs:
   - source patch;
   - geometrically transformed patch;
   - source and transformed masks;
   - homography matrix `H`.
3. Train KeyNet with MSIP loss.
4. Validate the detector using keypoint repeatability.
5. Save `best.pt`, `last.pt`, logs, and diagnostic images.

Subject IDs are used only to prevent subject leakage between splits. Identity labels are not used by the detector loss.

---

## Project layout

The filenames below match the current project version.

```text
project/
├── data.py
├── geometry.py
├── checkpoints.py
├── train_utils.py
├── train.py
├── inspect_pairs.py
├── evaluate.py
├── smoke.py
├── keyNet/
│   ├── model/
│   │   └── keynet_architecture.py
│   ├── loss/
│   │   └── score_loss_function.py
│   └── pretrained_nets/
│       └── keyNet.pt
├── requirements.txt
└── runs/
```

### File responsibilities

**`data.py`**

- builds the manifest;
- extracts subject IDs from paths or a metadata CSV;
- performs subject-disjoint splitting;
- loads images and optional masks;
- samples random crops;
- generates affine or `iris-shift` geometry;
- returns source/transformed training pairs.

**`geometry.py`**

Implements:

- point transformation by homography;
- image warping with `grid_sample`;
- border masking;
- mask erosion.

**`checkpoints.py`**

Strictly imports pretrained weights and checks architecture compatibility.

It also supports the explicit 10-to-14 input-channel expansion used by the Hermite variant.

**`train_utils.py`**

Contains:

- deterministic seeding;
- train/eval mode handling;
- score-map forward pass;
- one training epoch;
- keypoint NMS;
- repeatability;
- validation.

**`keyNet/loss/score_loss_function.py`**

Implements the multi-scale MSIP loss on positive score maps.

**`train.py`**

Main training entry point:

- parses arguments;
- creates data loaders;
- builds KeyNet;
- handles initialization and resume;
- runs training and validation;
- writes checkpoints and logs.

**`inspect_pairs.py`**

Visualizes the actual image/target/mask pairs used by the training loader.

**`evaluate.py`**

Runs detector-only validation or test evaluation and saves metrics and keypoint previews.

**`smoke.py`**

Creates a small synthetic dataset and exercises:

- baseline initialization;
- Hermite 10-to-14 expansion;
- `iris-shift`;
- scratch training with `softplus`;
- resume;
- equality between resumed and uninterrupted training;
- `best.pt` selection including the pre-training baseline.

---

## Installation

A dedicated virtual environment is recommended.

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch is not included in `requirements.txt`, install the appropriate build separately.

CPU example:

```bash
python -m pip install torch
```

For GPU training, install a PyTorch build compatible with your CUDA environment.

Check CUDA visibility with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Preparing the data

The project expects already prepared grayscale images.

For an iris experiment, these would normally be normalized iris strips rather than raw full-eye photographs.

Example layout:

```text
data/
├── normalized/
│   ├── 000/
│   │   ├── L/
│   │   │   ├── 00.png
│   │   │   ├── 01.png
│   │   │   └── 02.png
│   │   └── R/
│   └── 001/
└── masks/
    ├── 000/
    │   ├── L/
    │   │   ├── 00.png
    │   │   ├── 01.png
    │   │   └── 02.png
    │   └── R/
    └── 001/
```

A mask must:

- have the same spatial size as the corresponding image;
- be binary;
- use `0` for invalid regions;
- use `1` or `255` for valid regions;
- be stored as PNG.

### Automatic subject extraction

The current parser supports naming patterns such as:

```text
S5000L00.jpg
000_L_01.png
000/L/01.png
```

For a different layout, provide a CSV file:

```csv
path,subject
000/L/01.png,000
000/L/02.png,000
001/R/01.png,001
```

### Build a manifest with masks

```bash
python data.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json
```

### Without masks

```bash
python data.py \
  --data-dir data/normalized \
  --manifest data/split_no_masks.json
```

### With explicit metadata

```bash
python data.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --metadata data/subjects.csv
```

The manifest is intentionally not overwritten. If the path already exists, `data.py` exits with an error so an existing split is not silently replaced.

### Split logic

Subjects are shuffled with a fixed seed.

The current implementation assigns approximately:

- 15% of subjects to `test`;
- 15% to `val`;
- the remainder to `train`.

All images from the same subject stay in the same split.

---

## Inspecting training pairs

Before a real training run, inspect what the dataset loader is actually producing.

```bash
python inspect_pairs.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --out runs/pairs
```

This creates:

```text
runs/pairs/
├── pairs.png
└── pairs.json
```

`pairs.png` shows:

```text
source | transformed | source mask | transformed mask
```

`pairs.json` records the valid fraction and the number of fully visible MSIP windows at every configured scale.

If almost every window count is zero, fix the mask, crop size, border, or augmentation before starting a full training run.

---

## Running the smoke test

The smoke test checks the training pipeline itself.

```bash
python smoke.py --out runs/smoke
```

`runs/smoke` must be a fresh directory.

The script runs several very short training variants and also checks resume behavior.

A successful run writes:

```text
runs/smoke/summary.json
```

Example:

```json
{
  "status": "passed",
  "casia_trained": false,
  "variants": [
    "baseline",
    "hermite",
    "iris-shift",
    "scratch-softplus"
  ],
  "steps_per_mini_epoch": 2,
  "resume_max_error": 0.0
}
```

`casia_trained: false` is deliberate: the smoke test does not train on CASIA.

---

## Training the baseline

To fine-tune the supplied pretrained checkpoint:

```bash
python train.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --out runs/baseline \
  --init keyNet/pretrained_nets/keyNet.pt \
  --device cuda:0
```

CPU example:

```bash
python train.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --out runs/baseline_cpu \
  --init keyNet/pretrained_nets/keyNet.pt \
  --device cpu
```

If `runs/baseline/last.pt` already exists, a new non-resume run is rejected. Use `--resume` or choose another output directory.

---

## Hermite variant

The Hermite branch expands the first trainable layer from 10 to 14 input channels.

To initialize it from the old 10-channel checkpoint, explicitly request the expansion:

```bash
python train.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --out runs/hermite \
  --init keyNet/pretrained_nets/keyNet.pt \
  --hermite \
  --expand-input \
  --device cuda:0
```

The expansion is explicit:

- the first 10 input channels are copied from the checkpoint;
- the four new channels are initialized to zero;
- training then updates the full model normally.

Checkpoint loading is strict. Incompatible layers are not silently skipped.

---

## Training from scratch

If `--init` is omitted, the model uses the architecture's current initialization.

A useful separate experiment is `softplus` score activation:

```bash
python train.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --out runs/from_scratch \
  --score-activation softplus \
  --device cuda:0
```

This should be treated as its own experiment. With random initialization, ReLU can produce completely inactive score maps and therefore zero gradients.

---

## Iris-shift geometry

The default mode is:

```text
--geometry affine
```

The affine generator can use:

- rotation;
- scale;
- shear;
- horizontal translation;
- vertical translation.

For normalized iris strips, a simpler horizontal-shift experiment is available:

```bash
python train.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --out runs/iris_shift \
  --init keyNet/pretrained_nets/keyNet.pt \
  --geometry iris-shift \
  --device cuda:0
```

In `iris-shift` mode:

- angle = 0;
- scale = 1;
- shear = 0;
- vertical translation = 0;
- horizontal translation remains active.

This is a simplified geometry model, not a complete physical model of iris deformation.

---

## Resuming training

`last.pt` stores:

- model weights;
- optimizer state;
- scheduler state;
- current epoch;
- best validation score;
- CPU RNG state;
- CUDA RNG state.

Example: continue the same run to 60 epochs.

```bash
python train.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --out runs/baseline \
  --resume runs/baseline/last.pt \
  --epochs 60 \
  --device cuda:0
```

`--init` and `--resume` are mutually exclusive.

Resume also checks that the relevant experiment configuration is unchanged.

For a Hermite run, keep `--hermite` when resuming, but do not repeat `--expand-input`.

---

## Evaluating a model

Detector-only evaluation is performed with `evaluate.py`.

Example:

```bash
python evaluate.py \
  --checkpoint runs/baseline/best.pt \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --split test \
  --out runs/test_baseline \
  --device cuda:0
```

This creates:

```text
runs/test_baseline/
├── metrics.json
└── keypoints.png
```

`metrics.json` contains repeatability and diagnostic statistics.

`keypoints.png` shows detected points. Invalid mask regions are darkened with a red tint and keypoints are drawn as yellow circles.

This is a detector evaluation. It is not an identity-recognition accuracy measurement.

---

## Output files

A normal training run writes the following files under `--out`.

### `config.json`

The experiment configuration.

### `environment.json`

Basic environment information:

- Python version;
- PyTorch version;
- device.

### `before_training.json`

Validation result before the first optimizer step.

This matters for fine-tuning because the initial pretrained model can remain better than all later epochs.

### `history.jsonl`

One JSON object per epoch.

It includes values such as:

- training loss;
- fraction of positive raw logits;
- gradient norm;
- number of valid MSIP windows;
- validation loss;
- repeatability;
- mean number of detected points;
- number of empty pairs;
- learning rate.

### `last.pt`

The most recent complete training state.

Use it with `--resume`.

### `best.pt`

The checkpoint with the highest `val_repeatability_px`.

The pre-training model also participates in model selection, so:

```text
best_epoch = -1
```

is valid. It means fine-tuning did not improve the starting checkpoint on validation.

---

## Main command-line options

### Data

| Option | Meaning |
|---|---|
| `--data-dir` | Root directory containing images |
| `--mask-dir` | Root directory containing masks |
| `--manifest` | Split manifest |
| `--ignore-masks` | Ignore masks even when provided |

### Training

| Option | Default | Meaning |
|---|---:|---|
| `--epochs` | `30` | Number of epochs |
| `--batch-size` | `8` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--grad-clip` | `5.0` | Gradient clipping threshold |
| `--seed` | `42` | Random seed |
| `--threads` | `4` | CPU threads |
| `--device` | `cpu` | `cpu`, `cuda:0`, ... |
| `--max-steps` | unset | Limit batches per epoch for smoke/debug runs |

### Crop and border

| Option | Default |
|---|---:|
| `--patch-size` | `64` |
| `--border` | `4` |

Constraints:

```text
patch_size >= 32
border >= 0
2 * border < patch_size
```

### Geometry

| Option | Default |
|---|---:|
| `--geometry` | `affine` |
| `--max-angle` | `3.0` |
| `--max-scale` | `1.0` |
| `--max-shear` | `0.0` |
| `--max-shift` | `3.0` |

### MSIP

| Option | Default |
|---|---|
| `--windows` | `8,16,24` |
| `--factors` | `256,64,16` |
| `--coordinate-weighting` | `True` |

`windows` and `factors` must have the same length.

### Validation / keypoints

| Option | Default |
|---|---:|
| `--topk` | `25` |
| `--nms-size` | `5` |
| `--pixel-threshold` | `3.0` |

`nms-size` must be a positive odd integer.

### Architecture

| Option | Default |
|---|---:|
| `--num-filters` | `8` |
| `--num-learnable-blocks` | `3` |
| `--num-levels-within-net` | `3` |
| `--factor-scaling-pyramid` | `1.5` |
| `--conv-kernel-size` | `5` |

`conv-kernel-size` must be odd.

---

## How the loss and repeatability are computed

### Score maps

Before an image is passed through KeyNet, invalid pixels are replaced by a neutral value:

```python
image = image * mask + 0.5 * (1 - mask)
```

The mask is not automatically concatenated as an additional network input channel.

Raw scores are converted to a positive map using either:

```text
ReLU
```

or:

```text
Softplus
```

### MSIP

At every configured scale, for example:

```text
8 x 8
16 x 16
24 x 24
```

the score map is split into windows.

The proposal inside a window keeps the positive-map formulation based on:

```text
exp(score / window_max) - 1
```

Only fully visible windows contribute to the loss.

The loss is symmetric:

```text
source -> transformed
transformed -> source
```

If one direction has no valid windows, it does not dilute the valid direction.

### Repeatability

Validation:

1. performs NMS;
2. keeps at most `topk` points;
3. maps source points through `H`;
4. computes pairwise distances;
5. finds a maximum one-to-one matching;
6. reports the fraction of matches within `pixel-threshold`.

The default threshold is:

```text
3 px
```

Repeatability is reported in `[0, 1]`.

---

## Reproducibility

`fix_randseed()` seeds:

- Python `random`;
- NumPy;
- PyTorch;
- CUDA.

cuDNN deterministic mode is also enabled:

```python
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

For training samples, random crop generation depends on:

```text
seed + index + 1_000_003 * epoch
```

Validation and test pairs stay fixed because the epoch term is not applied to those splits.

The training DataLoader generator is also reseeded with:

```text
seed + epoch
```

This is part of making resumed training reproduce uninterrupted training.

---

## Common problems

### `Run exists; use --resume or new --out`

The selected output directory already contains `last.pt`.

Either resume it:

```bash
--resume runs/.../last.pt
```

or use a new `--out`.

### `No valid MSIP windows`

Check:

- masks;
- patch size;
- border width;
- MSIP window sizes;
- augmentation strength;
- score-map activity.

Run `inspect_pairs.py` first.

### `Zero gradient`

A ReLU score map can become completely inactive.

For a separate scratch experiment, try:

```text
--score-activation softplus
```

### 10-vs-14 channel mismatch

A Hermite fine-tuning run from the old checkpoint requires:

```text
--hermite --expand-input
```

### `Image smaller than patch_size`

At least one image dimension is smaller than `--patch-size`.

Reduce the patch size or preprocess the images accordingly.

### `Subject leakage`

A subject appears in more than one split.

Fix the manifest or metadata.

### `Manifest already exists`

The manifest is intentionally not overwritten.

Use a new filename, for example:

```text
data/split_v2.json
```

---

## Limitations

This repository should be interpreted as a detector training/evaluation pipeline.

It does not provide:

- iris segmentation;
- normalization of raw eye photographs;
- descriptor extraction;
- matching between real different captures;
- biometric identification or verification;
- a complete CASIA benchmark.

Synthetic repeatability is useful for studying detector stability under known geometry, but it is not by itself evidence of improved real biometric matching.

When comparing experiments, keep the following fixed unless they are the variable being studied:

- manifest;
- train/val/test split;
- seeds;
- number of keypoints;
- geometry;
- initialization;
- masks;
- evaluation threshold.

Changing several factors at once makes attribution of the result unreliable.

---

## Compatibility note for `evaluate.py`

The supplied `evaluate.py` contains:

```python
digest(a.manifest) != ck["config"]["manifest_sha256"]
```

and imports:

```python
from data import Pairs, digest
```

However, in the supplied `data.py`, `digest()` is not present, and the current `train.py` stores:

```text
manifest_path
manifest_mtime
```

rather than `manifest_sha256`.

These pieces therefore need to be synchronized before relying on `evaluate.py`.

There are two reasonable approaches.

### Option 1 — restore SHA-256 validation

Add `digest()` to `data.py` and store `manifest_sha256` in `train.py`.

This is the stricter solution because evaluation verifies manifest contents, not only a path or modification time.

### Option 2 — use the current training metadata

Remove the SHA-256 dependency from `evaluate.py` and validate the manifest using the same scheme used by `train.py`.

Do not keep the current mixed implementation: in that state, `evaluate.py` can fail before model evaluation starts.

---

## Recommended workflow

A typical experiment can follow this order:

```bash
# 1. Build the split
python data.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json

# 2. Inspect the generated pairs
python inspect_pairs.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --out runs/pairs

# 3. Check the training pipeline
python smoke.py --out runs/smoke

# 4. Train the baseline
python train.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --out runs/baseline \
  --init keyNet/pretrained_nets/keyNet.pt \
  --device cuda:0

# 5. Resume if necessary
python train.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --out runs/baseline \
  --resume runs/baseline/last.pt \
  --epochs 60 \
  --device cuda:0

# 6. Evaluate the best checkpoint
python evaluate.py \
  --checkpoint runs/baseline/best.pt \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --split test \
  --out runs/test_baseline \
  --device cuda:0
```

Before step 6, make sure the manifest check in `evaluate.py` is consistent with `data.py` and `train.py`.

---

## Scientific context

The architecture is based on Key.Net:

**Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters**  
Barroso-Laguna et al., ICCV 2019.

This repository should not be presented as an official implementation from the paper authors or as a ready-made benchmark for a particular iris dataset.

When reporting experiments, document at least:

- the source of the original architecture;
- the source of pretrained weights;
- the dataset and split;
- augmentation parameters;
- MSIP windows and weights;
- the repeatability criterion;
- the number of random seeds;
- whether masks were used;
- whether the Hermite variant was enabled.
