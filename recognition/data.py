"""Чтение metadata.csv, разделение по людям и выбор обучающих батчей."""

import csv
import random
import re

import numpy as np
from PIL import Image
import torch

from . import config


def find_eye(path):
    """Определяем L/R из имени CASIA или каталога вида 000/L/01.png."""
    match = re.search(r"S5\d{3}([LR])\d{2}", path, re.I)
    if not match:
        match = re.search(r"(?:^|[/_\\-])([LR])(?:[/_\\-]|$)", path, re.I)
    if match:
        return match[1].upper()
    raise ValueError(f"Добавьте eye в CSV: глаз не определён для {path}")


def read_metadata():
    """Возвращаем строки CSV с метками subject, eye и split."""
    with config.METADATA.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=config.CSV_SEPARATOR)
        required = {config.PATH_COLUMN, config.SUBJECT_COLUMN}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"В CSV нужны столбцы: {sorted(required)}")
        rows = []
        for source in reader:
            path = source[config.PATH_COLUMN].strip()
            subject = source[config.SUBJECT_COLUMN].strip()
            eye = source.get(config.EYE_COLUMN, "").strip().upper()
            if not path or not subject:
                raise ValueError("Пустой путь или номер человека в CSV")
            eye = eye or find_eye(path)
            if eye not in {"L", "R"}:
                raise ValueError(f"Глаз должен быть L или R: {path}")
            rows.append({
                "path": path, "subject": subject, "eye": eye,
                "split": source.get(config.SPLIT_COLUMN, "").strip().lower(),
                "mask": source.get(config.MASK_COLUMN, "").strip(),
            })
    if not rows:
        raise ValueError("CSV пуст")
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("В CSV повторяются пути изображений")
    if not any(row["split"] for row in rows):
        subjects = sorted({row["subject"] for row in rows})
        if len(subjects) < 6:
            raise ValueError("Для автоматического разделения нужно >= 6 людей")
        random.Random(config.SEED).shuffle(subjects)
        count = max(1, int(len(subjects) * 0.15))
        groups = {}
        for index, subject in enumerate(subjects):
            if index < count:
                groups[subject] = "test"
            elif index < 2 * count:
                groups[subject] = "val"
            else:
                groups[subject] = "train"
        for row in rows:
            row["split"] = groups[row["subject"]]
    groups = {}
    for row in rows:
        split = row["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError("Столбец split: укажите train, val или test")
        if row["subject"] in groups and groups[row["subject"]] != split:
            raise ValueError("Один человек попал в разные части выборки")
        groups[row["subject"]] = split
    return rows


def read_image(row):
    """Возвращаем серое изображение [1, H, W] и бинарную маску."""
    with Image.open(config.DATA_DIR / row["path"]) as image:
        array = np.array(image.convert("L"), dtype=np.float32) / 255
    mask = np.ones_like(array)
    if row["mask"]:
        with Image.open(config.DATA_DIR / row["mask"]) as image:
            mask = (np.array(image.convert("L")) > 0).astype(np.float32)
        if mask.shape != array.shape:
            raise ValueError(f"Размер маски не совпадает: {row['path']}")
    return torch.from_numpy(array)[None], torch.from_numpy(mask)[None]


def get_split(name):
    rows = [row for row in read_metadata() if row["split"] == name]
    if not rows:
        raise ValueError(f"Нет изображений для {name}")
    identities = sorted({(row["subject"], row["eye"]) for row in rows})
    labels = {identity: index for index, identity in enumerate(identities)}
    for row in rows:
        row["label"] = labels[(row["subject"], row["eye"])]
    return rows


def training_batches(rows, seed):
    """P разных радужек по K разных снимков. Батч — список строк CSV."""
    groups = {}
    for row in rows:
        groups.setdefault(row["label"], []).append(row)
    count = config.IMAGES_PER_IDENTITY
    groups = {label: items for label, items in groups.items()
              if len(items) >= count}
    people = config.IDENTITIES_PER_BATCH
    if people < 2 or count < 2 or len(groups) < people:
        raise ValueError("Нужно P >= 2 радужек с K >= 2 снимками каждой")
    rng = random.Random(seed)
    for _ in range(config.STEPS_PER_EPOCH):
        labels = rng.sample(list(groups), people)
        yield [row for label in labels for row in rng.sample(
            groups[label], count
        )]


if __name__ == "__main__":
    for name in ("train", "val", "test"):
        rows = get_split(name)
        people = len({row["subject"] for row in rows})
        print(f"{name}: {len(rows)} изображений, {people} людей")
