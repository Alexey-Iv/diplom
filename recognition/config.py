"""Все настройки меняются здесь. Команды запуска указаны в README."""

from pathlib import Path


# Пути относительно корня репозитория.
DATA_DIR = Path("data/normalized_train_YOLO_CASIA")
METADATA = Path("/home/a.ivanov/Datasets/diplom/KeyNet/metadata.csv")
KEYNET_DIR = Path("KeyNet")
KEYNET_WEIGHTS = "/home/a.ivanov/keynet-main/keyNet/weights/KeyNet_default_640_480_YOLO_best:)_0328_173133.log/best_model.pt" #"/home/a.ivanov/Datasets/diplom/KeyNet/runs/baseline_2026_09_15_01:40PM/best.pt"
RUNS_DIR = Path("runs/recognition_hardnet_2026_09_26_1")
HARDNET_IDENTITIES_PER_BATCH = 16
HARDNET_EER_EVERY = 5

# Названия столбцов вашего CSV. Измените правую часть при необходимости.
PATH_COLUMN = "path"
SUBJECT_COLUMN = "subject"
EYE_COLUMN = "eye"       # Необязателен, если L/R есть в имени или пути.
SPLIT_COLUMN = "split"   # Необязателен: train / val / test.
MASK_COLUMN = "mask"     # Необязателен: путь маски относительно DATA_DIR.
CSV_SEPARATOR = ","      # Для CSV из Excel иногда нужен ";".

DEVICE = "cuda:2"           # Для видеокарты: "cuda:0".
SEED = 42
THREADS = 4
EPOCHS = 30
STEPS_PER_EPOCH = 100
LEARNING_RATE = 0.0001
IDENTITIES_PER_BATCH = 2
IMAGES_PER_IDENTITY = 2

# Сначала точки находятся на изображении в исходном размере.
TOP_K = 200
NMS_SIZE = 5
PATCH_SIZE = 32
KAPPA = 10
IMAGE_SIZE = (64, 512)   # Размер входа AG-Net: высота, ширина.

EMBEDDING_DIM = 256
REGION_CHANNELS = 128
PRETRAINED_BACKBONE = True  # True: скачать ImageNet-веса ResNet-50.
ARC_WEIGHT = 3.0
TRIPLET_WEIGHT = 0.9

# Параметры сопоставления и формула гибрида из диплома.
MATCH_THRESHOLD = 0.4
RATIO = 0.8
HYBRID_WEIGHT = 0.85
HYBRID_ALPHA = 0.1
MAX_IMPOSTOR_PAIRS = 100000
