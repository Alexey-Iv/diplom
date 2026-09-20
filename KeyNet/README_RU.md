# KeyNet для обучения детектора ключевых точек

Этот репозиторий содержит учебно-исследовательскую версию пайплайна обучения **KeyNet** для детектирования ключевых точек. Проект сфокусирован именно на обучении и оценке детектора: классификация личности, извлечение дескрипторов и полноценная система распознавания радужки сюда не входят.

Обучение построено в self-supervised режиме. Для каждого изображения создаётся синтетически преобразованная копия, а известная геометрия между исходным и преобразованным изображением используется для вычисления MSIP loss и repeatability.

> Важно: smoke-тесты в проекте проверяют корректность пайплайна на искусственных данных. Они не являются обучением или benchmark на CASIA.

---

## Содержание

- [Что делает проект](#что-делает-проект)
- [Структура проекта](#структура-проекта)
- [Установка](#установка)
- [Подготовка данных](#подготовка-данных)
- [Проверка обучающих пар](#проверка-обучающих-пар)
- [Запуск smoke-теста](#запуск-smoke-теста)
- [Обучение baseline](#обучение-baseline)
- [Hermite-вариант](#hermite-вариант)
- [Обучение с нуля](#обучение-с-нуля)
- [Геометрия iris-shift](#геометрия-iris-shift)
- [Продолжение обучения](#продолжение-обучения)
- [Оценка модели](#оценка-модели)
- [Выходные файлы](#выходные-файлы)
- [Основные параметры](#основные-параметры)
- [Как считается loss и repeatability](#как-считается-loss-и-repeatability)
- [Воспроизводимость](#воспроизводимость)
- [Типичные проблемы](#типичные-проблемы)
- [Ограничения](#ограничения)
- [Примечание по совместимости evaluate.py](#примечание-по-совместимости-evaluatepy)

---

## Что делает проект

Пайплайн состоит из пяти основных этапов:

1. Подготовка manifest с разделением данных на `train`, `val` и `test` по субъектам.
2. Формирование self-supervised пар:
   - исходный патч;
   - геометрически преобразованный патч;
   - исходная и преобразованная маски;
   - матрица гомографии `H`.
3. Обучение KeyNet с MSIP loss.
4. Validation по repeatability ключевых точек.
5. Сохранение `best.pt`, `last.pt`, логов и диагностических изображений.

Идентификатор субъекта используется только для корректного разделения выборки. В loss метки личности не участвуют.

---

## Структура проекта

Названия ниже соответствуют файлам из текущей версии проекта.

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

### За что отвечает каждый файл

**`data.py`**

- строит manifest;
- определяет subject по пути или CSV;
- разделяет субъектов на `train/val/test`;
- загружает изображения и маски;
- делает случайный crop;
- генерирует affine или `iris-shift` преобразование;
- создаёт исходную и преобразованную пару.

**`geometry.py`**

Содержит геометрические операции:

- преобразование точек гомографией;
- warp изображения через `grid_sample`;
- исключение border;
- эрозию маски.

**`checkpoints.py`**

Загружает pretrained weights и строго проверяет совместимость архитектуры.

Отдельно поддерживается расширение первого слоя с 10 до 14 входных каналов для Hermite-варианта.

**`train_utils.py`**

Содержит:

- фиксацию random seed;
- train/eval режимы;
- forward score maps;
- один train epoch;
- NMS ключевых точек;
- repeatability;
- validation.

**`keyNet/loss/score_loss_function.py`**

Реализация MSIP loss с положительными score maps и многомасштабными окнами.

**`train.py`**

Главный training entry point:

- разбирает аргументы;
- создаёт DataLoader;
- строит KeyNet;
- загружает init или resume;
- запускает обучение;
- сохраняет checkpoints и JSON-логи.

**`inspect_pairs.py`**

Создаёт визуальную проверку того, какие пары и маски реально идут в обучение.

**`evaluate.py`**

Запускает detector-only evaluation на `val` или `test`, сохраняет метрики и визуализацию keypoints.

**`smoke.py`**

Создаёт маленький синтетический dataset и проверяет:

- baseline;
- Hermite 10→14;
- `iris-shift`;
- обучение с нуля через `softplus`;
- resume;
- совпадение resume с непрерывным обучением;
- выбор `best.pt` с учётом baseline до первого optimizer step.

---

## Установка

Рекомендуется отдельное virtual environment.

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

Если PyTorch не указан в `requirements.txt`, установите подходящую сборку отдельно.

Пример CPU-варианта:

```bash
python -m pip install torch
```

Для CUDA используйте сборку PyTorch, совместимую с вашей версией CUDA.

Проверить доступность GPU можно так:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Подготовка данных

Проект ожидает уже подготовленные grayscale-изображения.

Для iris-задачи это обычно нормализованные полосы радужки, а не полный исходный кадр глаза.

Пример структуры:

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

Маска должна:

- иметь тот же размер, что и изображение;
- быть бинарной;
- содержать `0` для невалидной области;
- содержать `1` или `255` для валидной области;
- храниться как PNG.

### Автоматическое определение subject

Поддерживаются, в частности, такие варианты:

```text
S5000L00.jpg
000_L_01.png
000/L/01.png
```

Если имена устроены иначе, используйте CSV:

```csv
path,subject
000/L/01.png,000
000/L/02.png,000
001/R/01.png,001
```

### Создание manifest с масками

```bash
python data.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json
```

### Без масок

```bash
python data.py \
  --data-dir data/normalized \
  --manifest data/split_no_masks.json
```

### С собственным CSV

```bash
python data.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --metadata data/subjects.csv
```

Manifest создаётся один раз. Если файл уже существует, `data.py` завершит работу с ошибкой — это сделано специально, чтобы случайно не заменить split.

### Как делается split

Субъекты перемешиваются с фиксированным seed.

При текущей реализации:

- около 15% субъектов идут в `test`;
- около 15% — в `val`;
- остальные — в `train`.

Изображения одного субъекта не должны попадать в разные части выборки.

---

## Проверка обучающих пар

Перед полноценным обучением полезно визуально проверить входные данные.

```bash
python inspect_pairs.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --out runs/pairs
```

После запуска появятся:

```text
runs/pairs/
├── pairs.png
└── pairs.json
```

`pairs.png` содержит для каждого примера:

```text
source | transformed | source mask | transformed mask
```

`pairs.json` показывает долю валидной области и число полностью валидных MSIP-окон каждого размера.

Если почти все окна имеют значение `0`, сначала исправьте маски, crop, border или augmentation. Запускать полноценное обучение в такой ситуации нет смысла.

---

## Запуск smoke-теста

Smoke-тест нужен для проверки самого training pipeline.

```bash
python smoke.py --out runs/smoke
```

Папка `runs/smoke` должна быть новой.

Тест запускает несколько коротких вариантов обучения и проверяет корректность resume.

Успешный результат записывается в:

```text
runs/smoke/summary.json
```

Пример:

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

`casia_trained: false` означает, что smoke-тест не является CASIA training run.

---

## Обучение baseline

Если используется исходный pretrained checkpoint:

```bash
python train.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --out runs/baseline \
  --init keyNet/pretrained_nets/keyNet.pt \
  --device cuda:0
```

Для CPU:

```bash
python train.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --out runs/baseline_cpu \
  --init keyNet/pretrained_nets/keyNet.pt \
  --device cpu
```

Если `runs/baseline/last.pt` уже существует, новый запуск без `--resume` будет остановлен. Для нового эксперимента используйте новый `--out`.

---

## Hermite-вариант

Hermite-вариант расширяет вход первого обучаемого слоя с 10 до 14 каналов.

Для загрузки старого 10-канального checkpoint нужно явно разрешить расширение:

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

При расширении:

- первые 10 каналов копируются из checkpoint;
- новые 4 канала инициализируются нулями;
- после этого все параметры обучаются обычным образом.

Если размеры слоёв не совпадают и допустимое расширение 10→14 не подходит, загрузка checkpoint завершается ошибкой. Несовместимые слои молча не пропускаются.

---

## Обучение с нуля

Если не передавать `--init`, модель обучается с текущей инициализацией архитектуры.

Для отдельного эксперимента можно использовать `softplus` вместо ReLU:

```bash
python train.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --out runs/from_scratch \
  --score-activation softplus \
  --device cuda:0
```

`softplus` полезен как отдельный controlled experiment, потому что при неудачной случайной инициализации ReLU может дать полностью неактивные score maps.

---

## Геометрия iris-shift

По умолчанию используется:

```text
--geometry affine
```

Affine-вариант может включать:

- rotation;
- scale;
- shear;
- horizontal shift;
- vertical shift.

Для нормализованной iris-полосы можно отдельно проверить только горизонтальный сдвиг:

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

В режиме `iris-shift`:

- angle = 0;
- scale = 1;
- shear = 0;
- vertical shift = 0;
- остаётся горизонтальный shift.

Это упрощённая модель геометрии нормализованной полосы. Она не моделирует все реальные изменения радужки.

---

## Продолжение обучения

Checkpoint `last.pt` содержит:

- веса модели;
- optimizer;
- scheduler;
- номер эпохи;
- best score;
- RNG state CPU;
- RNG state CUDA.

Пример продолжения до 60 эпох:

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

Нельзя одновременно задавать:

```text
--init
--resume
```

При resume проверяется совместимость основных параметров эксперимента.

Для Hermite-run при resume сохраняйте `--hermite`, но `--expand-input` повторно не нужен.

---

## Оценка модели

Detector-only evaluation запускается через `evaluate.py`.

Пример:

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

Результат:

```text
runs/test_baseline/
├── metrics.json
└── keypoints.png
```

`metrics.json` содержит repeatability и дополнительные диагностические значения.

`keypoints.png` показывает найденные keypoints. Невалидные области маски затемняются красным оттенком, keypoints рисуются жёлтыми окружностями.

Это оценка **детектора**, а не accuracy распознавания личности.

---

## Выходные файлы

После обычного training run в `--out` появляются:

### `config.json`

Параметры запуска.

### `environment.json`

Минимальная информация об окружении:

- версия Python;
- версия PyTorch;
- device.

### `before_training.json`

Validation до первого optimizer step.

Это важно при fine-tuning: pretrained baseline может уже быть лучше некоторых последующих эпох.

### `history.jsonl`

Одна JSON-запись на эпоху.

Содержит, в частности:

- train loss;
- positive logit fraction;
- gradient norm;
- число валидных MSIP windows;
- validation loss;
- repeatability;
- среднее число keypoints;
- число пустых пар;
- learning rate.

### `last.pt`

Последнее состояние training run.

Используется для `--resume`.

### `best.pt`

Checkpoint с лучшим `val_repeatability_px`.

Исходная модель до обучения тоже участвует в выборе. Поэтому возможен:

```text
best_epoch = -1
```

Это означает, что fine-tuning не превзошёл стартовый checkpoint на validation.

---

## Основные параметры

### Данные

| Параметр | Значение |
|---|---|
| `--data-dir` | Корень изображений |
| `--mask-dir` | Корень масок |
| `--manifest` | Manifest split |
| `--ignore-masks` | Игнорировать маски даже если они переданы |

### Обучение

| Параметр | По умолчанию | Назначение |
|---|---:|---|
| `--epochs` | `30` | Число эпох |
| `--batch-size` | `8` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--grad-clip` | `5.0` | Gradient clipping |
| `--seed` | `42` | Seed |
| `--threads` | `4` | CPU threads |
| `--device` | `cpu` | `cpu`, `cuda:0`, ... |
| `--max-steps` | нет | Ограничение batch на эпоху для smoke/debug |

### Crop и маска

| Параметр | По умолчанию |
|---|---:|
| `--patch-size` | `64` |
| `--border` | `4` |

Требования:

```text
patch_size >= 32
border >= 0
2 * border < patch_size
```

### Геометрия

| Параметр | По умолчанию |
|---|---:|
| `--geometry` | `affine` |
| `--max-angle` | `3.0` |
| `--max-scale` | `1.0` |
| `--max-shear` | `0.0` |
| `--max-shift` | `3.0` |

### MSIP

| Параметр | По умолчанию |
|---|---|
| `--windows` | `8,16,24` |
| `--factors` | `256,64,16` |
| `--coordinate-weighting` | `True` |

Размеры `windows` и число `factors` должны совпадать.

### Keypoints / validation

| Параметр | По умолчанию |
|---|---:|
| `--topk` | `25` |
| `--nms-size` | `5` |
| `--pixel-threshold` | `3.0` |

`nms-size` должен быть нечётным положительным числом.

### Архитектура

| Параметр | По умолчанию |
|---|---:|
| `--num-filters` | `8` |
| `--num-learnable-blocks` | `3` |
| `--num-levels-within-net` | `3` |
| `--factor-scaling-pyramid` | `1.5` |
| `--conv-kernel-size` | `5` |

`conv-kernel-size` должен быть нечётным.

---

## Как считается loss и repeatability

### Score maps

Перед подачей изображения в KeyNet маска используется для нейтрального заполнения невалидной области:

```python
image = image * mask + 0.5 * (1 - mask)
```

Сама маска при этом не добавляется автоматически как отдельный канал.

Raw scores переводятся в положительную карту:

```text
ReLU
```

или:

```text
Softplus
```

### MSIP

Для каждого масштаба:

```text
8 x 8
16 x 16
24 x 24
```

score map разбивается на окна.

Proposal внутри окна строится на основе исходной positive-map формулы:

```text
exp(score / window_max) - 1
```

В loss участвуют только полностью видимые окна.

Loss считается симметрично:

```text
source -> transformed
transformed -> source
```

Если одно направление полностью невалидно, оно не разбавляет loss второго направления.

### Repeatability

Validation:

1. выполняет NMS;
2. оставляет максимум `topk` точек;
3. переводит source points через `H`;
4. считает попарные расстояния;
5. строит максимум взаимно-однозначных соответствий;
6. считает долю соответствий в пределах `pixel-threshold`.

По умолчанию:

```text
pixel_threshold = 3 px
```

Repeatability лежит в диапазоне `[0, 1]`.

---

## Воспроизводимость

Функция `fix_randseed()` устанавливает seed для:

- Python `random`;
- NumPy;
- PyTorch;
- CUDA.

Также включается deterministic режим cuDNN:

```python
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

Для каждой train-эпохи random crop зависит от:

```text
seed + index + 1_000_003 * epoch
```

Validation использует фиксированную генерацию пар, потому что epoch-компонента для `val/test` не применяется.

DataLoader train shuffle также переинициализируется через:

```text
seed + epoch
```

Это сделано для воспроизводимого resume.

---

## Типичные проблемы

### `Run exists; use --resume or new --out`

В указанной папке уже есть `last.pt`.

Либо продолжите run:

```bash
--resume runs/.../last.pt
```

либо используйте новый `--out`.

### `No valid MSIP windows`

Проверьте:

- корректность mask;
- `patch-size`;
- слишком большой `border`;
- слишком большие MSIP windows;
- слишком сильную augmentation;
- активность score map.

Сначала запустите `inspect_pairs.py`.

### `Zero gradient`

При ReLU score map может полностью "умереть".

Для отдельного scratch-эксперимента можно проверить:

```text
--score-activation softplus
```

### Ошибка 10 vs 14 каналов

Для Hermite fine-tuning старого checkpoint нужен:

```text
--hermite --expand-input
```

### `Image smaller than patch_size`

Хотя бы одна сторона изображения меньше `--patch-size`.

Уменьшите patch или предварительно подготовьте изображения нужного размера.

### `Subject leakage`

Один subject оказался в нескольких split.

Исправьте manifest или metadata.

### `Manifest already exists`

Старый manifest специально не перезаписывается.

Создайте новый путь:

```text
data/split_v2.json
```

---

## Ограничения

Этот проект нужно интерпретировать именно как training/evaluation pipeline для детектора.

Он не включает:

- сегментацию радужки;
- нормализацию исходных eye images;
- дескриптор;
- matching реальных разных снимков;
- идентификацию или верификацию личности;
- готовый CASIA benchmark.

Синтетическая repeatability полезна для проверки геометрической стабильности детектора, но сама по себе не доказывает улучшение реального biometric matching.

Для сравнения вариантов желательно использовать:

- один manifest;
- одинаковые train/val/test split;
- одинаковые seeds;
- одинаковое число keypoints;
- одинаковую геометрию;
- несколько независимых запусков.

Менять сразу несколько факторов и затем приписывать изменение качества одному из них — плохой экспериментальный дизайн.

---

## Примечание по совместимости `evaluate.py`

В присланной версии `evaluate.py` есть проверка:

```python
digest(a.manifest) != ck["config"]["manifest_sha256"]
```

и импорт:

```python
from data import Pairs, digest
```

При этом в присланной версии `data.py` функция `digest()` отсутствует, а текущий `train.py` сохраняет:

```text
manifest_path
manifest_mtime
```

но не `manifest_sha256`.

То есть перед использованием `evaluate.py` эти части нужно синхронизировать.

Есть два нормальных варианта.

### Вариант 1 — вернуть SHA-256

Добавить `digest()` в `data.py` и сохранять `manifest_sha256` в `train.py`.

Это более строгий вариант: evaluation сможет проверить именно содержимое manifest.

### Вариант 2 — использовать текущую схему

Убрать SHA-256 проверку из `evaluate.py` и проверять manifest тем же способом, который используется в `train.py`.

Не стоит оставлять текущую смесь двух схем: в таком состоянии `evaluate.py` может упасть ещё до загрузки модели.

---

## Рекомендуемый порядок работы

Для обычного эксперимента достаточно такого порядка:

```bash
# 1. Создать split
python data.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json

# 2. Проверить пары
python inspect_pairs.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --out runs/pairs

# 3. Проверить pipeline
python smoke.py --out runs/smoke

# 4. Обучить baseline
python train.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --out runs/baseline \
  --init keyNet/pretrained_nets/keyNet.pt \
  --device cuda:0

# 5. При необходимости продолжить
python train.py \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --out runs/baseline \
  --resume runs/baseline/last.pt \
  --epochs 60 \
  --device cuda:0

# 6. Оценить best checkpoint
python evaluate.py \
  --checkpoint runs/baseline/best.pt \
  --data-dir data/normalized \
  --mask-dir data/masks \
  --manifest data/split.json \
  --split test \
  --out runs/test_baseline \
  --device cuda:0
```

Перед пунктом 6 убедитесь, что описанная выше проверка manifest в `evaluate.py` согласована с `data.py` и `train.py`.

---

## Научный контекст

Архитектура проекта основана на Key.Net:

**Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters**  
Barroso-Laguna et al., ICCV 2019.

Этот репозиторий не следует воспринимать как официальную реализацию авторов статьи или как готовый benchmark конкретного iris dataset.

Если вы публикуете результаты, отдельно укажите:

- источник исходной архитектуры;
- источник pretrained weights;
- dataset и split;
- параметры augmentation;
- MSIP windows/factors;
- критерий repeatability;
- число seeds;
- использовались ли маски;
- использовался ли Hermite-вариант.
