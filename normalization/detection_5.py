import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from matplotlib.patches import Arc
from math import radians, degrees
from scipy.ndimage import convolve
from tensorflow.keras.models import load_model
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D, GaussianNoise, Dropout,
    Flatten, Dense, LeakyReLU, UpSampling2D, Concatenate
)


class IrisSegmenter:
    def __init__(self, model_path=None, sigma=2, total_arc_angle_deg=180, gpu_id=0):
        """
        Инициализация сегментатора радужки

        Args:
            model_path: путь к файлу весов модели (.keras или .h5)
            sigma: параметр для гауссова размытия
            total_arc_angle_deg: суммарный угол дуг интегрирования в градусах
            gpu_id: ID GPU для использования (0, 1, 2, ...)
        """
        self.sigma = sigma
        self.total_arc_angle_deg = total_arc_angle_deg
        self.gpu_id = gpu_id

        # Проверка корректности угла
        if not (0 < self.total_arc_angle_deg <= 360):
            raise ValueError("Суммарный угол должен быть в диапазоне (0, 360] градусов")

        # Настройка GPU
        self._setup_gpu()

        # Загрузка модели
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            try:
                print(f"🔄 Загрузка модели из {model_path}...")
                self.model = load_model(model_path, compile=False)
                print("✅ Модель успешно загружена!")
            except Exception as e:
                print(f"❌ Ошибка загрузки модели: {e}")
                print("🔄 Создание базовой модели...")
                self.model = self._create_basic_unet_architecture()
        else:
            print("🔄 Создание базовой модели (файл модели не указан или не существует)...")
            self.model = self._create_basic_unet_architecture()

        # Определяем параметры модели
        self.input_shape = self.model.input_shape
        print(f"📐 Архитектура модели: {self.input_shape}")

    def _setup_gpu(self):
        """Настройка GPU для TensorFlow"""
        try:
            gpus = tf.config.list_physical_devices('GPU')

            if not gpus:
                print("⚠️ GPU не найдены, будет использоваться CPU")
                return

            print(f"✅ Доступные GPU: {len(gpus)}")

            # Ограничиваем виртуальную память
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Если указан конкретный GPU
            if self.gpu_id < len(gpus):
                tf.config.set_visible_devices(gpus[self.gpu_id], 'GPU')
                print(f"✅ Используется GPU:{self.gpu_id} - {gpus[self.gpu_id].name}")
            else:
                print(f"⚠️ GPU:{self.gpu_id} недоступен, используется GPU:0")
                tf.config.set_visible_devices(gpus[0], 'GPU')

        except Exception as e:
            print(f"❌ Ошибка настройки GPU: {e}")
            print("⚠️ Будет использоваться CPU")

    def _create_basic_unet_architecture(self):
        """Создает базовую архитектуру U-Net для сегментации радужки"""
        # Входной слой
        inputs = Input(shape=(240, 320, 1))

        # Энкодер
        # Блок 1
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        # Блок 2
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # Блок 3 (мост)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)

        # Декодер
        # Блок 4
        up4 = UpSampling2D(size=(2, 2))(conv3)
        up4 = Concatenate()([up4, conv2])
        conv4 = Conv2D(128, 3, activation='relu', padding='same')(up4)
        conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)

        # Блок 5
        up5 = UpSampling2D(size=(2, 2))(conv4)
        up5 = Concatenate()([up5, conv1])
        conv5 = Conv2D(64, 3, activation='relu', padding='same')(up5)
        conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)

        # Выходной слой
        outputs = Conv2D(1, 1, activation='sigmoid')(conv5)

        model = Model(inputs=inputs, outputs=outputs, name="Basic_IRIS_UNet")
        return model

    def predict_iris_mask(self, image_path):
        """
        Предсказывает маску радужки с использованием загруженной модели
        """
        try:
            # Загрузка изображения
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path}")

            # Сохраняем оригинальные размеры для масштабирования
            original_height, original_width = image.shape

            # Изменяем размер до ожидаемого моделью
            input_height, input_width = self.model.input_shape[1:3]
            resized_image = cv2.resize(image, (input_width, input_height))

            # Нормализация и добавление размерностей
            image_norm = resized_image / 255.0
            image_arr = np.expand_dims(image_norm, axis=-1)  # Добавляем channel dimension
            image_arr = np.expand_dims(image_arr, axis=0)    # Добавляем batch dimension

            # Предсказание
            with tf.device(f'/GPU:{self.gpu_id}' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                prediction = self.model.predict(image_arr, verbose=0)

            # Обработка предсказания
            if len(prediction.shape) == 4:
                mask = prediction[0]  # Берем первый элемент батча
            else:
                mask = prediction

            # Бинаризация
            binary_mask = (mask > 0.5).astype(np.float32)

            # Если маска имеет несколько каналов, берем первый
            if len(binary_mask.shape) == 3 and binary_mask.shape[-1] > 1:
                binary_mask = binary_mask[:, :, 0]

            # Масштабируем обратно к оригинальному размеру
            if binary_mask.shape[:2] != (original_height, original_width):
                binary_mask = cv2.resize(binary_mask, (original_width, original_height))

            return binary_mask

        except Exception as e:
            print(f"❌ Ошибка в predict_iris_mask: {e}")
            # Возвращаем пустую маску в случае ошибки
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                return np.zeros_like(image, dtype=np.float32)
            else:
                return None

    def gaussian_derivative(self, img, sigma):
        # Преобразуем в float для точности вычислений
        img = img.astype(np.float64)

        # Определяем оптимальный размер ядра (6*sigma покрывает 99.7% распределения)
        kernel_size = max(3, int(6 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Создаем координатную сетку с центром в (0, 0)
        ax = np.arange(-kernel_size//2 + 1, kernel_size//2 + 1)
        xx, yy = np.meshgrid(ax, ax)

        # Вычисляем ядро Гаусса
        gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

        # Аналитические производные ядра Гаусса
        kernel_dx = -xx * gaussian / (sigma**2)
        kernel_dy = -yy * gaussian / (sigma**2)

        # Нормализуем ядра для сохранения энергии
        norm_factor = np.sum(gaussian) * sigma
        if norm_factor != 0:
            kernel_dx /= norm_factor
            kernel_dy /= norm_factor

        # Применяем свертку с ядрами производных
        grad_x = convolve(img, kernel_dx, mode='constant', cval=0.0)
        grad_y = convolve(img, kernel_dy, mode='constant', cval=0.0)

        return grad_x, grad_y

    def compute_circular_integral(self, grad_x, grad_y, x0, y0, r, angles):
        """Вычисляет интеграл вдоль дуги окружности"""
        integral = 0
        count = 0

        for angle in angles:
            # Координаты точки на окружности
            x = int(x0 + r * np.cos(angle))
            y = int(y0 + r * np.sin(angle))

            # Нормаль к окружности (направление радиуса)
            nx = np.cos(angle)
            ny = np.sin(angle)

            # Производная вдоль нормали (скалярное произведение)
            if 0 <= x < grad_x.shape[1] and 0 <= y < grad_x.shape[0]:
                derivative = grad_x[y, x] * nx + grad_y[y, x] * ny
                integral += derivative
                count += 1

        return integral / count if count > 0 else 0

    def compute_region_intensities_two_regions(self, img, x0, y0, r_outer):
        """Вычисляет средние интенсивности в двух симметричных областях"""
        height, width = img.shape
        intensities = []

        # Вычисляем половину угла для каждой области в градусах
        half_angle_deg = self.total_arc_angle_deg / 4.0
        half_angle_rad = np.deg2rad(half_angle_deg)

        # Область 1: правая (-half_angle_rad до +half_angle_rad)
        region_right_points = []
        for angle in np.linspace(-half_angle_rad, half_angle_rad, 30):
            for dr in [0, 5, 10]:
                r = r_outer + dr
                x = int(x0 + r * np.cos(angle))
                y = int(y0 + r * np.sin(angle))
                if 0 <= x < width and 0 <= y < height:
                    region_right_points.append(img[y, x])

        # Область 2: левая (np.pi - half_angle_rad до np.pi + half_angle_rad)
        region_left_points = []
        for angle in np.linspace(np.pi - half_angle_rad, np.pi + half_angle_rad, 30):
            for dr in [0, 5, 10]:
                r = r_outer + dr
                x = int(x0 + r * np.cos(angle))
                y = int(y0 + r * np.sin(angle))
                if 0 <= x < width and 0 <= y < height:
                    region_left_points.append(img[y, x])

        # Вычисляем средние интенсивности
        for points in [region_right_points, region_left_points]:
            if points:
                intensities.append(np.mean(points))
            else:
                intensities.append(255)  # Максимальная интенсивность если область пуста

        return intensities

    def find_outer_boundary(self, img, pupil_center, pupil_radius):
        """Находит внешнюю границу радужной оболочки с учетом параметрического угла"""
        img_grad = img.copy()
        img_grad[img_grad > 180] = 0

        grad_x, grad_y = self.gaussian_derivative(img_grad, self.sigma)
        height, width = img.shape
        best_score = -float('inf')
        best_params = None

        # Вычисляем половину угла для каждой области в радианах
        half_angle_deg = self.total_arc_angle_deg / 4.0
        half_angle_rad = np.deg2rad(half_angle_deg)

        # Диапазоны поиска
        r_min = int(pupil_radius * 1.5)
        r_max = int(min(width, height) * 0.5)

        # Перебор по радиусу и смещению центра
        for r in range(r_min, r_max + 1):
            for dx in range(-7, 8):
                for dy in range(-7, 8):
                    x0 = pupil_center[0] + dx
                    y0 = pupil_center[1] + dy

                    # Вычисляем интенсивности только в правой и левой областях
                    intensities = self.compute_region_intensities_two_regions(img, x0, y0, r)
                    total_intensity = sum(intensities)

                    # Рассчитываем углы для двух областей
                    angles_per_region = []
                    for I_i in intensities:
                        alpha_i = (self.total_arc_angle_deg * I_i) / total_intensity if total_intensity > 0 else half_angle_deg * 2
                        angles_per_region.append(np.deg2rad(alpha_i))

                    # Суммарный интеграл по двум областям
                    total_integral = 0
                    total_weight = 0

                    # Правая область: (-half_angle_rad до -half_angle_rad + angles_per_region[0])
                    start_angle_right = -half_angle_rad
                    end_angle_right = start_angle_right + angles_per_region[0]
                    if angles_per_region[0] > 0:
                        angles_right = np.linspace(start_angle_right, end_angle_right, max(5, int(angles_per_region[0] * 10)))
                        integral_right = self.compute_circular_integral(grad_x, grad_y, x0, y0, r, angles_right)
                        total_integral += integral_right * angles_per_region[0]
                        total_weight += angles_per_region[0]

                    # Левая область: (np.pi - half_angle_rad до np.pi - half_angle_rad + angles_per_region[1])
                    start_angle_left = np.pi - half_angle_rad
                    end_angle_left = start_angle_left + angles_per_region[1]
                    if angles_per_region[1] > 0:
                        angles_left = np.linspace(start_angle_left, end_angle_left, max(5, int(angles_per_region[1] * 10)))
                        integral_left = self.compute_circular_integral(grad_x, grad_y, x0, y0, r, angles_left)
                        total_integral += integral_left * angles_per_region[1]
                        total_weight += angles_per_region[1]

                    # Взвешенный средний интеграл
                    if total_weight > 0:
                        score = total_integral / total_weight

                        if score > best_score:
                            best_score = score
                            best_params = (x0, y0, r, intensities)

        return best_params

    def segment_iris(self, image_path, pupil_center, pupil_radius):
        """Полная сегментация радужки"""
        # Загрузка и предобработка изображения
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Не удалось загрузить изображение")

        # Поиск внешней границы
        outer_params = self.find_outer_boundary(img, pupil_center, pupil_radius)

        if outer_params is None:
            return None, None, None

        x0, y0, r_outer, intensities = outer_params

        # Создание масок
        mask_iris = np.zeros_like(img)
        mask_pupil = np.zeros_like(img)

        # Маска радужки (внешняя граница)
        cv2.circle(mask_iris, (x0, y0), r_outer, 255, -1)

        # Маска зрачка (внутренняя граница)
        cv2.circle(mask_pupil, pupil_center, pupil_radius, 255, -1)

        # Финальная маска радужки (внешняя минус внутренняя)
        mask_final = cv2.subtract(mask_iris, mask_pupil)

        # Сегментированное изображение
        segmented = cv2.bitwise_and(img, img, mask=mask_final)

        return segmented, mask_final, (pupil_center, pupil_radius, (x0, y0, r_outer))

    def get_pupil_center_from_iris_contours(self, iris_mask):
        """
        Получение центра зрачка через анализ контуров радужки
        Находит внутренний контур (граница радужки и зрачка)
        """
        binary_mask = iris_mask.astype(np.uint8) * 255

        h, w = binary_mask.shape[:2]

        alpha = h / 240
        betta = w / 320

        # Вариант 2 (рекомендуется): Запасной метод через центр масс
        M = cv2.moments(binary_mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (int(alpha * cx), int(betta * cy))
        else:
            h, w = binary_mask.shape[:2]
            return (int(alpha * w//2), int(betta * h//2))

    def validate_iris_mask(
        self,
        mask,
        min_area_ratio=0.01,
        max_area_ratio=0.3,
        min_circularity=0.4,
        max_components=3
    ):
        """
        Проверяет качество маски радужки и возвращает True, если маска пригодна для использования
        """
        # Нормализуем форму маски
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze()

        h, w = mask.shape[:2]
        total_pixels = h * w

        # Преобразуем в бинарную маску для обработки
        binary_mask = mask.astype(np.uint8) * 255

        # 1. Проверка площади (радужка должна занимать разумную долю изображения)
        area = np.count_nonzero(binary_mask)
        area_ratio = area / total_pixels

        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            return False

        # 2. Проверка количества связных компонентов
        num_labels, labels = cv2.connectedComponents(binary_mask.astype(np.uint8))
        # num_labels включает фон (0), поэтому реальное количество компонент = num_labels - 1
        num_components = num_labels - 1

        if num_components == 0:
            return False

        if num_components > max_components:
            return False

        # 3. Проверка формы основного компонента (компактность)
        # Находим самый большой компонент
        component_sizes = [np.sum(labels == i) for i in range(1, num_labels)]
        largest_idx = np.argmax(component_sizes) + 1

        # Создаем маску только для самого большого компонента
        largest_component = (labels == largest_idx).astype(np.uint8) * 255

        # Находим контуры
        contours, _ = cv2.findContours(largest_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        cnt = contours[0]
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            return False

        # Вычисляем круглость (circularity): 1 для идеального круга
        circularity = 4 * np.pi * area / (perimeter ** 2)

        if circularity < min_circularity:
            return False

        # 4. Проверка позиции центра (должен быть в центральной области)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = w // 2, h // 2

        # Проверяем, что центр в пределах 30-70% изображения
        if not (0.15 * w <= cx <= 0.85 * w and 0.15 * h <= cy <= 0.85 * h):
             print(f"Маска отклонена: центр вне допустимой области ({cx}, {cy})")
             return False
        return True

    def validate_pupil(self, image_path, pupil_center, pupil_radius, threshold=40):
        """
        Проверяет корректность определения зрачка по средней интенсивности

        Args:
            image_path: путь к изображению
            pupil_center: кортеж (x, y) с центром зрачка
            pupil_radius: радиус зрачка
            threshold: пороговое значение средней интенсивности (0-255)

        Returns:
            bool: True если зрачок определен корректно, иначе False
        """
        # Загружаем изображение в градациях серого
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False

        x, y = pupil_center
        r = pupil_radius

        # Создаем маску круга зрачка
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)

        # Получаем пиксели внутри зрачка
        pupil_pixels = img[mask > 0]

        if len(pupil_pixels) == 0:
            return False

        pupil_pixels[pupil_pixels > 180] = 0

        # Вычисляем среднюю интенсивность
        avg_intensity = np.mean(pupil_pixels)

        print(avg_intensity)
        # Зрачок должен быть темным, поэтому средняя интенсивность должна быть ниже порога
        return avg_intensity < threshold

    def daugman_circle_detection(self, image_path, iris_mask=None, use_projections=True):
        """
        Поиск круга зрачка методом Daugman с использованием маски радужки
        """

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Не удалось загрузить изображение")

        h, w = image.shape
        r_min, r_max = min(h, w)//20, min(h, w)//8

        cx_init, cy_init = w//2, h//2  # Значения по умолчанию
        center_range = 50  # Стандартный диапазон поиска
        mask_used = False  # Флаг использования маски

        # Шаг 1: Получение и валидация маски
        current_mask = None
        if iris_mask is not None:
            # Используем переданную маску
            current_mask = iris_mask
        else:
            # Предсказываем маску, если она не передана
            current_mask = self.predict_iris_mask(image_path)

        # Шаг 2: Проверка качества маски
        is_mask_valid = False
        if current_mask is not None:
            is_mask_valid = self.validate_iris_mask(current_mask)

            # Если маска плохая - сохраняем для отладки
            #if not is_mask_valid:
            #   timestamp = datetime.now().strftime("%H%M%S")
            #   bad_mask_path = f"bad_mask_{timestamp}.png"
             #  cv2.imwrite(bad_mask_path, (current_mask * 255).astype(np.uint8))

        # Шаг 3: Выбор метода определения центра
        if is_mask_valid:
            # Используем маску для поиска центра
            estimated_center = self.get_pupil_center_from_iris_contours(current_mask)
            cx_init, cy_init = estimated_center
            center_range = 20  # Узкий диапазон поиска
            mask_used = True
            print(f"Центр зрачка из ВАЛИДНОЙ маски: ({cx_init}, {cy_init})")
        else:
            print("Качество маски низкое, используем стандартный метод")
            # Стандартный метод без маски
            if use_projections:
                smoothed = cv2.bilateralFilter(image, 9, 75, 75)
                region = smoothed[h//4:3*h//4, w//4:3*w//4]

                # Вертикальная проекция (для поиска X) - axis=0
                v_proj = np.mean(region, axis=0)
                # Горизонтальная проекция (для поиска Y) - axis=1
                h_proj = np.mean(region, axis=1)

                cx_init = v_proj.argmin() + w//4  # Ищем минимум (зрачок темный)
                cy_init = h_proj.argmin() + h//4
                print(f"Центр из проекций: ({cx_init}, {cy_init})")
            else:
                cx_init, cy_init = w//2, h//2
                print(f"Центр по умолчанию: ({cx_init}, {cy_init})")

            center_range = 50  # Стандартный диапазон


        # Градиенты изображения
        grad_x, grad_y = self.gaussian_derivative(image, self.sigma)

        # Параметры поиска
        step_center = 1
        step_radius = 1  # Можно сделать шаг меньше благодаря точной оценке центра

        best_energy = -np.inf
        best_cx, best_cy, best_r = cx_init, cy_init, r_min

        # Перебор центров вблизи оцененного (теперь очень маленькая область!)
        for dx in range(-center_range, center_range+1, step_center):
            for dy in range(-center_range, center_range+1, step_center):
                cx = cx_init + dx
                cy = cy_init + dy

                # Проверка границ
                if cx < 0 or cx >= w or cy < 0 or cy >= h:
                    continue

                # Перебор радиусов
                for r in range(r_min, r_max+1, step_radius):
                    sample_count = max(360, int(2*np.pi*r))
                    alpha = np.linspace(0, 2*np.pi, sample_count, endpoint=False)
                    xs = np.round(cx + r * np.cos(alpha)).astype(int)
                    ys = np.round(cy + r * np.sin(alpha)).astype(int)

                    # Фильтруем точки внутри изображения
                    mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                    xs = xs[mask]
                    ys = ys[mask]

                    if len(xs) == 0:
                        continue

                    # Вычисление энергии по Daugman
                    normals = (xs - cx) * grad_x[ys, xs] + (ys - cy) * grad_y[ys, xs]
                    energy = np.sum(np.abs(normals))

                    if energy > best_energy:
                        best_energy = energy
                        best_cx, best_cy, best_r = cx, cy, r

        return best_cx, best_cy, best_r

    def normalize_iris(self, image_path, boundaries, normalized_width=64, normalized_height=512):
        """
        Нормализация радужки по методу Daugman's Rubber Sheet Model
        Преобразует кольцевую область радужки в прямоугольное представление

        Args:
            image_path: путь к исходному изображению
            boundaries: границы радужки и зрачка
            normalized_width: ширина нормализованного изображения (радиус)
            normalized_height: высота нормализованного изображения (угол)

        Returns:
            normalized_iris: нормализованное изображение радужки
        """
        # Загружаем исходное изображение
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Не удалось загрузить изображение")

        # Извлекаем параметры границ
        pupil_center, pupil_radius, iris_params = boundaries
        iris_center = (iris_params[0], iris_params[1])
        iris_radius = iris_params[2]

        # Создаем пустое нормализованное изображение
        normalized_iris = np.zeros((normalized_height, normalized_width), dtype=np.uint8)

        # Вычисляем смещение центров
        dx = iris_center[0] - pupil_center[0]
        dy = iris_center[1] - pupil_center[1]

        # Нормализация для каждого пикселя в нормализованном изображении
        for y in range(normalized_height):
            # Угол от 0 до 2π
            theta = 2 * np.pi * y / normalized_height

            # Вычисляем текущий радиус для этого угла с учетом смещения центров
            current_pupil_radius = pupil_radius
            current_iris_radius = iris_radius

            # Для каждого радиуса от 0 до 1 (от зрачка до радужки)
            for x in range(normalized_width):
                r = x / normalized_width  # от 0 (зрачок) до 1 (радужка)

                # Вычисляем координаты в исходном изображении
                pupil_x = pupil_center[0] + current_pupil_radius * np.cos(theta)
                pupil_y = pupil_center[1] + current_pupil_radius * np.sin(theta)

                iris_x = iris_center[0] + current_iris_radius * np.cos(theta)
                iris_y = iris_center[1] + current_iris_radius * np.sin(theta)

                # Интерполяция между границами зрачка и радужки
                src_x = pupil_x + r * (iris_x - pupil_x)
                src_y = pupil_y + r * (iris_y - pupil_y)

                # Билинейная интерполяция
                if (0 <= src_x < img.shape[1] and 0 <= src_y < img.shape[0]):
                    x1, y1 = int(src_x), int(src_y)
                    x2, y2 = min(x1 + 1, img.shape[1] - 1), min(y1 + 1, img.shape[0] - 1)

                    # Коэффициенты интерполяции
                    dx1 = src_x - x1
                    dy1 = src_y - y1
                    dx2 = 1 - dx1
                    dy2 = 1 - dy1

                    # Билинейная интерполяция
                    value = (img[y1, x1] * dx2 * dy2 +
                            img[y1, x2] * dx1 * dy2 +
                            img[y2, x1] * dx2 * dy1 +
                            img[y2, x2] * dx1 * dy1)

                    normalized_iris[y, x] = np.clip(value, 0, 255)

        return normalized_iris.T

    def enhance_normalized_iris(self, normalized_iris):
        """
        Улучшение нормализованного изображения радужки
        """
        # Применяем гистограммную эквализацию для улучшения контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(normalized_iris)

        # Легкое размытие для уменьшения шума
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

        return enhanced

    def visualize_boundaries(self, image_path, boundaries, output_path="boundaries_visualization.jpg"):
        """Визуализация границ на исходном изображении"""
        # Загружаем цветное изображение
        img_color = cv2.imread(image_path)
        if img_color is None:
            raise ValueError("Не удалось загрузить изображение")

        img_with_boundaries = img_color.copy()

        # Извлекаем параметры границ
        pupil_center, pupil_radius, iris_params = boundaries
        iris_center = (iris_params[0], iris_params[1])
        iris_radius = iris_params[2]

        # Рисуем границы разными цветами
        # Внешняя граница радужки - зеленый
        cv2.circle(img_with_boundaries, iris_center, iris_radius, (0, 255, 0), 2)
        cv2.circle(img_with_boundaries, iris_center, 2, (0, 255, 0), 3)  # Центр

        # Внутренняя граница (зрачок) - красный
        cv2.circle(img_with_boundaries, pupil_center, pupil_radius, (0, 0, 255), 2)
        cv2.circle(img_with_boundaries, pupil_center, 2, (0, 0, 255), 3)  # Центр

        # Линия, соединяющая центры - синяя
        cv2.line(img_with_boundaries, pupil_center, iris_center, (255, 0, 0), 1)

        # Добавляем подписи
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_with_boundaries, f"Pupil R: {pupil_radius}",
                   (10, 30), font, 0.6, (0, 0, 255), 2)
        cv2.putText(img_with_boundaries, f"Iris R: {iris_radius}",
                   (10, 60), font, 0.6, (0, 255, 0), 2)
        cv2.putText(img_with_boundaries, f"Arc Angle: {self.total_arc_angle_deg}°",
                   (10, 90), font, 0.6, (255, 255, 0), 2)

        # Сохраняем результат
        cv2.imwrite(output_path, img_with_boundaries)

        return img_with_boundaries

    def visualize_integration_arcs(self, image_path, pupil_center, pupil_radius,
                                   output_path="integration_arcs_two_regions.png"):
        """
        Визуализирует дуги, используемые при подсчёте интеграла
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Не удалось загрузить изображение для визуализации дуг")

        # Находим внешнюю границу
        grad_x, grad_y = self.gaussian_derivative(img, self.sigma)
        outer_params = self.find_outer_boundary(img, pupil_center, pupil_radius)

        if outer_params is None:
            raise ValueError("Не удалось найти внешнюю границу для визуализации")

        x0, y0, r_outer, intensities = outer_params

        # Вычисляем углы для двух областей в градусах
        total_intensity = sum(intensities)
        angles_deg = [(self.total_arc_angle_deg * I_i) / total_intensity if total_intensity > 0 else self.total_arc_angle_deg/2
                     for I_i in intensities]

        # Вычисляем половину угла для начальных позиций областей
        half_start_angle_deg = self.total_arc_angle_deg / 4.0

        # Создаём график
        plt.figure(figsize=(12, 10))
        plt.imshow(img, cmap='gray')
        ax = plt.gca()

        # Цвета для областей
        colors = ['lime', 'magenta']
        region_names = ['Правая область', 'Левая область']

        # Рисуем дуги для правой области
        start_angle_right_deg = -half_start_angle_deg
        end_angle_right_deg = start_angle_right_deg + angles_deg[0]
        arc_right = Arc((x0, y0), 2*r_outer, 2*r_outer,
                      angle=0, theta1=start_angle_right_deg, theta2=end_angle_right_deg,
                      edgecolor=colors[0], lw=2.5, label=f"{region_names[0]}: {angles_deg[0]:.1f}°")
        ax.add_patch(arc_right)

        # Рисуем дуги для левой области
        start_angle_left_deg = 180 - half_start_angle_deg
        end_angle_left_deg = start_angle_left_deg + angles_deg[1]
        arc_left = Arc((x0, y0), 2*r_outer, 2*r_outer,
                      angle=0, theta1=start_angle_left_deg, theta2=end_angle_left_deg,
                      edgecolor=colors[1], lw=2.5, label=f"{region_names[1]}: {angles_deg[1]:.1f}°")
        ax.add_patch(arc_left)

        # Рисуем центры и границы
        plt.scatter([x0], [y0], color='yellow', s=80, marker='x', linewidth=2, label='Центр радужки')
        plt.scatter([pupil_center[0]], [pupil_center[1]], color='cyan', s=80, marker='x', linewidth=2, label='Центр зрачка')
        circle_iris = plt.Circle((x0, y0), r_outer, color='yellow', fill=False, linestyle='--', linewidth=1.5, label='Граница радужки')
        ax.add_patch(circle_iris)
        circle_pupil = plt.Circle(pupil_center, pupil_radius, color='cyan', fill=False, linestyle='--', linewidth=1.5, label='Граница зрачка')
        ax.add_patch(circle_pupil)

        # Добавляем легенду и заголовок
        plt.legend(loc='best', fontsize=10)
        plt.title(f'Дуги интегрирования (суммарный угол: {self.total_arc_angle_deg}°)\n'
                 f'Правая область: {start_angle_right_deg:.1f}° до {end_angle_right_deg:.1f}°\n'
                 f'Левая область: {start_angle_left_deg:.1f}° до {end_angle_left_deg:.1f}°',
                 fontsize=12, pad=20)

        plt.axis('off')
        plt.tight_layout()

        # Сохраняем изображение
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

        return output_path

    def count_total_images(self, input_base_dir):
        """Подсчитывает общее количество изображений в датасете"""
        total_count = 0
        for subject_dir in glob.glob(os.path.join(input_base_dir, "*")):
            if not os.path.isdir(subject_dir):
                continue
            for eye_type in ['L', 'R']:
                eye_dir = os.path.join(subject_dir, eye_type)
                if os.path.exists(eye_dir):
                    total_count += len(glob.glob(os.path.join(eye_dir, "*.jpg")))
        return total_count

    def process_dataset_sequential(self, input_base_dir, output_base_dir, sigma=1, total_arc_angle_deg=180):
        """
        Последовательная обработка датасета с использованием GPU
        """
        # Обновляем параметры сегментатора
        self.sigma = sigma
        self.total_arc_angle_deg = total_arc_angle_deg

        # Создаем структуру папок для результатов
        os.makedirs(output_base_dir, exist_ok=True)

        # Счетчики для статистики
        total_images = 0
        successful_segmentations = 0

        # Подсчитываем общее количество изображений для прогресс-бара
        print("Подсчет общего количества изображений...")
        total_image_count = self.count_total_images(input_base_dir)
        print(f"Найдено изображений: {total_image_count}")
        print(f"Суммарный угол дуг интегрирования: {total_arc_angle_deg}°")

        # Выводим информацию о GPU
        gpus = tf.config.get_visible_devices('GPU')
        if gpus:
            print(f"Используется GPU:{self.gpu_id} для обработки")
        else:
            print("Используется CPU для обработки")

        # Создаем прогресс-бар
        pbar = tqdm(total=total_image_count, desc="Обработка датасета", unit="image")

        # Проходим по всем папкам с изображениями
        for subject_dir in glob.glob(os.path.join(input_base_dir, "*")):
            if not os.path.isdir(subject_dir):
                continue

            subject_id = os.path.basename(subject_dir)

            # Обрабатываем левые и правые глаза
            for eye_type in ['L', 'R']:
                eye_dir = os.path.join(subject_dir, eye_type)
                if not os.path.exists(eye_dir):
                    continue

                # Создаем соответствующие папки для результатов
                output_subject_dir = os.path.join(output_base_dir, subject_id, eye_type)
                os.makedirs(output_subject_dir, exist_ok=True)

                # Обрабатываем все изображения в папке
                for img_path in glob.glob(os.path.join(eye_dir, "*.jpg")):
                    total_images += 1
                    img_name = os.path.basename(img_path)

                    try:
                        # Обновляем описание прогресс-бара
                        pbar.set_description(f"Обработка: {subject_id}/{eye_type}/{img_name}")

                        # Используем существующий сегментатор вместо создания нового
                        # Все операции с моделью выполняются на GPU
                        with tf.device(f'/GPU:{self.gpu_id}'):
                            # Сегментация
                            x, y, pupil_radius = self.daugman_circle_detection(img_path)

                        pupil_center = (x, y)

                        # Проверяем корректность зрачка перед сегментацией
                        if not self.validate_pupil(img_path, pupil_center, pupil_radius):
                            print(f"Зрачок не прошел валидацию для {img_path}, пропускаем")
                            pbar.update(1)
                            continue

                        # Сегментация (без использования модели, можно на CPU)
                        segmented, mask, boundaries = self.segment_iris(img_path, pupil_center, pupil_radius)

                        if segmented is not None:
                            successful_segmentations += 1

                            # Визуализация границ
                            boundaries_path = os.path.join(output_subject_dir, f"{os.path.splitext(img_name)[0]}_boundaries.jpg")
                            img_with_boundaries = self.visualize_boundaries(img_path, boundaries, boundaries_path)

                            # Нормализация радужки
                            normalized_iris = self.normalize_iris(img_path, boundaries)
                            enhanced_iris = self.enhance_normalized_iris(normalized_iris)

                            # Сохраняем результаты
                            base_name = os.path.splitext(img_name)[0]

                            cv2.imwrite(os.path.join(output_subject_dir, f"{base_name}_segmented.jpg"), segmented)
                            cv2.imwrite(os.path.join(output_subject_dir, f"{base_name}_mask.jpg"), mask)
                            cv2.imwrite(os.path.join(output_subject_dir, f"{base_name}_normalized.jpg"), normalized_iris)
                            cv2.imwrite(os.path.join(output_subject_dir, f"{base_name}_enhanced.jpg"), enhanced_iris)

                            # Визуализация дуг интегрирования (если нужно)
                            try:
                                arcs_path = os.path.join(output_subject_dir, f"{base_name}_integration_arcs.png")
                                self.visualize_integration_arcs(
                                    image_path=img_path,
                                    pupil_center=pupil_center,
                                    pupil_radius=pupil_radius,
                                    output_path=arcs_path
                                )
                            except Exception as e:
                                print(f"Предупреждение: не удалось создать визуализацию дуг для {img_path}: {str(e)}")
                        else:
                            print(f"Не удалось сегментировать: {img_path}")

                    except Exception as e:
                        print(f"Ошибка при обработке {img_path}: {str(e)}")

                    pbar.update(1)

        # Закрываем прогресс-бар
        pbar.close()

        print(f"\nОбработка завершена!")
        print(f"Всего изображений: {total_images}")
        print(f"Успешно сегментировано: {successful_segmentations}")
        print(f"Процент успеха: {(successful_segmentations/total_images)*100:.2f}%")


# Пример использования
def main():
    # Проверка существования файла модели
    model_path = "/home/a.ivanov/mdl_wts.keras"
    if not os.path.exists(model_path):
        print(f"⚠️ Файл модели {model_path} не найден!")
        print("🔄 Будет создана базовая модель...")
        model_path = None

    print(f"Текущая рабочая директория: {os.getcwd()}")
    # # Параметры (нужно определить зрачок заранее)
    #image_path = "/home/flex/Desktop/Diplom/diplom/datasets/CASIA-Iris-Thousand/108/L/S5108L05.jpg"

    # # Создание сегментатора с параметрическим углом (140 градусов)
    segmenter = IrisSegmenter(model_path, sigma=2, total_arc_angle_deg=140, gpu_id=5)
    # #
    # # # Обнаружение зрачка
    # x, y, pupil_radius = segmenter.daugman_circle_detection(image_path)
    # pupil_center = (x, y)
    #
    # # Валидация зрачка
    # if not segmenter.validate_pupil(image_path, pupil_center, pupil_radius):
    #     print(f"Зрачок не прошел валидацию для {image_path}")
    #     # Можно попробовать альтернативные методы или пропустить изображение
    # else:
    #     print("✅ Зрачок прошел валидацию!")
    #
    # # Сегментация
    # segmented, mask, boundaries = segmenter.segment_iris(image_path, (x, y), pupil_radius)
    #
    # if segmented is not None:
    #     normalized_iris = segmenter.normalize_iris(image_path, boundaries)
    #     enhanced_iris = segmenter.enhance_normalized_iris(normalized_iris)
    #
    #     # Визуализируем дуги интегрирования
    #     arc_visualization_path = segmenter.visualize_integration_arcs(
    #             image_path=image_path,
    #             pupil_center=pupil_center,
    #             pupil_radius=pupil_radius,
    #             output_path="integration_arcs_custom_angle.png"
    #     )
    #
    #     print(f"Визуализация успешно создана: {arc_visualization_path}")
    #
    #     # Визуализация границ
    #     img_with_boundaries = segmenter.visualize_boundaries(image_path, boundaries, "boundaries_custom_angle.jpg")
    #
    #     # Создание комплексной визуализации
    #     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    #
    #     # Исходное изображение
    #     img_original = cv2.imread(image_path)
    #     img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    #     axes[0, 0].imshow(img_original)
    #     axes[0, 0].set_title("Исходное изображение")
    #     axes[0, 0].axis('off')
    #
    #     # Изображение с границами
    #     img_boundaries_rgb = cv2.cvtColor(img_with_boundaries, cv2.COLOR_BGR2RGB)
    #     axes[0, 1].imshow(img_boundaries_rgb)
    #     axes[0, 1].set_title(f"Границы радужки и зрачка\nСуммарный угол: {segmenter.total_arc_angle_deg}°")
    #     axes[0, 1].axis('off')
    #
    #     # Сегментированная радужка
    #     axes[1, 0].imshow(segmented, cmap='gray')
    #     axes[1, 0].set_title("Сегментированная радужка")
    #     axes[1, 0].axis('off')
    #
    #     # Маска
    #     axes[1, 1].imshow(mask, cmap='gray')
    #     axes[1, 1].set_title("Маска радужки")
    #     axes[1, 1].axis('off')
    #
    #     plt.tight_layout()
    #     plt.savefig("iris_circles_custom_angle.png")
    #
    #     # Сохранение результатов
    #     cv2.imwrite("segmented_iris_custom_angle.jpg", segmented)
    #     cv2.imwrite("iris_mask_custom_angle.jpg", mask)
    #     cv2.imwrite("iris_normalized_custom_angle.jpg", normalized_iris)
    #     cv2.imwrite("iris_enhanced_custom_angle.jpg", enhanced_iris)
    #
    #     print("✅ Сегментация завершена успешно!")
    #     print(f"Границы: зрачок {boundaries[0]}, R={boundaries[1]}, радужка {boundaries[2][:2]}, R={boundaries[2][2]}")
    #     print(f"Использован суммарный угол дуг: {segmenter.total_arc_angle_deg}°")
    # else:
    #     print("❌ Сегментация не удалась")

    # Пример обработки датасета с нестандартным углом
    input_dir = "./CASIA-TH-ALL"
    #input_dir = "/home/flex/Desktop/Diplom/diplom/datasets/CASIA-Iris-Thousand"
    output_dir = "./CASIA-Iris-Training"
    print("\nЗапуск обработки с суммарным углом 140°...")
    segmenter.process_dataset_sequential(input_dir, output_dir, sigma=2, total_arc_angle_deg=140)


if __name__ == "__main__":
    main()
