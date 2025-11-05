import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


class IrisSegmenter:
    def __init__(self, sigma=2):
        self.sigma = sigma

    def gaussian_derivative(self, img, sigma):
        """Вычисляет производную по радиусу от гауссова размытия"""
        # Гауссово размытие
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)

        # Вычисление градиента (производной)
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

        self.grad_x = grad_x
        self.grad_y = grad_y

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

    def compute_region_intensities(self, img, x0, y0, r_outer):
        """Вычисляет средние интенсивности в 4 прямоугольных областях"""
        height, width = img.shape

        # Определяем 4 прямоугольные области вокруг предполагаемой внешней границы
        regions = []

        # Область 1: верхняя (0-90 градусов)
        region1_points = []
        for angle in np.linspace(0, np.pi/2, 30):
            for dr in [0, 5, 10]:  # Несколько точек вдоль радиуса
                r = r_outer + dr
                x = int(x0 + r * np.cos(angle))
                y = int(y0 + r * np.sin(angle))
                if 0 <= x < width and 0 <= y < height:
                    region1_points.append(img[y, x])

        # Область 2: правая (90-180 градусов)
        region2_points = []
        for angle in np.linspace(np.pi/2, np.pi, 30):
            for dr in [0, 5, 10]:
                r = r_outer + dr
                x = int(x0 + r * np.cos(angle))
                y = int(y0 + r * np.sin(angle))
                if 0 <= x < width and 0 <= y < height:
                    region2_points.append(img[y, x])

        # Область 3: нижняя (180-270 градусов)
        region3_points = []
        for angle in np.linspace(np.pi, 3*np.pi/2, 30):
            for dr in [0, 5, 10]:
                r = r_outer + dr
                x = int(x0 + r * np.cos(angle))
                y = int(y0 + r * np.sin(angle))
                if 0 <= x < width and 0 <= y < height:
                    region3_points.append(img[y, x])

        # Область 4: левая (270-360 градусов)
        region4_points = []
        for angle in np.linspace(3*np.pi/2, 2*np.pi, 30):
            for dr in [0, 5, 10]:
                r = r_outer + dr
                x = int(x0 + r * np.cos(angle))
                y = int(y0 + r * np.sin(angle))
                if 0 <= x < width and 0 <= y < height:
                    region4_points.append(img[y, x])

        # Вычисляем средние интенсивности
        intensities = []
        for points in [region1_points, region2_points, region3_points, region4_points]:
            if points:
                intensities.append(np.mean(points))
            else:
                intensities.append(255)  # Максимальная интенсивность если область пуста

        return intensities

    def find_outer_boundary(self, img, pupil_center, pupil_radius):
        """Находит внешнюю границу радужной оболочки"""
        grad_x, grad_y = self.gaussian_derivative(img, self.sigma)

        height, width = img.shape
        best_score = -float('inf')
        best_params = None

        # Диапазоны поиска
        r_min = int(pupil_radius * 1.6)
        r_max = int(min(width, height) * 0.7)

        # Перебор по радиусу и смещению центра
        for r in range(r_min, r_max + 1):
            for dx in range(-5, 6):  # Смещение по x
                for dy in range(-5, 6):  # Смещение по y
                    x0 = pupil_center[0] + dx
                    y0 = pupil_center[1] + dy

                    # Вычисляем интенсивности в 4 областях
                    intensities = self.compute_region_intensities(img, x0, y0, r)
                    total_intensity = sum(intensities)

                    # Вычисляем углы для каждой области согласно формуле
                    angles_per_region = []
                    for I_i in intensities:
                        alpha_i = (90 * I_i) / total_intensity if total_intensity > 0 else 22.5
                        angles_per_region.append(np.deg2rad(alpha_i))

                    # Суммарный интеграл по всем областям
                    total_integral = 0
                    total_weight = 0

                    # Область 1: верхняя (0-90°)
                    if angles_per_region[0] > 0:
                        angles1 = np.linspace(0, angles_per_region[0], max(5, int(angles_per_region[0] * 10)))
                        integral1 = self.compute_circular_integral(grad_x, grad_y, x0, y0, r, angles1)
                        total_integral += integral1 * angles_per_region[0]
                        total_weight += angles_per_region[0]

                    # Область 2: правая (90-180°)
                    if angles_per_region[1] > 0:
                        angles2 = np.linspace(np.pi/2, np.pi/2 + angles_per_region[1],
                                            max(5, int(angles_per_region[1] * 10)))
                        integral2 = self.compute_circular_integral(grad_x, grad_y, x0, y0, r, angles2)
                        total_integral += integral2 * angles_per_region[1]
                        total_weight += angles_per_region[1]

                    # Область 3: нижняя (180-270°)
                    if angles_per_region[2] > 0:
                        angles3 = np.linspace(np.pi, np.pi + angles_per_region[2],
                                            max(5, int(angles_per_region[2] * 10)))
                        integral3 = self.compute_circular_integral(grad_x, grad_y, x0, y0, r, angles3)
                        total_integral += integral3 * angles_per_region[2]
                        total_weight += angles_per_region[2]

                    # Область 4: левая (270-360°)
                    if angles_per_region[3] > 0:
                        angles4 = np.linspace(3*np.pi/2, 3*np.pi/2 + angles_per_region[3],
                                            max(5, int(angles_per_region[3] * 10)))
                        integral4 = self.compute_circular_integral(grad_x, grad_y, x0, y0, r, angles4)
                        total_integral += integral4 * angles_per_region[3]
                        total_weight += angles_per_region[3]

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

    def daugman_circle_detection(self, image_path, estimated_center=None):
        """
        Поиск круга (зрачок или радужка) методом Daugman.
        Возвращает: (cx, cy, radius, result_image)
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Не удалось загрузить изображение")

        h, w = image.shape

        r_min, r_max = min(h, w)//16, min(h, w)//8

        if estimated_center is None:
            cx_init, cy_init = w//2, h//2
        else:
            cx_init, cy_init = estimated_center

        # Градиенты изображения
        grad_x, grad_y = self.gaussian_derivative(image, self.sigma)

        # Параметры поиска
        step_center = 3   # шаг перебора центра в пикселях
        step_radius = 2   # шаг радиуса
        center_range = 200  # диапазон смещения центра

        best_energy = -np.inf
        best_cx, best_cy, best_r = cx_init, cy_init, r_min
        best_xs, best_ys = None, None

        # Перебор центров вокруг примерного
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

                    energy = np.sum(np.abs(grad_x[ys, xs]) + np.abs(grad_y[ys, xs]))

                    if energy > best_energy:
                        best_energy = energy
                        best_cx, best_cy, best_r = cx, cy, r
                        best_xs, best_ys = xs, ys

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
            # (эллиптическая модель вместо круговой)
            current_pupil_radius = pupil_radius
            current_iris_radius = iris_radius

            # Для каждого радиуса от 0 до 1 (от зрачка до радужки)
            for x in range(normalized_width):
                r = x / normalized_width  # от 0 (зрачок) до 1 (радужка)

                # Вычисляем координаты в исходном изображении
                # Учитываем смещение центров для более точного отображения
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

        # Сохраняем результат
        cv2.imwrite(output_path, img_with_boundaries)

        return img_with_boundaries


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


    def _process_single_image(self, args):
        """
        Обрабатывает одно изображение (внутренний метод для многопроцессорной обработки)

        Args:
            args: кортеж (img_path, output_base_dir, sigma)

        Returns:
            result: словарь с результатами обработки
        """
        img_path, output_base_dir, sigma = args

        try:
            # Создаем сегментатор для каждого процесса
            segmenter = IrisSegmenter(sigma=sigma)

            # Определяем структуру выходных папок
            parts = img_path.split(os.sep)
            subject_id = parts[-3]  # ID субъекта
            eye_type = parts[-2]    # L или R
            img_name = parts[-1]    # имя файла

            # Создаем выходную директорию
            output_subject_dir = os.path.join(output_base_dir, subject_id, eye_type)
            os.makedirs(output_subject_dir, exist_ok=True)

            # Обрабатываем изображение
            x, y, pupil_radius = segmenter.daugman_circle_detection(img_path)
            segmented, mask, boundaries = segmenter.segment_iris(img_path, (x, y), pupil_radius)

            if segmented is not None:
                # Нормализация радужки
                normalized_iris = segmenter.normalize_iris(img_path, boundaries)
                enhanced_iris = segmenter.enhance_normalized_iris(normalized_iris)

                # Сохраняем результаты
                base_name = os.path.splitext(img_name)[0]

                cv2.imwrite(os.path.join(output_subject_dir, f"{base_name}_segmented.jpg"), segmented)
                cv2.imwrite(os.path.join(output_subject_dir, f"{base_name}_mask.jpg"), mask)
                cv2.imwrite(os.path.join(output_subject_dir, f"{base_name}_normalized.jpg"), normalized_iris)
                cv2.imwrite(os.path.join(output_subject_dir, f"{base_name}_enhanced.jpg"), enhanced_iris)

                return {"status": "success", "path": img_path}
            else:
                return {"status": "failed", "path": img_path, "error": "Сегментация не удалась"}

        except Exception as e:
            return {"status": "error", "path": img_path, "error": str(e)}

    def process_dataset_parallel(self, input_base_dir, output_base_dir, sigma=1, num_processes=None):
        """
        Обрабатывает весь датасет параллельно с использованием многопроцессорности

        Args:
            input_base_dir: Базовая директория датасета
            output_base_dir: Базовая директория для результатов
            sigma: параметр для сегментатора
            num_processes: количество процессов (по умолчанию = количество ядер CPU)
        """
        if num_processes is None:
            num_processes = mp.cpu_count()

        print(f"Используется {num_processes} процессов для обработки")

        # Создаем структуру папок для результатов
        os.makedirs(output_base_dir, exist_ok=True)

        # Собираем все пути к изображениям
        print("Сбор путей к изображениям...")
        image_paths = []
        for subject_dir in glob.glob(os.path.join(input_base_dir, "*")):
            if not os.path.isdir(subject_dir):
                continue

            subject_id = os.path.basename(subject_dir)

            for eye_type in ['L', 'R']:
                eye_dir = os.path.join(subject_dir, eye_type)
                if not os.path.exists(eye_dir):
                    continue

                # Собираем все изображения
                for img_path in glob.glob(os.path.join(eye_dir, "*.jpg")):
                    image_paths.append(img_path)

        total_images = len(image_paths)
        print(f"Найдено изображений: {total_images}")

        if total_images == 0:
            print("Изображения не найдены!")
            return

        # Подготавливаем аргументы для многопроцессорной обработки
        process_args = [(img_path, output_base_dir, sigma) for img_path in image_paths]

        # Счетчики для статистики
        successful_segmentations = 0
        failed_segmentations = 0
        errors = 0

        # Многопроцессорная обработка с прогресс-баром
        print("Запуск параллельной обработки...")
        with mp.Pool(processes=num_processes) as pool:
            # Используем imap для получения результатов по мере готовности
            results = list(tqdm(
                pool.imap(self._process_single_image, process_args),
                total=total_images,
                desc="Обработка датасета",
                unit="image"
            ))

        # Собираем статистику
        for result in results:
            if result["status"] == "success":
                successful_segmentations += 1
            elif result["status"] == "failed":
                failed_segmentations += 1
            else:
                errors += 1

        # Выводим статистику
        print(f"\nОбработка завершена!")
        print(f"Всего изображений: {total_images}")
        print(f"Успешно сегментировано: {successful_segmentations}")
        print(f"Не удалось сегментировать: {failed_segmentations}")
        print(f"Ошибок обработки: {errors}")
        print(f"Процент успеха: {(successful_segmentations/total_images)*100:.2f}%")

        # Сохраняем отчет
        self._save_processing_report(output_base_dir, results)

    def _save_processing_report(self, output_base_dir, results):
        """Сохраняет отчет о обработке"""
        report_path = os.path.join(output_base_dir, "processing_report.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Отчет о обработке датасета\n")
            f.write("=" * 50 + "\n\n")

            successful = [r for r in results if r["status"] == "success"]
            failed = [r for r in results if r["status"] == "failed"]
            errors = [r for r in results if r["status"] == "error"]

            f.write(f"Всего изображений: {len(results)}\n")
            f.write(f"Успешно обработано: {len(successful)}\n")
            f.write(f"Не удалось сегментировать: {len(failed)}\n")
            f.write(f"Ошибок обработки: {len(errors)}\n")
            f.write(f"Процент успеха: {(len(successful)/len(results))*100:.2f}%\n\n")

            if errors:
                f.write("Ошибки обработки:\n")
                f.write("-" * 30 + "\n")
                for error in errors:
                    f.write(f"{error['path']}: {error['error']}\n")

            if failed:
                f.write("\nНе удалось сегментировать:\n")
                f.write("-" * 30 + "\n")
                for fail in failed:
                    f.write(f"{fail['path']}\n")

    def process_dataset_sequential(self, input_base_dir, output_base_dir, sigma=1):
        """
        Последовательная обработка датасета (оригинальный метод)
        """
        # Создаем структуру папок для результатов
        os.makedirs(output_base_dir, exist_ok=True)

        # Счетчики для статистики
        total_images = 0
        successful_segmentations = 0

        # Подсчитываем общее количество изображений для прогресс-бара
        print("Подсчет общего количества изображений...")
        total_image_count = self.count_total_images(input_base_dir)
        print(f"Найдено изображений: {total_image_count}")

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

                        # Сегментация
                        x, y, pupil_radius = self.daugman_circle_detection(img_path)
                        segmented, mask, boundaries = self.segment_iris(img_path, (x, y), pupil_radius)

                        if segmented is not None:
                            successful_segmentations += 1

                            # Нормализация радужки
                            normalized_iris = self.normalize_iris(img_path, boundaries)
                            enhanced_iris = self.enhance_normalized_iris(normalized_iris)

                            # Сохраняем результаты
                            base_name = os.path.splitext(img_name)[0]

                            cv2.imwrite(os.path.join(output_subject_dir, f"{base_name}_segmented.jpg"), segmented)
                            cv2.imwrite(os.path.join(output_subject_dir, f"{base_name}_mask.jpg"), mask)
                            cv2.imwrite(os.path.join(output_subject_dir, f"{base_name}_normalized.jpg"), normalized_iris)
                            cv2.imwrite(os.path.join(output_subject_dir, f"{base_name}_enhanced.jpg"), enhanced_iris)

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
    # Параметры (нужно определить зрачок заранее)
    image_path = "/home/flex/Desktop/Diplom/diplom/datasets/CASIA-Iris-Thousand/624/L/S5624L01.jpg"

    # Создание сегментатора
    segmenter = IrisSegmenter(sigma=1)
    x, y, pupil_radius = segmenter.daugman_circle_detection(image_path)

    # Сегментация
    segmented, mask, boundaries = segmenter.segment_iris(image_path, (x, y), pupil_radius)
    normalized_iris = segmenter.normalize_iris(image_path, boundaries)
    enhanced_iris = segmenter.enhance_normalized_iris(normalized_iris)

    if segmented is not None:
        # Визуализация границ
        img_with_boundaries = segmenter.visualize_boundaries(image_path, boundaries)

        # Создание комплексной визуализации
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Исходное изображение
        img_original = cv2.imread(image_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(img_original)
        axes[0, 0].set_title("Исходное изображение")
        axes[0, 0].axis('off')

        # Изображение с границами
        img_boundaries_rgb = cv2.cvtColor(img_with_boundaries, cv2.COLOR_BGR2RGB)
        axes[0, 1].imshow(img_boundaries_rgb)
        axes[0, 1].set_title("Границы радужки и зрачка")
        axes[0, 1].axis('off')

        # Сегментированная радужка
        axes[1, 0].imshow(segmented, cmap='gray')
        axes[1, 0].set_title("Сегментированная радужка")
        axes[1, 0].axis('off')

        # Маска
        axes[1, 1].imshow(mask, cmap='gray')
        axes[1, 1].set_title("Маска радужки")
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

        # Сохранение результатов
        cv2.imwrite("segmented_iris.jpg", segmented)
        cv2.imwrite("iris_mask.jpg", mask)
        cv2.imwrite("boundaries_visualization.jpg", img_with_boundaries)
        cv2.imwrite(f"iris_normalized.jpg", normalized_iris)

        # Улучшенная нормализованная радужка
        cv2.imwrite(f"iris_enhanced.jpg", enhanced_iris)

        print("Сегментация завершена успешно!")
        print(f"Границы: зрачок {boundaries[0]}, R={boundaries[1]}, радужка {boundaries[2][:2]}, R={boundaries[2][2]}")
    else:
        print("Сегментация не удалась")

    input_dir = "/home/flex/Desktop/Diplom/diplom/datasets/CASIA-Iris-Thousand"
    output_dir = "/home/flex/Desktop/Diplom/diplom/datasets/CASIA-Iris-Thousand-Segmented"

    print("\nЗапуск параллельной обработки...")
    segmenter.process_dataset_parallel(input_dir, output_dir, sigma=1, num_processes=8)

if __name__ == "__main__":
    main()
