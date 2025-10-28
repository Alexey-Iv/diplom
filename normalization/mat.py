import cv2
import numpy as np


def daugman_circle_detection(image, feature="pupil", estimated_center=None, debug=False):
    """
    Поиск круга (зрачок или радужка) методом Daugman.
    Возвращает: (cx, cy, radius, result_image)
    """
    # Конвертируем в серый
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape

    # Настройка диапазона радиусов и центра
    if feature == "pupil":
        r_min, r_max = min(h, w)//15, min(h, w)//8
    elif feature == "limbus":
        r_min, r_max = min(h, w)//10, min(h, w)//2
    else:
        raise ValueError("Unknown feature")

    print(r_min, r_max)
    if estimated_center is None:
        cx_init, cy_init = w//2, h//2
    else:
        cx_init, cy_init = estimated_center

    # Градиенты изображения
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Параметры поиска
    step_center = 2   # шаг перебора центра в пикселях
    step_radius = 1   # шаг радиуса
    center_range = 50  # диапазон смещения центра

    if feature == 'limbus':
        center_range = 10

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

    # Визуализация точек
    result_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if debug and best_xs is not None:
        for px, py in zip(best_xs, best_ys):
            cv2.circle(result_img, (px, py), 1, (0, 0, 255), -1)
        cv2.imshow(f"Sample Points ({feature})", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return best_cx, best_cy, best_r, result_img


# ===== Основной блок =====
if __name__ == "__main__":
    img = cv2.imread("/home/flex/Desktop/Diplom/diplom/datasets/CASIA-Iris-Thousand/127/L/S5127L01.jpg")

    # Найдём зрачок
    px, py, pr, result_pupil = daugman_circle_detection(img, feature="pupil", debug=True)
    print("Pupil:", px, py, pr)

    # Найдём радужку
    ix, iy, ir, result_iris = daugman_circle_detection(img, feature="limbus", estimated_center=(px, py), debug=True)
    print("Iris:", ix, iy, ir)

    # Совместим оба результата
    combined = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    cv2.circle(combined, (px, py), pr, (255, 0, 0), 2)
    cv2.circle(combined, (ix, iy), ir, (0, 255, 0), 2)

    cv2.imshow("Final Segmentation", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
