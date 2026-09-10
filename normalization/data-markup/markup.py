#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-mode iris/eyelid annotator.

Modes
-----
1) circles
    - 3 points on the pupil + 3 points on the iris.
    - Fits two circles and saves a normalized iris strip.

2) eyelids
    - Runs a trained YOLO Detect model first and displays predicted pupil/iris circles.
    - 3 points on the upper eyelid: image-left, center, image-right.
    - 3 points on the lower eyelid: image-left, center, image-right.
    - Fits a quadratic curve through each triplet.
    - Saves a binary source-image mask: 255 only between the eyelids.
    - Normalizes the iris image and the eyelid mask with identical cv2.remap maps.
    - Saves an Ultralytics YOLO pose label with 6 keypoints.

The point order is always based on IMAGE coordinates, not anatomical left/right:
upper_left, upper_center, upper_right, lower_left, lower_center, lower_right.
This is important for horizontal flips during YOLO pose training.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

CIRCLE_LABELS = [
    "1/6: pupil point 1",
    "2/6: pupil point 2",
    "3/6: pupil point 3",
    "4/6: iris point 1",
    "5/6: iris point 2",
    "6/6: iris point 3",
]

EYELID_LABELS = [
    "1/6: UPPER eyelid - image LEFT",
    "2/6: UPPER eyelid - CENTER",
    "3/6: UPPER eyelid - image RIGHT",
    "4/6: LOWER eyelid - image LEFT",
    "5/6: LOWER eyelid - CENTER",
    "6/6: LOWER eyelid - image RIGHT",
]

POINT_COLORS = [
    (0, 255, 255),
    (0, 165, 255),
    (0, 255, 0),
    (255, 0, 255),
    (255, 0, 0),
    (255, 0, 200),
]


@dataclass
class AppConfig:
    mode: str
    input_dir: Path
    output_dir: Path
    log_file: Path
    normalization_mode: str = "polar_iris"
    radial_res: int = 64
    angular_res: int = 512
    screen_margin: int = 80
    grab_radius: int = 12
    dataset_root: Path | None = None
    split: str = "train"
    copy_images: bool = True
    full_image_box: bool = True
    circle_model_path: Path | None = None
    circle_conf: float = 0.25
    circle_iou: float = 0.70
    circle_imgsz: int = 640
    pupil_class: str = "pupil"
    iris_class: str = "iris"
    prediction_device: int | str = 0
    auto_eyelid_points: bool = True
    snap_endpoints_to_iris: bool = True
    require_circle_prediction: bool = False


def circle_from_3_points(p1: Sequence[float], p2: Sequence[float], p3: Sequence[float]):
    """Return ((cx, cy), radius) for the circle through three points."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-6:
        return None

    ux = (
        (x1**2 + y1**2) * (y2 - y3)
        + (x2**2 + y2**2) * (y3 - y1)
        + (x3**2 + y3**2) * (y1 - y2)
    ) / d
    uy = (
        (x1**2 + y1**2) * (x3 - x2)
        + (x2**2 + y2**2) * (x1 - x3)
        + (x3**2 + y3**2) * (x2 - x1)
    ) / d

    return (float(ux), float(uy)), float(np.hypot(x1 - ux, y1 - uy))




def _class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names[int(class_id)])
    return str(names[int(class_id)])


def _best_box_for_class(result, target_name: str):
    if result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError("Circle model returned no detections")

    best = None
    for box in result.boxes:
        cls_id = int(box.cls.item())
        if _class_name(result.names, cls_id) != target_name:
            continue
        confidence = float(box.conf.item())
        xyxy = box.xyxy[0].detach().cpu().numpy().astype(np.float64)
        if best is None or confidence > best[0]:
            best = (confidence, xyxy)

    if best is None:
        available = sorted(
            {_class_name(result.names, int(box.cls.item())) for box in result.boxes}
        )
        raise RuntimeError(
            f"Class '{target_name}' not found. Predicted classes: {available}"
        )
    return best


def _circle_from_box(xyxy):
    x1, y1, x2, y2 = map(float, xyxy)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    radius = ((x2 - x1) + (y2 - y1)) / 4.0
    return (cx, cy), radius


def predict_circles_from_yolo(
    model, image, conf=0.25, iou=0.70, imgsz=640,
    pupil_class="pupil", iris_class="iris", device=0,
):
    """Predict pupil/iris circles from the best bbox of each configured class."""
    result = model.predict(
        image, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False
    )[0]
    pupil_conf, pupil_box = _best_box_for_class(result, pupil_class)
    iris_conf, iris_box = _best_box_for_class(result, iris_class)
    pupil_center, pupil_radius = _circle_from_box(pupil_box)
    iris_center, iris_radius = _circle_from_box(iris_box)

    if pupil_radius <= 0 or iris_radius <= pupil_radius:
        raise RuntimeError(
            f"Invalid predicted radii: pupil={pupil_radius:.2f}, iris={iris_radius:.2f}"
        )

    return {
        "pupil_center": tuple(map(float, pupil_center)),
        "pupil_radius": float(pupil_radius),
        "iris_center": tuple(map(float, iris_center)),
        "iris_radius": float(iris_radius),
        "pupil_conf": float(pupil_conf),
        "iris_conf": float(iris_conf),
    }


def project_point_to_circle(point, center, radius):
    """Project a point to the nearest point on a circle."""
    px, py = map(float, point)
    cx, cy = map(float, center)
    dx, dy = px - cx, py - cy
    norm = float(np.hypot(dx, dy))
    if norm < 1e-8:
        return cx + radius, cy
    scale = radius / norm
    return cx + dx * scale, cy + dy * scale


def suggest_eyelid_points(circle_geometry, image_shape):
    """Create editable initial eyelid landmarks from the predicted iris geometry."""
    h, w = image_shape[:2]
    cx, cy = circle_geometry["iris_center"]
    r = circle_geometry["iris_radius"]

    def on_iris(degrees):
        theta = np.deg2rad(degrees)
        return (
            float(np.clip(cx + r * np.cos(theta), 0, w - 1)),
            float(np.clip(cy + r * np.sin(theta), 0, h - 1)),
        )

    # Endpoints lie on the iris circle; center landmarks remain free/editable.
    upper_left = on_iris(210.0)
    upper_right = on_iris(330.0)
    lower_left = on_iris(150.0)
    lower_right = on_iris(30.0)
    upper_center = (float(np.clip(cx, 0, w - 1)), float(np.clip(cy - 0.62 * r, 0, h - 1)))
    lower_center = (float(np.clip(cx, 0, w - 1)), float(np.clip(cy + 0.55 * r, 0, h - 1)))
    return [upper_left, upper_center, upper_right, lower_left, lower_center, lower_right]


def ray_circle_intersection(origin, direction, center, radius):
    ox, oy = origin
    dx, dy = direction
    cx, cy = center

    fx = ox - cx
    fy = oy - cy
    a = dx * dx + dy * dy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius

    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None

    sqrt_d = np.sqrt(discriminant)
    t1 = (-b - sqrt_d) / (2 * a)
    t2 = (-b + sqrt_d) / (2 * a)
    candidates = [t for t in (t1, t2) if t >= 0]
    if not candidates:
        return None

    t = min(candidates)
    return ox + t * dx, oy + t * dy


def build_normalization_maps(
    pupil_center,
    pupil_radius,
    iris_center,
    iris_radius,
    radial_res=64,
    angular_res=512,
    mode="polar_iris",
):
    """Build cv2.remap maps once, so the image and mask use identical geometry."""
    theta = np.linspace(0, 2 * np.pi, angular_res, endpoint=False, dtype=np.float64)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    px, py = pupil_center
    ix, iy = iris_center

    iris_x = ix + iris_radius * cos_t
    iris_y = iy + iris_radius * sin_t

    if mode == "daugman":
        pupil_x = px + pupil_radius * cos_t
        pupil_y = py + pupil_radius * sin_t
    elif mode == "polar_iris":
        pupil_x = np.zeros_like(theta)
        pupil_y = np.zeros_like(theta)
        for i in range(angular_res):
            intersection = ray_circle_intersection(
                (ix, iy), (cos_t[i], sin_t[i]), (px, py), pupil_radius
            )
            if intersection is None:
                pupil_x[i] = px + pupil_radius * cos_t[i]
                pupil_y[i] = py + pupil_radius * sin_t[i]
            else:
                pupil_x[i], pupil_y[i] = intersection
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    r = np.linspace(0, 1, radial_res, dtype=np.float64).reshape(-1, 1)
    map_x = ((1 - r) * pupil_x + r * iris_x).astype(np.float32)
    map_y = ((1 - r) * pupil_y + r * iris_y).astype(np.float32)
    return map_x, map_y


def normalize_with_maps(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, is_mask=False):
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT if is_mask else cv2.BORDER_REPLICATE,
        borderValue=0,
    )


def order_triplet_left_to_right(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """Sort a 3-point eyelid triplet by image x coordinate."""
    if len(points) != 3:
        raise ValueError("Expected exactly three points")
    return [(float(x), float(y)) for x, y in sorted(points, key=lambda p: p[0])]


def fit_quadratic_y_of_x(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Fit y = ax^2 + bx + c through 3 ordered points."""
    pts = np.asarray(points, dtype=np.float64)
    if len(np.unique(np.round(pts[:, 0], 6))) < 3:
        raise ValueError("Eyelid points must have three distinct x coordinates")
    return np.polyfit(pts[:, 0], pts[:, 1], deg=2)


def sample_eyelid_curve(points: Sequence[Sequence[float]], samples=300):
    ordered = order_triplet_left_to_right(points)
    coeff = fit_quadratic_y_of_x(ordered)
    xs = np.linspace(ordered[0][0], ordered[-1][0], samples)
    ys = np.polyval(coeff, xs)
    return xs, ys, ordered


def build_eyelid_mask(
    image_shape,
    points: Sequence[Sequence[float]],
    x_range: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Build an eyelid visibility mask for subsequent iris normalization.

    Output semantics:
        255 -> valid / visible (not occluded by eyelids)
        0   -> occluded

    IMPORTANT:
    We do NOT fill only the narrow region between the parabolas.
    Instead, for each x-column we black out everything ABOVE the upper
    parabola and everything BELOW the lower parabola. This produces the
    expected normalized mask where the occluded zones become black bands.
    """
    h, w = image_shape[:2]
    upper = order_triplet_left_to_right(points[:3])
    lower = order_triplet_left_to_right(points[3:])

    upper_coef = fit_quadratic_y_of_x(upper)
    lower_coef = fit_quadratic_y_of_x(lower)

    if x_range is None:
        x_start, x_end = 0, w - 1
    else:
        x_start = max(0, int(np.floor(x_range[0])))
        x_end = min(w - 1, int(np.ceil(x_range[1])))

    if x_end <= x_start:
        raise ValueError("Invalid x-range for eyelid mask")

    xs = np.arange(x_start, x_end + 1, dtype=np.float64)
    upper_y = np.polyval(upper_coef, xs)
    lower_y = np.polyval(lower_coef, xs)

    # In image coordinates y grows downward.
    if np.median(lower_y - upper_y) <= 0:
        raise ValueError("Lower eyelid is above upper eyelid; check point order")

    upper_y = np.clip(np.rint(upper_y).astype(np.int32), 0, h - 1)
    lower_y = np.clip(np.rint(lower_y).astype(np.int32), 0, h - 1)

    # Start from full white and erase occluded zones.
    mask = np.full((h, w), 255, dtype=np.uint8)

    for x, y_top, y_bottom in zip(xs.astype(np.int32), upper_y, lower_y):
        if y_top > 0:
            mask[:y_top, x] = 0
        if y_bottom < h - 1:
            mask[y_bottom + 1 :, x] = 0

    return mask


def draw_eyelid_overlay(image: np.ndarray, points, mask=None) -> np.ndarray:
    """
    Draw only the eyelid curves and landmark points.

    We intentionally do not tint the whole mask region on the overlay,
    because that preview was misleading for the final normalized mask.
    """
    overlay = image.copy()

    for group, color in ((points[:3], (0, 255, 0)), (points[3:], (0, 0, 255))):
        if len(group) == 3:
            xs, ys, _ = sample_eyelid_curve(group)
            curve = np.stack([xs, ys], axis=1)
            curve = np.rint(curve).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [curve], False, color, 2, cv2.LINE_AA)

    for idx, (x, y) in enumerate(points):
        cv2.circle(overlay, (int(round(x)), int(round(y))), 4, POINT_COLORS[idx], -1, cv2.LINE_AA)
    return overlay


def write_pose_dataset_yaml(dataset_root: Path):
    dataset_root = dataset_root.resolve()
    yaml_path = dataset_root / "data.yaml"
    text = f"""# Six-keypoint eyelid pose dataset
path: {dataset_root.as_posix()}
train: images/train
val: images/val

kpt_shape: [6, 3]
# Horizontal flip swaps image-left and image-right landmarks.
flip_idx: [2, 1, 0, 5, 4, 3]

names:
    0: eye

kpt_names:
    0:
    - upper_left
    - upper_center
    - upper_right
    - lower_left
    - lower_center
    - lower_right
"""
    yaml_path.write_text(text, encoding="utf-8")


def make_pose_label(points, image_w, image_h, full_image_box=True, bbox_pad=0.10):
    """Create one Ultralytics YOLO pose row with six visible keypoints."""
    pts = np.asarray(points, dtype=np.float64)

    if full_image_box:
        cx, cy, bw, bh = 0.5, 0.5, 1.0, 1.0
    else:
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        pad_x = max(2.0, (x_max - x_min) * bbox_pad)
        pad_y = max(2.0, (y_max - y_min) * bbox_pad)
        x_min = max(0.0, x_min - pad_x)
        y_min = max(0.0, y_min - pad_y)
        x_max = min(float(image_w), x_max + pad_x)
        y_max = min(float(image_h), y_max + pad_y)
        cx = ((x_min + x_max) / 2.0) / image_w
        cy = ((y_min + y_max) / 2.0) / image_h
        bw = (x_max - x_min) / image_w
        bh = (y_max - y_min) / image_h

    values = [0, cx, cy, bw, bh]
    for x, y in pts:
        values.extend([np.clip(x / image_w, 0, 1), np.clip(y / image_h, 0, 1), 2])

    return " ".join(
        str(int(v)) if isinstance(v, (int, np.integer)) else f"{float(v):.8f}"
        for v in values
    )


def get_screen_size():
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        size = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return size
    except Exception:
        return 1920, 1080


class GeometryAnnotator:
    """Interactive geometry editor with a persistent main window and magnifier.

    The OpenCV windows are intentionally kept alive between images.  A small
    view-state dictionary stores zoom and the logical image point under the
    centre of the viewport, so the next image opens at the same view.
    """

    SNAP_POINT_INDICES = {0, 2, 3, 5}
    MAIN_WINDOW = "CASIA Annotator"
    MAG_WINDOW = "CASIA Magnifier"
    _windows_ready = False

    def __init__(
        self,
        image_path: Path,
        mode: str,
        screen_margin=80,
        grab_radius=12,
        circle_geometry=None,
        initial_points=None,
        snap_endpoints_to_iris=True,
        view_state=None,
        magnifier_size=320,
        magnifier_zoom=8.0,
    ):
        self.image_path = Path(image_path)
        self.mode = mode
        self.orig = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if self.orig is None:
            raise ValueError(f"Cannot open image: {self.image_path}")
        self.h, self.w = self.orig.shape[:2]

        screen_w, screen_h = get_screen_size()
        self.win_w = max(640, screen_w - screen_margin)
        self.win_h = max(480, screen_h - screen_margin)
        self.base_scale = min(self.win_w / self.w, self.win_h / self.h)

        self.view_state = view_state if view_state is not None else {}
        self.zoom = float(self.view_state.get("zoom", 1.0))
        self.zoom = float(np.clip(self.zoom, 0.5, 20.0))
        self.fullscreen = bool(self.view_state.get("fullscreen", False))

        # First image: fit and centre.  Following images: preserve the logical
        # point that was under the centre of the viewport on the previous one.
        if "center_x_ratio" in self.view_state and "center_y_ratio" in self.view_state:
            cx = float(self.view_state["center_x_ratio"]) * max(1, self.w - 1)
            cy = float(self.view_state["center_y_ratio"]) * max(1, self.h - 1)
            total_scale = self.base_scale * self.zoom
            self.pan_x = self.win_w * 0.5 - cx * total_scale
            self.pan_y = self.win_h * 0.5 - cy * total_scale
        else:
            self._fit_view()

        self.circle_geometry = circle_geometry
        self.snap_endpoints_to_iris = bool(snap_endpoints_to_iris and circle_geometry is not None)
        self.auto_points = list(initial_points or [])
        self.points: list[tuple[float, float]] = list(self.auto_points)
        self.dragging_idx = None
        self.grab_radius = grab_radius

        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.pan_start_ox = 0.0
        self.pan_start_oy = 0.0

        self.cursor_sx = self.win_w // 2
        self.cursor_sy = self.win_h // 2
        self.cursor_ox = self.w * 0.5
        self.cursor_oy = self.h * 0.5

        self.magnifier_enabled = bool(self.view_state.get("magnifier_enabled", True))
        self.magnifier_size = int(self.view_state.get("magnifier_size", magnifier_size))
        self.magnifier_zoom = float(self.view_state.get("magnifier_zoom", magnifier_zoom))
        self.magnifier_zoom = float(np.clip(self.magnifier_zoom, 2.0, 30.0))

    @property
    def point_labels(self):
        return CIRCLE_LABELS if self.mode == "circles" else EYELID_LABELS

    def _fit_view(self):
        self.zoom = 1.0
        total_scale = self.base_scale
        self.pan_x = (self.win_w - self.w * total_scale) * 0.5
        self.pan_y = (self.win_h - self.h * total_scale) * 0.5

    def _save_view_state(self):
        total_scale = max(1e-9, self.base_scale * self.zoom)
        center_ox = (self.win_w * 0.5 - self.pan_x) / total_scale
        center_oy = (self.win_h * 0.5 - self.pan_y) / total_scale
        self.view_state.update(
            {
                "zoom": float(self.zoom),
                "center_x_ratio": float(np.clip(center_ox / max(1, self.w - 1), 0.0, 1.0)),
                "center_y_ratio": float(np.clip(center_oy / max(1, self.h - 1), 0.0, 1.0)),
                "fullscreen": bool(self.fullscreen),
                "magnifier_enabled": bool(self.magnifier_enabled),
                "magnifier_size": int(self.magnifier_size),
                "magnifier_zoom": float(self.magnifier_zoom),
            }
        )

    def _screen_to_orig(self, sx, sy):
        total_scale = self.base_scale * self.zoom
        return (sx - self.pan_x) / total_scale, (sy - self.pan_y) / total_scale

    def _orig_to_screen(self, ox, oy):
        total_scale = self.base_scale * self.zoom
        return ox * total_scale + self.pan_x, oy * total_scale + self.pan_y

    def _update_cursor(self, x, y):
        self.cursor_sx, self.cursor_sy = int(x), int(y)
        ox, oy = self._screen_to_orig(x, y)
        self.cursor_ox = float(np.clip(ox, 0, self.w - 1))
        self.cursor_oy = float(np.clip(oy, 0, self.h - 1))

    def _constrain_point(self, point_index, ox, oy):
        ox = float(np.clip(ox, 0, self.w - 1))
        oy = float(np.clip(oy, 0, self.h - 1))
        if (
            self.mode == "eyelids"
            and self.snap_endpoints_to_iris
            and self.circle_geometry is not None
            and point_index in self.SNAP_POINT_INDICES
        ):
            ox, oy = project_point_to_circle(
                (ox, oy),
                self.circle_geometry["iris_center"],
                self.circle_geometry["iris_radius"],
            )
            ox = float(np.clip(ox, 0, self.w - 1))
            oy = float(np.clip(oy, 0, self.h - 1))
        return ox, oy

    def _find_nearest_point_idx(self, disp_x, disp_y):
        best_idx = None
        best_dist = self.grab_radius
        total_scale = self.base_scale * self.zoom
        for i, (px, py) in enumerate(self.points):
            sx = px * total_scale + self.pan_x
            sy = py * total_scale + self.pan_y
            d = np.hypot(sx - disp_x, sy - disp_y)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _zoom_at(self, sx, sy, factor):
        ox, oy = self._screen_to_orig(sx, sy)
        new_zoom = float(np.clip(self.zoom * factor, 0.5, 20.0))
        if abs(new_zoom - self.zoom) < 1e-9:
            return
        self.zoom = new_zoom
        total_scale = self.base_scale * self.zoom
        self.pan_x = sx - ox * total_scale
        self.pan_y = sy - oy * total_scale

    def _start_pan(self, x, y):
        self.panning = True
        self.pan_start_x = x
        self.pan_start_y = y
        self.pan_start_ox = self.pan_x
        self.pan_start_oy = self.pan_y

    def _on_mouse(self, event, x, y, flags, _param):
        self._update_cursor(x, y)

        if event == cv2.EVENT_LBUTTONDOWN and not self.panning:
            idx = self._find_nearest_point_idx(x, y)
            if idx is not None:
                self.dragging_idx = idx
            elif len(self.points) < 6:
                ox, oy = self._screen_to_orig(x, y)
                ox, oy = self._constrain_point(len(self.points), ox, oy)
                self.points.append((ox, oy))

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_idx is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
                ox, oy = self._screen_to_orig(x, y)
                ox, oy = self._constrain_point(self.dragging_idx, ox, oy)
                self.points[self.dragging_idx] = (ox, oy)
            elif self.panning:
                self.pan_x = self.pan_start_ox + (x - self.pan_start_x)
                self.pan_y = self.pan_start_oy + (y - self.pan_start_y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_idx = None

        elif event in (cv2.EVENT_MBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self._start_pan(x, y)

        elif event in (cv2.EVENT_MBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.panning = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            self._zoom_at(x, y, 1.15 if flags > 0 else 1 / 1.15)

    def _draw_predicted_circles(self, disp, total_scale):
        if self.circle_geometry is None:
            return
        for center_key, radius_key, color, label in (
            ("pupil_center", "pupil_radius", (0, 255, 255), "YOLO pupil"),
            ("iris_center", "iris_radius", (255, 255, 0), "YOLO iris"),
        ):
            cx, cy = self.circle_geometry[center_key]
            radius = self.circle_geometry[radius_key]
            sx = int(cx * total_scale + self.pan_x)
            sy = int(cy * total_scale + self.pan_y)
            sr = max(1, int(radius * total_scale))
            cv2.circle(disp, (sx, sy), sr, color, 2, cv2.LINE_AA)
            cv2.putText(
                disp, label, (sx + 6, sy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )

    def _draw_circle_preview(self, disp, total_scale):
        if len(self.points) >= 3:
            res = circle_from_3_points(*self.points[:3])
            if res:
                (cx, cy), radius = res
                cv2.circle(
                    disp,
                    (int(cx * total_scale + self.pan_x), int(cy * total_scale + self.pan_y)),
                    max(1, int(radius * total_scale)),
                    (0, 255, 0), 2, cv2.LINE_AA,
                )
        if len(self.points) >= 6:
            res = circle_from_3_points(*self.points[3:6])
            if res:
                (cx, cy), radius = res
                cv2.circle(
                    disp,
                    (int(cx * total_scale + self.pan_x), int(cy * total_scale + self.pan_y)),
                    max(1, int(radius * total_scale)),
                    (0, 0, 255), 2, cv2.LINE_AA,
                )

    def _draw_eyelid_preview(self, disp, total_scale):
        for group, color in ((self.points[:3], (0, 255, 0)), (self.points[3:6], (0, 0, 255))):
            if len(group) == 3:
                try:
                    xs, ys, _ = sample_eyelid_curve(group)
                except ValueError:
                    continue
                curve = np.column_stack(
                    [xs * total_scale + self.pan_x, ys * total_scale + self.pan_y]
                )
                curve = np.rint(curve).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(disp, [curve], False, color, 2, cv2.LINE_AA)

    def _render(self):
        total_scale = self.base_scale * self.zoom
        transform = np.float32([[total_scale, 0, self.pan_x], [0, total_scale, self.pan_y]])
        disp = cv2.warpAffine(
            self.orig,
            transform,
            (self.win_w, self.win_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(35, 35, 35),
        )

        if self.mode == "circles":
            self._draw_circle_preview(disp, total_scale)
        else:
            self._draw_predicted_circles(disp, total_scale)
            self._draw_eyelid_preview(disp, total_scale)

        for i, (px, py) in enumerate(self.points):
            sx = int(px * total_scale + self.pan_x)
            sy = int(py * total_scale + self.pan_y)
            radius = 8 if i == self.dragging_idx else 5
            cv2.circle(disp, (sx, sy), radius, POINT_COLORS[i], -1, cv2.LINE_AA)
            cv2.circle(disp, (sx, sy), radius + 2, (255, 255, 255), 1, cv2.LINE_AA)

        overlay_h = 112 if self.mode == "eyelids" and self.circle_geometry is not None else 88
        cv2.rectangle(disp, (0, 0), (disp.shape[1], overlay_h), (0, 0, 0), -1)
        if len(self.points) < 6:
            msg = self.point_labels[len(self.points)]
        else:
            msg = "ENTER=save  U=undo  R=points reset  N=skip  Q/Esc=quit"
        cv2.putText(disp, msg, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        controls = "wheel=zoom@cursor  RMB/MMB-drag=pan  arrows=pan  0=fit  F=fullscreen  M=magnifier"
        cv2.putText(
            disp, controls, (8, 50), cv2.FONT_HERSHEY_SIMPLEX,
            0.44, (210, 210, 210), 1, cv2.LINE_AA,
        )
        cv2.putText(
            disp,
            f"{self.image_path.name}   zoom={self.zoom:.2f}x   cursor=({self.cursor_ox:.1f},{self.cursor_oy:.1f})",
            (8, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 255, 150), 1, cv2.LINE_AA,
        )
        if self.mode == "eyelids" and self.circle_geometry is not None:
            snap_text = (
                "S=snap endpoints ON/OFF  A=restore auto points   "
                f"snap={'ON' if self.snap_endpoints_to_iris else 'OFF'}"
            )
            cv2.putText(
                disp, snap_text, (8, 98), cv2.FONT_HERSHEY_SIMPLEX,
                0.44, (180, 255, 180), 1, cv2.LINE_AA,
            )
        return disp

    def _render_magnifier(self):
        """Render an exact high-resolution patch around the mouse cursor."""
        size = max(160, int(self.magnifier_size))
        mag = max(2.0, float(self.magnifier_zoom))
        # Number of source pixels visible in the magnifier.
        src_side = max(8, int(round(size / mag)))
        half = src_side // 2
        cx = int(round(self.cursor_ox))
        cy = int(round(self.cursor_oy))

        x0, x1 = cx - half, cx - half + src_side
        y0, y1 = cy - half, cy - half + src_side
        patch = np.full((src_side, src_side, 3), 35, dtype=np.uint8)

        sx0, sx1 = max(0, x0), min(self.w, x1)
        sy0, sy1 = max(0, y0), min(self.h, y1)
        if sx1 > sx0 and sy1 > sy0:
            dx0, dy0 = sx0 - x0, sy0 - y0
            patch[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = self.orig[sy0:sy1, sx0:sx1]

        view = cv2.resize(patch, (size, size), interpolation=cv2.INTER_NEAREST)
        c = size // 2
        cv2.line(view, (c, 0), (c, size - 1), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(view, (0, c), (size - 1, c), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(view, (c, c), 5, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw nearby annotation points in magnifier coordinates.
        scale = size / float(src_side)
        for i, (px, py) in enumerate(self.points):
            mx = int(round((px - x0) * scale))
            my = int(round((py - y0) * scale))
            if 0 <= mx < size and 0 <= my < size:
                cv2.circle(view, (mx, my), 5, POINT_COLORS[i], -1, cv2.LINE_AA)
                cv2.circle(view, (mx, my), 7, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.rectangle(view, (0, 0), (size - 1, 28), (0, 0, 0), -1)
        cv2.putText(
            view,
            f"{mag:.1f}x  x={self.cursor_ox:.2f} y={self.cursor_oy:.2f}   [ / ] zoom",
            (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA,
        )
        return view

    def _ensure_windows(self):
        if not GeometryAnnotator._windows_ready:
            cv2.namedWindow(self.MAIN_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.MAIN_WINDOW, self.win_w, self.win_h)
            cv2.namedWindow(self.MAG_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.MAG_WINDOW, self.magnifier_size, self.magnifier_size)
            GeometryAnnotator._windows_ready = True
        cv2.setMouseCallback(self.MAIN_WINDOW, self._on_mouse)
        try:
            cv2.setWindowTitle(self.MAIN_WINDOW, f"CASIA Annotator - {self.image_path.name}")
        except Exception:
            pass
        try:
            cv2.setWindowProperty(
                self.MAIN_WINDOW,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL,
            )
        except Exception:
            pass

    def _finish(self, status, points):
        self._save_view_state()
        return status, points

    def run(self):
        self._ensure_windows()
        while True:
            cv2.imshow(self.MAIN_WINDOW, self._render())
            if self.magnifier_enabled:
                cv2.imshow(self.MAG_WINDOW, self._render_magnifier())
            else:
                # Keep the window alive but unobtrusive, avoiding recreation per frame/image.
                blank = np.zeros((120, 240, 3), dtype=np.uint8)
                cv2.putText(blank, "Magnifier OFF (M)", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.imshow(self.MAG_WINDOW, blank)

            key = cv2.waitKeyEx(20)
            low = key & 0xFF

            if low == ord("u"):
                if self.points and self.dragging_idx is None:
                    self.points.pop()
            elif low == ord("r"):
                self.points = []
                self.dragging_idx = None
            elif low == ord("a") and self.mode == "eyelids" and self.auto_points:
                self.points = list(self.auto_points)
                self.dragging_idx = None
            elif low == ord("s") and self.mode == "eyelids" and self.circle_geometry is not None:
                self.snap_endpoints_to_iris = not self.snap_endpoints_to_iris
                if self.snap_endpoints_to_iris:
                    for idx in self.SNAP_POINT_INDICES:
                        if idx < len(self.points):
                            self.points[idx] = self._constrain_point(idx, *self.points[idx])
            elif low in (13, 10):
                if len(self.points) == 6:
                    return self._finish("done", list(self.points))
            elif low == ord("n"):
                return self._finish("skip", None)
            elif low in (ord("q"), 27):
                return self._finish("quit", None)
            elif low in (ord("+"), ord("=")):
                self._zoom_at(self.win_w // 2, self.win_h // 2, 1.2)
            elif low == ord("-"):
                self._zoom_at(self.win_w // 2, self.win_h // 2, 1 / 1.2)
            elif low == ord("0"):
                self._fit_view()
            elif low == ord("f"):
                self.fullscreen = not self.fullscreen
                try:
                    cv2.setWindowProperty(
                        self.MAIN_WINDOW,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL,
                    )
                except Exception:
                    pass
            elif low == ord("m"):
                self.magnifier_enabled = not self.magnifier_enabled
            elif low == ord("["):
                self.magnifier_zoom = max(2.0, self.magnifier_zoom / 1.25)
            elif low == ord("]"):
                self.magnifier_zoom = min(30.0, self.magnifier_zoom * 1.25)
            # OpenCV waitKeyEx arrow codes (Windows/Linux); the low-byte codes
            # cover several backends as well.
            elif key in (2424832, 65361, 81):       # left
                self.pan_x += 40
            elif key in (2555904, 65363, 83):       # right
                self.pan_x -= 40
            elif key in (2490368, 65362, 82):       # up
                self.pan_y += 40
            elif key in (2621440, 65364, 84):       # down
                self.pan_y -= 40


def load_done_filenames(log_file: Path, mode: str):
    """Read both legacy circle CSV and the newer multimode eyelid CSV."""
    done = set()
    if not log_file.exists():
        return done
    with log_file.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "done":
                continue
            row_mode = (row.get("mode") or "").strip()
            if row_mode and row_mode != mode:
                continue
            filename = (row.get("filename") or "").strip()
            if filename:
                done.add(filename)
    return done


CIRCLE_LOG_FIELDS = [
    "filename", "status",
    "pupil_cx", "pupil_cy", "pupil_r",
    "iris_cx", "iris_cy", "iris_r",
    "output", "timestamp",
]

EYELID_LOG_FIELDS = [
    "filename", "mode", "status", "points_json",
    "output", "label", "timestamp",
]


def append_log_row(log_file: Path, row: dict):
    """Append a row while preserving the schema of an existing CSV."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_file.exists() and log_file.stat().st_size > 0

    if file_exists:
        with log_file.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            fieldnames = next(reader, None)
        if not fieldnames:
            fieldnames = list(row.keys())
        unexpected = set(row) - set(fieldnames)
        if unexpected:
            raise ValueError(
                f"Log schema mismatch for {log_file}. Unexpected fields: {sorted(unexpected)}. "
                "Use a separate --log-file for circles and eyelids."
            )
    else:
        fieldnames = list(row.keys())

    with log_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def validate_circle_points(points):
    pupil = circle_from_3_points(*points[:3])
    iris = circle_from_3_points(*points[3:6])
    if not pupil or not iris:
        raise ValueError("Three circle points are collinear")
    pupil_c, pupil_r = pupil
    iris_c, iris_r = iris
    if pupil_r < 1 or iris_r < 1 or iris_r <= pupil_r:
        raise ValueError("Iris radius must be larger than pupil radius")
    return pupil_c, pupil_r, iris_c, iris_r


def process_circle_annotation(path: Path, points, cfg: AppConfig):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("Cannot read image")

    pupil_c, pupil_r, iris_c, iris_r = validate_circle_points(points)
    map_x, map_y = build_normalization_maps(
        pupil_c,
        pupil_r,
        iris_c,
        iris_r,
        cfg.radial_res,
        cfg.angular_res,
        cfg.normalization_mode,
    )
    normalized = normalize_with_maps(gray, map_x, map_y, is_mask=False)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{path.stem}_norm.png"
    output_path = cfg.output_dir / output_name
    if not cv2.imwrite(str(output_path), normalized):
        raise IOError(f"Cannot save {output_path}")
    return output_name, ""


def process_eyelid_annotation(path: Path, points, cfg: AppConfig, circle_geometry=None):
    """
    Saves:
            1) the eyelid mask in the original coordinates;
            2) the original image with YOLO circles and eyelid curves;
            3) the normalized iris;
            4) the normalized binary eyelid mask;
            5) the normalized iris with the applied Mask;
            6) the YOLO Pose 6‑point annotation (if dataset_root is specified).

    For normalization, the circles predicted by YOLO Detect are used.
        The image and the mask pass through the same map_x/map_y.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or gray is None:
        raise ValueError("Cannot read image")
    h, w = image.shape[:2]

    if circle_geometry is None:
        raise ValueError(
            "Eyelids mode requires pupil/iris circles from YOLO Detect. "
            "Pass a trained model with --circle-model."
        )

    # Making the semantic order: left / center / right
    ordered_points = (
        order_triplet_left_to_right(points[:3])
        + order_triplet_left_to_right(points[3:6])
    )

    # We form the mask not as a “narrow rectangle between the curves,”
    # but as an occlusion mask: everything above the upper parabola and everything below the lower
    # parabola becomes black. For stability, we limit the calculation
    # to the X range of the iris
    iris_cx, iris_cy = circle_geometry["iris_center"]
    iris_r = circle_geometry["iris_radius"]
    source_mask = build_eyelid_mask(
        image.shape,
        ordered_points,
        x_range=(iris_cx - iris_r, iris_cx + iris_r),
    )

    # Overlay of the original image: only circles, curves, and points,
    # without the green fill of the mask
    overlay = draw_eyelid_overlay(image, ordered_points, None)
    for center_key, radius_key, color, label in (
        ("pupil_center", "pupil_radius", (0, 255, 255), "YOLO pupil"),
        ("iris_center", "iris_radius", (255, 255, 0), "YOLO iris"),
    ):
        center = tuple(int(round(v)) for v in circle_geometry[center_key])
        radius = max(1, int(round(circle_geometry[radius_key])))
        cv2.circle(overlay, center, radius, color, 2, cv2.LINE_AA)
        cv2.putText(
            overlay,
            label,
            (center[0] + 5, center[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    # The same normalization maps are used for both the image and the mask
    map_x, map_y = build_normalization_maps(
        circle_geometry["pupil_center"],
        circle_geometry["pupil_radius"],
        circle_geometry["iris_center"],
        circle_geometry["iris_radius"],
        radial_res=cfg.radial_res,
        angular_res=cfg.angular_res,
        mode=cfg.normalization_mode,
    )

    normalized_iris = normalize_with_maps(
        gray, map_x, map_y, is_mask=False
    )

    normalized_mask = normalize_with_maps(
        source_mask, map_x, map_y, is_mask=True
    )
    # After INTER_NEAREST, the mask is already binary, but we fix it to 0/255
    normalized_mask = np.where(normalized_mask >= 128, 255, 0).astype(np.uint8)

    normalized_masked = cv2.bitwise_and(
        normalized_iris,
        normalized_iris,
        mask=normalized_mask,
    )

    # The Folders of results
    source_masks_dir = cfg.output_dir / "source_masks"
    overlays_dir = cfg.output_dir / "overlays"
    normalized_dir = cfg.output_dir / "normalized"
    normalized_masks_dir = cfg.output_dir / "normalized_masks"
    normalized_masked_dir = cfg.output_dir / "normalized_masked"
    previews_dir = cfg.output_dir / "normalization_previews"

    for directory in (
        source_masks_dir,
        overlays_dir,
        normalized_dir,
        normalized_masks_dir,
        normalized_masked_dir,
        previews_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    source_mask_name = f"{path.stem}_eyelids_source_mask.png"
    overlay_name = f"{path.stem}_eyelids_overlay.jpg"
    normalized_name = f"{path.stem}_norm.png"
    normalized_mask_name = f"{path.stem}_norm_eyelids_mask.png"
    normalized_masked_name = f"{path.stem}_norm_masked.png"
    preview_name = f"{path.stem}_normalization_preview.png"

    source_mask_path = source_masks_dir / source_mask_name
    overlay_path = overlays_dir / overlay_name
    normalized_path = normalized_dir / normalized_name
    normalized_mask_path = normalized_masks_dir / normalized_mask_name
    normalized_masked_path = normalized_masked_dir / normalized_masked_name
    preview_path = previews_dir / preview_name

    outputs = (
        (source_mask_path, source_mask, "source eyelid mask"),
        (overlay_path, overlay, "overlay"),
        (normalized_path, normalized_iris, "normalized iris"),
        (normalized_mask_path, normalized_mask, "normalized eyelid mask"),
        (normalized_masked_path, normalized_masked, "normalized masked iris"),
    )
    for output_path, output_image, description in outputs:
        if not cv2.imwrite(str(output_path), output_image):
            raise IOError(f"Cannot save {description}: {output_path}")

    # Three images side by side for quick visual check:
    # normalized iris | normalized mask | normalized masked iris.
    preview = cv2.hconcat(
        [normalized_iris, normalized_mask, normalized_masked]
    )
    if not cv2.imwrite(str(preview_path), preview):
        raise IOError(f"Cannot save normalization preview: {preview_path}")

    label_rel = ""
    if cfg.dataset_root is not None:
        image_dir = cfg.dataset_root / "images" / cfg.split
        label_dir = cfg.dataset_root / "labels" / cfg.split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        target_image = image_dir / path.name
        if cfg.copy_images:
            shutil.copy2(path, target_image)

        label_path = label_dir / f"{path.stem}.txt"
        label_path.write_text(
            make_pose_label(
                ordered_points,
                w,
                h,
                full_image_box=cfg.full_image_box,
            )
            + "\n",
            encoding="utf-8",
        )
        write_pose_dataset_yaml(cfg.dataset_root)
        label_rel = str(label_path)

    metadata_dir = cfg.output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "image": path.name,
        "eyelid_points": [[float(x), float(y)] for x, y in ordered_points],
        "circle_prediction": circle_geometry,
        "normalization_mode": cfg.normalization_mode,
        "radial_res": cfg.radial_res,
        "angular_res": cfg.angular_res,
        "outputs": {
            "source_mask": str(source_mask_path),
            "overlay": str(overlay_path),
            "normalized_iris": str(normalized_path),
            "normalized_mask": str(normalized_mask_path),
            "normalized_masked": str(normalized_masked_path),
            "preview": str(preview_path),
        },
        "mask_definition": {
            "255": "visible / valid area after eyelid occlusion masking",
            "0": "area above upper eyelid or below lower eyelid",
        },
        "endpoint_definition": (
            "upper/lower left/right points are constrained "
            "to the predicted iris circle when snap is enabled"
        ),
    }
    (metadata_dir / f"{path.stem}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # The log records the already corrected semantic order of the points
    points[:] = ordered_points

    print(f"  source mask: {source_mask_path}")
    print(f"  normalized iris: {normalized_path}")
    print(f"  normalized mask: {normalized_mask_path}")
    print(f"  normalized masked iris: {normalized_masked_path}")
    print(f"  preview: {preview_path}")

    # We consider the normalized mask to be the main result of the eyelids mode
    return str(normalized_mask_path), label_rel


def process_one_image(path: Path, cfg: AppConfig, circle_model=None, view_state=None):
    circle_geometry = None
    initial_points = None
    if cfg.mode == "eyelids" and circle_model is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Cannot read image")
        try:
            circle_geometry = predict_circles_from_yolo(
                circle_model,
                image,
                conf=cfg.circle_conf,
                iou=cfg.circle_iou,
                imgsz=cfg.circle_imgsz,
                pupil_class=cfg.pupil_class,
                iris_class=cfg.iris_class,
                device=cfg.prediction_device,
            )
            if cfg.auto_eyelid_points:
                initial_points = suggest_eyelid_points(circle_geometry, image.shape)
            print(
                "  circles: "
                f"pupil conf={circle_geometry['pupil_conf']:.3f}, "
                f"iris conf={circle_geometry['iris_conf']:.3f}"
            )
        except Exception as exc:
            if cfg.require_circle_prediction:
                raise RuntimeError(f"Circle prediction failed: {exc}") from exc
            print(f"  warning: circle prediction failed, manual eyelid annotation only: {exc}")

    # If geometry validation fails, keep the six points so the user can adjust
    # them instead of clicking all six again.
    retry_points = initial_points
    while True:
        status, points = GeometryAnnotator(
            path,
            cfg.mode,
            cfg.screen_margin,
            cfg.grab_radius,
            circle_geometry=circle_geometry,
            initial_points=retry_points,
            snap_endpoints_to_iris=cfg.snap_endpoints_to_iris,
            view_state=view_state,
        ).run()
        if status in ("quit", "skip"):
            return status, None, None

        try:
            if cfg.mode == "circles":
                output, label = process_circle_annotation(path, points, cfg)
            else:
                output, label = process_eyelid_annotation(
                    path, points, cfg, circle_geometry=circle_geometry
                )
            return "done", points, (output, label)
        except ValueError as exc:
            print(f"  Invalid annotation: {exc}. Adjust the points and press ENTER again.")
            retry_points = list(points)


def parse_args():
    parser = argparse.ArgumentParser(description="Annotate iris circles or eyelid keypoints")
    parser.add_argument("--mode", choices=("circles", "eyelids"), required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Default: annotations_log.csv for circles, annotations_log_2.csv for eyelids",
    )
    parser.add_argument("--normalization-mode", choices=("daugman", "polar_iris"), default="polar_iris")
    parser.add_argument("--radial-res", type=int, default=64)
    parser.add_argument("--angular-res", type=int, default=512)
    parser.add_argument("--screen-margin", type=int, default=80)
    parser.add_argument("--grab-radius", type=int, default=12)
    parser.add_argument(
        "--circle-model",
        type=Path,
        default="./yolov8l.pt",
        help=(
            "Required in eyelids mode: trained YOLO Detect weights "
            "with pupil and iris classes, for example runs/detect/.../weights/best.pt"
        ),
    )
    parser.add_argument(
        "--gpu",
        default="cpu",
        help="Physical GPU id for CUDA_VISIBLE_DEVICES, for example 6; or 'cpu'",
    )
    parser.add_argument("--circle-conf", type=float, default=0.05)
    parser.add_argument("--circle-iou", type=float, default=0.4)
    parser.add_argument("--circle-imgsz", type=int, default=640)
    parser.add_argument("--pupil-class", default="pupil")
    parser.add_argument("--iris-class", default="iris")
    parser.add_argument(
        "--no-auto-eyelid-points",
        action="store_true",
        help="Show predicted circles but do not prefill the six eyelid points",
    )
    parser.add_argument(
        "--no-snap-endpoints",
        action="store_true",
        help="Do not constrain eyelid endpoint landmarks to the predicted iris circle",
    )
    parser.add_argument(
        "--require-circle-prediction",
        action="store_true",
        help="Stop/mark error when the circle model cannot predict both circles",
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="For eyelids mode: root of YOLO pose dataset to create/update",
    )
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--no-copy-images", action="store_true")
    parser.add_argument(
        "--tight-box",
        action="store_true",
        help="Use a keypoint-derived bbox instead of full-image bbox",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search images inside input-dir",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.log_file is None:
        args.log_file = Path("./annotations_log.csv" if args.mode == "circles" else "./annotations_log_2.csv")

    circle_model = None
    prediction_device: int | str = "cpu" if str(args.gpu).lower() == "cpu" else 0

    if args.mode == "eyelids":
        if args.circle_model is None:
            raise ValueError(
                "For --mode eyelids pass trained circle weights with --circle-model"
            )
        if not args.circle_model.exists():
            raise FileNotFoundError(
                f"YOLO circle model not found: {args.circle_model}"
            )

        # CUDA_VISIBLE_DEVICES задаётся до импорта ultralytics/torch.
        if prediction_device != "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

        from ultralytics import YOLO

        print(f"Loading YOLO Detect circle model: {args.circle_model}")
        circle_model = YOLO(str(args.circle_model))

    cfg = AppConfig(
        mode=args.mode,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        log_file=args.log_file,
        normalization_mode=args.normalization_mode,
        radial_res=args.radial_res,
        angular_res=args.angular_res,
        screen_margin=args.screen_margin,
        grab_radius=args.grab_radius,
        dataset_root=args.dataset_root,
        split=args.split,
        copy_images=not args.no_copy_images,
        full_image_box=not args.tight_box,
        circle_model_path=args.circle_model,
        circle_conf=args.circle_conf,
        circle_iou=args.circle_iou,
        circle_imgsz=args.circle_imgsz,
        pupil_class=args.pupil_class,
        iris_class=args.iris_class,
        prediction_device=prediction_device,
        auto_eyelid_points=not args.no_auto_eyelid_points,
        snap_endpoints_to_iris=not args.no_snap_endpoints,
        require_circle_prediction=(
            True if args.mode == "eyelids" else args.require_circle_prediction
        ),
    )

    if not cfg.input_dir.exists():
        raise FileNotFoundError(cfg.input_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if args.recursive:
        files = sorted(p for p in cfg.input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXT)
    else:
        files = sorted(p for p in cfg.input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT)
    if not files:
        print(f"No supported images in {cfg.input_dir}")
        sys.exit(1)

    done = load_done_filenames(cfg.log_file, cfg.mode)
    todo = [p for p in files if p.name not in done]
    print(f"Mode: {cfg.mode}. Total: {len(files)}, already done: {len(done)}, left: {len(todo)}")
    if cfg.mode == "eyelids":
        if circle_model is None:
            print("Circle guide: disabled (pass --circle-model to enable it)")
        else:
            print(
                f"Circle guide: {cfg.circle_model_path}; "
                f"auto-points={cfg.auto_eyelid_points}; "
                f"snap-endpoints={cfg.snap_endpoints_to_iris}"
            )

    # Shared by all images: this is what keeps zoom/pan/fullscreen/magnifier
    # state stable while advancing through the dataset.
    view_state = {}

    for idx, path in enumerate(todo, 1):
        print(f"[{idx}/{len(todo)}] {path.name}")
        try:
            status, points, result = process_one_image(
                path, cfg, circle_model=circle_model, view_state=view_state
            )
        except Exception as exc:
            print(f"  Error: {exc}")
            if cfg.mode == "circles":
                error_row = {
                    "filename": path.name,
                    "status": "error",
                    "pupil_cx": "", "pupil_cy": "", "pupil_r": "",
                    "iris_cx": "", "iris_cy": "", "iris_r": "",
                    "output": "",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            else:
                error_row = {
                    "filename": path.name,
                    "mode": cfg.mode,
                    "status": "error",
                    "points_json": "",
                    "output": "",
                    "label": "",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
            append_log_row(cfg.log_file, error_row)
            continue

        if status == "quit":
            print("Session stopped. Progress is preserved.")
            break
        if status == "skip":
            print("  skipped")
            continue

        output, label = result
        if cfg.mode == "circles":
            pupil_c, pupil_r, iris_c, iris_r = validate_circle_points(points)
            row = {
                "filename": path.name,
                "status": "done",
                "pupil_cx": f"{pupil_c[0]:.6f}",
                "pupil_cy": f"{pupil_c[1]:.6f}",
                "pupil_r": f"{pupil_r:.6f}",
                "iris_cx": f"{iris_c[0]:.6f}",
                "iris_cy": f"{iris_c[1]:.6f}",
                "iris_r": f"{iris_r:.6f}",
                "output": output,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        else:
            row = {
                "filename": path.name,
                "mode": cfg.mode,
                "status": "done",
                "points_json": json.dumps(points, ensure_ascii=False),
                "output": output,
                "label": label,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        append_log_row(cfg.log_file, row)
        print(f"  saved: {output}")
        if cfg.mode == "circles":
            print(
                f"  pupil=({pupil_c[0]:.2f}, {pupil_c[1]:.2f}, r={pupil_r:.2f})  "
                f"iris=({iris_c[0]:.2f}, {iris_c[1]:.2f}, r={iris_r:.2f})"
            )
        if label:
            print(f"  YOLO label: {label}")

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
