"""Create a minimal, animated single-line portrait from a photo.

Usage:
    python trace_portrait.py portrait.jpg --output-prefix output\\portrait
    python trace_portrait.py portrait.jpg --grabcut
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from skimage import feature, measure


# Adjust these four settings to change the line-art style.
DETAIL_LEVEL = 0.48  # 0 = very minimal, 1 = more facial contours
SMOOTHING_AMOUNT = 0.42  # 0 = preserve edges, 1 = smoother curves
LINE_WIDTH = 2.0  # Output stroke width in pixels
MIN_STROKE_LENGTH = 0.03  # Ignore strokes shorter than this fraction of height

REMBG_MODEL = "u2net"
MAX_FACE_STROKES = 20
ANTIALIAS_SCALE = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Make transparent white/black line art and a single-path animated SVG."
        )
    )
    parser.add_argument("image", type=Path, help="portrait image to trace")
    parser.add_argument(
        "--output-prefix",
        "-o",
        type=Path,
        default=Path("portrait-trace"),
        help="output path prefix (default: portrait-trace)",
    )
    parser.add_argument(
        "--grabcut",
        action="store_true",
        help="use OpenCV GrabCut even if rembg is installed",
    )
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"image not found: {args.image}")
    return args


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image: {path}")
    return image


def rembg_alpha(image: np.ndarray) -> np.ndarray | None:
    if importlib.util.find_spec("rembg") is None:
        return None

    try:
        from rembg import new_session, remove

        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("could not encode the source image for rembg")
        session = new_session(REMBG_MODEL)
        result = remove(encoded.tobytes(), session=session)
        if isinstance(result, np.ndarray):
            rgba = result
        else:
            rgba = cv2.imdecode(
                np.frombuffer(result, dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            raise RuntimeError("rembg did not return an RGBA cutout")
        if rgba.shape[:2] != image.shape[:2]:
            rgba = cv2.resize(
                rgba,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return rgba[:, :, 3]
    except Exception as error:
        print(
            f"Warning: rembg failed ({error}); trying OpenCV GrabCut instead.",
            file=sys.stderr,
        )
        return None


def grabcut_alpha(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    margin_x = max(1, int(width * 0.015))
    margin_y = max(1, int(height * 0.015))
    rect = (
        margin_x,
        margin_y,
        max(1, width - 2 * margin_x),
        max(1, height - 2 * margin_y),
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        image,
        mask,
        rect,
        background_model,
        foreground_model,
        8,
        cv2.GC_INIT_WITH_RECT,
    )
    foreground = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
    return foreground.astype(np.uint8) * 255


def clean_person_mask(alpha: np.ndarray) -> np.ndarray:
    mask = (alpha >= 112).astype(np.uint8)
    close_size = max(3, int(round(min(mask.shape) * 0.008)) | 1)
    open_size = max(3, int(round(min(mask.shape) * 0.004)) | 1)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_size, close_size),
        ),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_size, open_size),
        ),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        raise ValueError("background removal found no person; try a clearer photo")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest_area = stats[largest, cv2.CC_STAT_AREA]
    if largest_area < mask.size * 0.01:
        raise ValueError("the person cutout is too small; try a closer portrait")
    return (labels == largest).astype(np.uint8)


def find_face_box(image: np.ndarray, mask: np.ndarray) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    person_ys, person_xs = np.nonzero(mask)
    if person_xs.size == 0:
        raise ValueError("could not locate the person in the image")
    person_width = int(person_xs.max() - person_xs.min() + 1)
    person_height = int(person_ys.max() - person_ys.min() + 1)
    person_area = int(mask.sum())
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    detections: list[tuple[int, tuple[int, int, int, int]]] = []

    if not cascade.empty():
        minimum = max(24, int(min(gray.shape) * 0.07))
        for rotation in range(4):
            rotated = np.rot90(gray, rotation).copy()
            faces = cascade.detectMultiScale(
                rotated,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(minimum, minimum),
            )
            for x, y, width, height in faces:
                marker = np.zeros(rotated.shape, dtype=np.uint8)
                cv2.rectangle(
                    marker,
                    (int(x), int(y)),
                    (int(x + width - 1), int(y + height - 1)),
                    255,
                    -1,
                )
                original_marker = np.rot90(marker, -rotation)
                ys, xs = np.nonzero(original_marker)
                if xs.size:
                    box = (
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max() - xs.min() + 1),
                        int(ys.max() - ys.min() + 1),
                    )
                    box_width, box_height = box[2], box[3]
                    if (
                        box_width <= person_width * 0.65
                        and box_height <= person_height * 0.62
                        and box_width * box_height <= person_area * 0.5
                    ):
                        detections.append((width * height, box))

    if detections:
        _, (x, y, width, height) = max(detections, key=lambda item: item[0])
        extra_x = int(width * 0.1)
        extra_top = int(height * 0.06)
        extra_bottom = int(height * 0.14)
        left = max(0, x - extra_x)
        top = max(0, y - extra_top)
        right = min(image.shape[1], x + width + extra_x)
        bottom = min(image.shape[0], y + height + extra_bottom)
        return left, top, right - left, bottom - top

    person_left, person_right = int(person_xs.min()), int(person_xs.max()) + 1
    person_top = int(person_ys.min())
    left = person_left + int(person_width * 0.22)
    top = person_top + int(person_height * 0.015)
    width = max(1, int(person_width * 0.56))
    height = max(1, int(person_height * 0.35))
    return (
        max(0, left),
        max(0, top),
        min(width, image.shape[1] - left),
        min(height, image.shape[0] - top),
    )


def polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def smooth_stroke(points: np.ndarray, closed: bool) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) > 1:
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 0.25]
        points = points[keep]
    if closed and len(points) > 2 and np.linalg.norm(points[0] - points[-1]) < 2:
        points = points[:-1]
    if len(points) < 4:
        return points

    length = polyline_length(points)
    spline_degree = min(3, len(points) - 1)
    smoothing = (4.0 * SMOOTHING_AMOUNT) ** 2 * len(points)
    spline, _ = splprep(
        [points[:, 0], points[:, 1]],
        s=smoothing,
        k=spline_degree,
        per=int(closed),
    )
    sample_count = max(32, min(5000, int(length / 1.4)))
    parameter = np.linspace(0, 1, sample_count, endpoint=not closed)
    x, y = splev(parameter, spline)
    result = np.column_stack((x, y))
    if closed:
        result = np.vstack((result, result[0]))
    return result


def extract_strokes(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[list[np.ndarray], tuple[int, int, int, int]]:
    height, width = mask.shape
    min_length = MIN_STROKE_LENGTH * height
    outer_contours, _ = cv2.findContours(
        mask * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not outer_contours:
        raise ValueError("could not trace the person silhouette")
    outline_contour = max(outer_contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(outline_contour, True)
    outline_contour = cv2.approxPolyDP(
        outline_contour,
        max(0.6, perimeter * 0.0008),
        True,
    )
    outline = smooth_stroke(outline_contour[:, 0, :], closed=True)
    center_x = width * 0.5
    outline_start = int(
        np.argmin(outline[:-1, 1] + np.abs(outline[:-1, 0] - center_x) * 0.16)
    )
    outline = np.vstack((outline[outline_start:-1], outline[: outline_start + 1]))
    strokes = [outline]

    face_x, face_y, face_width, face_height = find_face_box(image, mask)
    face_region = np.zeros_like(mask, dtype=bool)
    face_region[
        face_y : face_y + face_height,
        face_x : face_x + face_width,
    ] = True
    erode_size = max(3, int(round(min(height, width) * 0.006)) | 1)
    inner_mask = cv2.erode(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (erode_size, erode_size),
        ),
    ).astype(bool)
    allowed = face_region & inner_mask
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    sigma = 2.2 - 1.1 * float(np.clip(DETAIL_LEVEL, 0, 1))
    edges = feature.canny(
        gray,
        sigma=sigma,
        low_threshold=0.055,
        high_threshold=0.16,
        mask=allowed,
    )
    component_count, edge_labels, edge_stats, _ = cv2.connectedComponentsWithStats(
        edges.astype(np.uint8),
        8,
    )
    cleaned_edges = np.zeros_like(edges)
    min_component_size = max(8, int(min_length * 0.45))
    for label in range(1, component_count):
        if edge_stats[label, cv2.CC_STAT_AREA] >= min_component_size:
            cleaned_edges[edge_labels == label] = True
    edges = cleaned_edges

    face_strokes: list[tuple[float, np.ndarray]] = []
    for contour in measure.find_contours(edges.astype(np.float32), 0.5):
        points = np.column_stack((contour[:, 1], contour[:, 0]))
        if len(points) < 4:
            continue
        closed = bool(np.linalg.norm(points[0] - points[-1]) < 2.0)
        smoothed = smooth_stroke(points, closed=closed)
        length = polyline_length(smoothed)
        if length >= min_length:
            face_strokes.append((length, smoothed))

    face_strokes.sort(key=lambda item: item[0], reverse=True)
    stroke_limit = max(
        6,
        min(MAX_FACE_STROKES, int(6 + 18 * np.clip(DETAIL_LEVEL, 0, 1))),
    )
    strokes.extend(stroke for _, stroke in face_strokes[:stroke_limit])
    return order_strokes(strokes), (face_x, face_y, face_width, face_height)


def order_strokes(strokes: Sequence[np.ndarray]) -> list[np.ndarray]:
    if len(strokes) <= 1:
        return list(strokes)
    remaining = [np.asarray(stroke, dtype=np.float64) for stroke in strokes[1:]]
    ordered = [np.asarray(strokes[0], dtype=np.float64)]
    while remaining:
        current = ordered[-1][-1]
        best_index = 0
        reverse_best = False
        best_distance = math.inf
        for index, stroke in enumerate(remaining):
            start_distance = float(np.linalg.norm(current - stroke[0]))
            end_distance = float(np.linalg.norm(current - stroke[-1]))
            if start_distance < best_distance:
                best_index, reverse_best, best_distance = index, False, start_distance
            if end_distance < best_distance:
                best_index, reverse_best, best_distance = index, True, end_distance
        next_stroke = remaining.pop(best_index)
        ordered.append(next_stroke[::-1] if reverse_best else next_stroke)
    return ordered


def unit_tangent(points: np.ndarray, from_end: bool) -> np.ndarray:
    if from_end:
        vector = points[-1] - points[-2]
    else:
        vector = points[1] - points[0]
    length = float(np.linalg.norm(vector))
    if length <= 1e-6:
        return np.array([1.0, 0.0])
    return vector / length


def build_continuous_path(
    strokes: Sequence[np.ndarray],
) -> tuple[str, np.ndarray]:
    if not strokes:
        raise ValueError("no sufficiently long portrait lines were found")

    dense_points = [np.asarray(point, dtype=np.float64) for point in strokes[0]]
    first = dense_points[0]
    commands = [f"M {first[0]:.2f} {first[1]:.2f}"]
    for point in strokes[0][1:]:
        commands.append(f"L {point[0]:.2f} {point[1]:.2f}")

    for stroke in strokes[1:]:
        stroke = np.asarray(stroke, dtype=np.float64)
        start = dense_points[-1]
        end = stroke[0]
        distance = float(np.linalg.norm(end - start))
        tangent_start = unit_tangent(np.asarray(dense_points[-4:]), from_end=True)
        tangent_end = unit_tangent(stroke[:4], from_end=False)
        handle = min(distance * 0.34, max(2.0, min(30.0, distance * 0.42)))
        control_1 = start + tangent_start * handle
        control_2 = end - tangent_end * handle
        commands.append(
            "C "
            f"{control_1[0]:.2f} {control_1[1]:.2f} "
            f"{control_2[0]:.2f} {control_2[1]:.2f} "
            f"{end[0]:.2f} {end[1]:.2f}"
        )
        parameter = np.linspace(0, 1, max(12, int(distance / 3)), endpoint=True)
        for t in parameter[1:]:
            point = (
                (1 - t) ** 3 * start
                + 3 * (1 - t) ** 2 * t * control_1
                + 3 * (1 - t) * t**2 * control_2
                + t**3 * end
            )
            dense_points.append(point)
        simplified = cv2.approxPolyDP(
            stroke.astype(np.float32).reshape(-1, 1, 2),
            max(0.25, SMOOTHING_AMOUNT * 0.7),
            False,
        )[:, 0, :]
        for point in simplified[1:]:
            commands.append(f"L {point[0]:.2f} {point[1]:.2f}")
            dense_points.append(point.astype(np.float64))

    return " ".join(commands), np.asarray(dense_points, dtype=np.float64)


def render_line_png(
    points: np.ndarray,
    width: int,
    height: int,
    white: bool,
) -> np.ndarray:
    scale = max(1, ANTIALIAS_SCALE)
    alpha = np.zeros((height * scale, width * scale), dtype=np.uint8)
    scaled = np.rint(points * scale).astype(np.int32).reshape(-1, 1, 2)
    thickness = max(1, int(round(LINE_WIDTH * scale)))
    cv2.polylines(
        alpha,
        [scaled],
        False,
        255,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    radius = max(1, thickness // 2)
    for point in (scaled[0, 0], scaled[-1, 0]):
        cv2.circle(alpha, tuple(point), radius, 255, -1, cv2.LINE_AA)
    alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_AREA)
    color = 255 if white else 0
    bgra = np.empty((height, width, 4), dtype=np.uint8)
    bgra[:, :, :3] = color
    bgra[:, :, 3] = alpha
    return bgra


def animated_svg(path_data: str, width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" fill="none" stroke="#fff" '
        f'stroke-width="{LINE_WIDTH:g}" stroke-linecap="round" '
        f'stroke-linejoin="round">'
        "<style>"
        "@keyframes draw{to{stroke-dashoffset:0}}"
        "path{stroke-dasharray:1;stroke-dashoffset:1;"
        "animation:draw 7s cubic-bezier(.42,0,.58,1) forwards}"
        "@media(prefers-reduced-motion:reduce){path{stroke-dashoffset:0;"
        "animation:none}}"
        "</style>"
        f'<path pathLength="1" d="{path_data}"/>'
        "</svg>"
    )


def write_png(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OSError(f"could not encode PNG: {path}")
    encoded.tofile(str(path))


def create_line_art(
    image_path: Path,
    output_prefix: Path,
    force_grabcut: bool = False,
) -> list[Path]:
    image = read_image(image_path)
    alpha = None if force_grabcut else rembg_alpha(image)
    segmenter = "OpenCV GrabCut" if force_grabcut else "rembg"
    if alpha is None:
        alpha = grabcut_alpha(image)
        segmenter = "OpenCV GrabCut"
    mask = clean_person_mask(alpha)
    height, width = mask.shape
    strokes, _ = extract_strokes(image, mask)
    path_data, points = build_continuous_path(strokes)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    white_path = output_prefix.with_name(output_prefix.name + "-white.png")
    black_path = output_prefix.with_name(output_prefix.name + "-black.png")
    svg_path = output_prefix.with_name(output_prefix.name + ".svg")
    write_png(white_path, render_line_png(points, width, height, white=True))
    write_png(black_path, render_line_png(points, width, height, white=False))
    svg_path.write_text(
        animated_svg(path_data, width, height),
        encoding="utf-8",
    )
    print(f"Background removal: {segmenter}")
    print(f"Retained {len(strokes)} connected strokes in one continuous path.")
    return [white_path, black_path, svg_path]


def main() -> None:
    args = parse_args()
    try:
        outputs = create_line_art(
            args.image,
            args.output_prefix,
            force_grabcut=args.grabcut,
        )
    except (OSError, ValueError, cv2.error, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
