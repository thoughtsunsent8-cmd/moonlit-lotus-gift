"""Create line-art and gradient image layers from a transparent PNG cutout.

Usage:
    python make_layers.py cutout.png img name [teal|rose|violet]

Outputs:
    name-lines.svg  Animated vector outline
    name-lines.png  Static transparent outline
    name-fill.png   Gradient-tinted transparent figure
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


GRADIENTS = {
    "teal": [(30, 215, 170), (50, 140, 230), (150, 90, 235)],
    "rose": [(255, 150, 120), (235, 90, 150), (120, 70, 220)],
    "violet": [(255, 200, 110), (200, 100, 200), (70, 90, 230)],
}
OUTPUT_WIDTH = 800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create animated outline and gradient image layers "
        "from a transparent PNG cutout."
    )
    parser.add_argument("source", type=Path, help="transparent PNG cutout")
    parser.add_argument("output_dir", type=Path, help="directory for image layers")
    parser.add_argument("name", help="output filename prefix, e.g. your-smile")
    parser.add_argument(
        "gradient",
        nargs="?",
        choices=GRADIENTS,
        default="teal",
        help="gradient palette (default: teal)",
    )
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        parser.error("name must be a filename prefix, not a path")
    return args


def write_image(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise OSError(f"Could not encode image: {path}")
    try:
        encoded.tofile(str(path))
    except OSError as error:
        raise OSError(f"Could not write image: {path}") from error


def load_cutout(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Cutout image not found: {path}")
    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise ValueError(f"Could not read image: {path}")
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(
            "The source must be a PNG with transparency (an RGBA image)."
        )

    ys, xs = np.where(rgba[..., 3] > 10)
    if ys.size == 0:
        raise ValueError("The source image has no visible pixels.")

    rgba = rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    scale = OUTPUT_WIDTH / rgba.shape[1]
    resized = cv2.resize(
        rgba,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    return resized[..., :3], resized[..., 3]


def extract_lines(
    bgr: np.ndarray,
    alpha: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.bilateralFilter(gray, 9, 60, 9)
    smoothed = cv2.GaussianBlur(smoothed, (7, 7), 1.5)
    edges = cv2.Canny(smoothed, 35, 100)
    inner = cv2.erode(
        (alpha > 128).astype(np.uint8),
        np.ones((7, 7), np.uint8),
    )
    edges = edges * inner
    face_gray = cv2.bilateralFilter(gray, 7, 45, 7)
    face_gray = cv2.GaussianBlur(face_gray, (5, 5), 1.0)
    face_detail = cv2.Canny(face_gray, 32, 90)
    face_mask = np.zeros_like(alpha)
    face_center = (int(alpha.shape[1] * 0.45), int(alpha.shape[0] * 0.13))
    face_axes = (int(alpha.shape[1] * 0.15), int(alpha.shape[0] * 0.125))
    cv2.ellipse(
        face_mask,
        face_center,
        face_axes,
        0,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    face_detail = cv2.bitwise_and(face_detail, inner, face_mask)
    edges = cv2.bitwise_or(edges, face_detail)
    mouth_center = (int(alpha.shape[1] * 0.465), int(alpha.shape[0] * 0.25))
    mouth_axes = (int(alpha.shape[1] * 0.06), int(alpha.shape[0] * 0.035))
    mouth_mask = np.zeros_like(alpha)
    cv2.ellipse(mouth_mask, mouth_center, mouth_axes, 0, 0, 360, 255, -1)
    edges = cv2.bitwise_and(edges, cv2.bitwise_not(mouth_mask))
    outline = cv2.morphologyEx(
        (alpha > 128).astype(np.uint8) * 255,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    )
    edges = cv2.bitwise_or(edges, outline)
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    contours = [
        contour[:, 0, :]
        for contour in contours
        if len(contour) > 20
        and cv2.arcLength(contour, False) > 30
    ]
    cx, cy = mouth_center
    half_width = int(alpha.shape[1] * 0.046)
    lip_height = max(5, int(alpha.shape[1] * 0.01))

    def bezier(
        start: tuple[int, int],
        control_1: tuple[int, int],
        control_2: tuple[int, int],
        end: tuple[int, int],
    ) -> np.ndarray:
        t = np.linspace(0, 1, 25)[:, None]
        points = (
            (1 - t) ** 3 * np.asarray(start)
            + 3 * (1 - t) ** 2 * t * np.asarray(control_1)
            + 3 * (1 - t) * t**2 * np.asarray(control_2)
            + t**3 * np.asarray(end)
        )
        return np.rint(points).astype(np.int32)

    left = (cx - half_width, cy + 2)
    right = (cx + half_width, cy + 2)
    contours.append(
        bezier(
            left,
            (cx - half_width // 3, cy - lip_height // 2),
            (cx + half_width // 3, cy - lip_height // 2),
            right,
        )
    )
    contours.append(
        bezier(
            (cx - half_width // 2, cy + lip_height // 2),
            (cx - half_width // 4, cy + lip_height // 2 + 2),
            (cx + half_width // 4, cy + lip_height // 2 + 2),
            (cx + half_width // 2, cy + lip_height // 2 + 1),
        )
    )
    contours.sort(
        key=lambda contour: (
            contour[:, 1].min() // 60,
            contour[:, 0].min(),
        )
    )

    line_alpha = np.zeros_like(alpha)
    for contour in contours:
        cv2.polylines(
            line_alpha,
            [contour.reshape(-1, 1, 2)],
            False,
            255,
            1,
            cv2.LINE_AA,
        )
    return line_alpha, contours


def animated_svg(contours: list[np.ndarray], width: int, height: int) -> str:
    paths = []
    path_count = len(contours)
    for index, contour in enumerate(contours):
        points = cv2.approxPolyDP(
            contour.reshape(-1, 1, 2),
            0.6,
            False,
        )[:, 0, :]
        if len(points) < 2:
            continue
        path_data = "M" + " L".join(f"{x} {y}" for x, y in points)
        delay = 3.5 * index / max(path_count - 1, 1)
        paths.append(
            f'<path pathLength="1" d="{path_data}" '
            f'style="animation-delay:{delay:.3f}s"/>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" fill="none" stroke="#fff" '
        f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
        "<style>"
        "@keyframes draw{to{stroke-dashoffset:0}}"
        "path{stroke-dasharray:1;stroke-dashoffset:1;"
        "animation:draw 5.2s cubic-bezier(0.42,0,0.58,1) forwards}"
        "@media(prefers-reduced-motion:reduce){"
        "path{stroke-dashoffset:0;animation:none}}"
        "</style>"
        + "".join(paths)
        + "</svg>"
    )


def gradient_fill(
    bgr: np.ndarray,
    alpha: np.ndarray,
    color_stops: list[tuple[int, int, int]],
) -> np.ndarray:
    height, width = alpha.shape
    stops = np.asarray(color_stops, dtype=np.float32)[:, ::-1]
    position = np.linspace(0, 1, height) * (len(stops) - 1)
    index = np.clip(position.astype(int), 0, len(stops) - 2)
    fraction = (position - index)[:, None]
    colors = stops[index] * (1 - fraction) + stops[index + 1] * fraction

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    luminance = cv2.normalize(
        gray,
        None,
        0.25,
        1.0,
        cv2.NORM_MINMAX,
    )
    fill = np.clip(colors[:, None, :] * luminance[..., None], 0, 255)
    mask = cv2.GaussianBlur(
        (alpha > 128).astype(np.float32) * 255,
        (0, 0),
        1.0,
    ).astype(np.uint8)
    return np.dstack((fill.astype(np.uint8), mask))


def make_layers(
    source: Path,
    output_dir: Path,
    name: str,
    gradient: str = "teal",
) -> tuple[Path, Path, Path]:
    bgr, alpha = load_cutout(source)
    line_alpha, contours = extract_lines(bgr, alpha)
    output_dir.mkdir(parents=True, exist_ok=True)

    height, width = alpha.shape
    outline = np.dstack((np.full((height, width, 3), 255, np.uint8), line_alpha))
    fill = gradient_fill(bgr, alpha, GRADIENTS[gradient])

    lines_png = output_dir / f"{name}-lines.png"
    lines_svg = output_dir / f"{name}-lines.svg"
    fill_png = output_dir / f"{name}-fill.png"
    write_image(lines_png, outline)
    lines_svg.write_text(
        animated_svg(contours, width, height),
        encoding="utf-8",
    )
    write_image(fill_png, fill)
    return lines_svg, lines_png, fill_png


def main() -> None:
    args = parse_args()
    try:
        outputs = make_layers(
            args.source,
            args.output_dir,
            args.name,
            args.gradient,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"Created layers for {args.name}:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
