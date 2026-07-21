# C:\Users\CMME260629\Desktop\Synthetic_Data_ver4\super_resolution\final_training_data\scans\output\calc_abs_error.py

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_as_grayscale(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

    if image.ndim == 2:
        return image.astype(np.float32)

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    raise ValueError(f"지원하지 않는 이미지 shape입니다: {image.shape}")


def compute_absolute_error(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    return np.abs(image1 - image2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="두 이미지의 absolute error를 계산하고 colorbar가 포함된 colormap 이미지 1개를 저장합니다."
    )
    parser.add_argument("image1", type=str, help="첫 번째 이미지 경로")
    parser.add_argument("image2", type=str, help="두 번째 이미지 경로")
    parser.add_argument(
        "--cmap",
        "--colormap",
        dest="cmap",
        type=str,
        default="turbo",
        choices=["turbo", "jet", "hot", "bone", "viridis", "plasma", "inferno", "magma"],
        help="matplotlib colormap 이름",
    )
    parser.add_argument("--dpi", type=int, default=200, help="저장 DPI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image1_path = Path(args.image1).resolve()
    image2_path = Path(args.image2).resolve()

    image1 = load_as_grayscale(image1_path)
    image2 = load_as_grayscale(image2_path)

    if image1.shape != image2.shape:
        raise ValueError(
            f"이미지 크기가 다릅니다. image1={image1.shape}, image2={image2.shape}"
        )

    abs_error = compute_absolute_error(image1, image2)

    output_path = (
        image1_path.parent
        / f"{image1_path.stem}__vs__{image2_path.stem}_abs_error_colormap.png"
    )

    vmin = float(np.min(abs_error))
    vmax = float(np.max(abs_error))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    height, width = abs_error.shape
    fig_width = max(6, width / 100)
    fig_height = max(6, height / 100)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(abs_error, cmap=args.cmap, vmin=vmin, vmax=vmax)
    ax.set_title("Absolute Error")
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Absolute Error Value")

    plt.tight_layout()
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    print(f"[완료] 저장: {output_path}")
    print(f"MAE: {float(np.mean(abs_error)):.6f}")
    print(f"Min AE: {float(np.min(abs_error)):.6f}")
    print(f"Max AE: {float(np.max(abs_error)):.6f}")


if __name__ == "__main__":
    main()