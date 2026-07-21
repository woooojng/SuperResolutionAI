# File: diff_two_pngs.py
# File: diff_two_pngs.py
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image(path: Path, grayscale: bool = True) -> np.ndarray:
    image = Image.open(path)
    if grayscale:
        image = image.convert("L")
    return np.array(image, dtype=np.uint8)


def resize_to_match(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image)
    resized = pil_image.resize(target_size, Image.BILINEAR)
    return np.array(resized, dtype=np.uint8)


def compute_difference(
    image_a: np.ndarray,
    image_b: np.ndarray,
    mode: str = "abs",
) -> np.ndarray:
    a = image_a.astype(np.int16)
    b = image_b.astype(np.int16)

    if mode == "abs":
        diff = np.abs(a - b)
    elif mode == "subtract":
        diff = np.clip(a - b, 0, 255)
    elif mode == "subtract_reverse":
        diff = np.clip(b - a, 0, 255)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return diff.astype(np.uint8)


def save_image_with_colorbar(
    image: np.ndarray,
    output_path: Path,
    cmap: str = "inferno",
    title: str = "Difference Image",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 6))
    im = axis.imshow(image, cmap=cmap, vmin=0, vmax=255)
    axis.set_title(title)
    axis.axis("off")

    colorbar = figure.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Absolute pixel difference (0-255)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a PNG diff image from two PNG files with a colorbar."
    )
    parser.add_argument("image_a", type=Path, help="First PNG path")
    parser.add_argument("image_b", type=Path, help="Second PNG path")
    parser.add_argument("output", type=Path, help="Output PNG path")
    parser.add_argument(
        "--mode",
        choices=["abs", "subtract", "subtract_reverse"],
        default="abs",
        help="Diff mode. abs = |A-B|, subtract = A-B clipped, subtract_reverse = B-A clipped",
    )
    parser.add_argument(
        "--resize-to-first",
        action="store_true",
        help="Resize the second image to the first image size if shapes differ",
    )
    parser.add_argument(
        "--cmap",
        default="inferno",
        help="Matplotlib colormap name, e.g. inferno, jet, viridis, magma, hot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image_a.exists():
        raise FileNotFoundError(f"First image not found: {args.image_a}")
    if not args.image_b.exists():
        raise FileNotFoundError(f"Second image not found: {args.image_b}")

    image_a = load_image(args.image_a, grayscale=True)
    image_b = load_image(args.image_b, grayscale=True)

    if image_a.shape != image_b.shape:
        if not args.resize_to_first:
            raise ValueError(
                f"Image shapes differ: {image_a.shape} vs {image_b.shape}. "
                "Use --resize-to-first to force resize."
            )
        target_size = (image_a.shape[1], image_a.shape[0])
        image_b = resize_to_match(image_b, target_size)

    diff_image = compute_difference(image_a, image_b, mode=args.mode)
    save_image_with_colorbar(
        diff_image,
        args.output,
        cmap=args.cmap,
        title=f"Diff ({args.mode})",
    )

    print(f"Saved diff image: {args.output}")
    print(f"Image A shape: {image_a.shape}")
    print(f"Image B shape: {image_b.shape}")
    print(f"Diff min/max: {diff_image.min()} / {diff_image.max()}")
    print(f"Diff mean: {diff_image.mean():.4f}")


if __name__ == "__main__":
    main()