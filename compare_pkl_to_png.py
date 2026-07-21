# File: compare_pkl_to_png.py
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


TARGET_FILES = {
    "scans": [
        "scan_benign_10.pkl",
        "scan_malignant_5.pkl",
    ],
    "labels": [
        "label_benign_10.pkl",
        "label_malignant_5.pkl",
    ],
}


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def to_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    try:
        array = np.asarray(value)
        if array.dtype == object and array.size == 1:
            return to_numpy(array.item())
        if array.size == 0:
            return None
        return array
    except Exception:
        return None


def extract_array(data: Any, category: str) -> np.ndarray:
    preferred_keys = {
        "scans": ["scan", "image", "bmode", "data", "array", "input"],
        "labels": ["label", "mask", "segmentation", "target", "array"],
    }

    if isinstance(data, dict):
        for key in preferred_keys[category]:
            if key in data:
                array = to_numpy(data[key])
                if array is not None:
                    return array

        for value in data.values():
            array = to_numpy(value)
            if array is not None:
                return array

    array = to_numpy(data)
    if array is None:
        raise ValueError(f"Could not extract ndarray from {category} pickle.")

    return array


def squeeze_for_display(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    while result.ndim > 2:
        result = result[0]
    if result.ndim != 2:
        raise ValueError(f"Expected 2D displayable array, got shape {result.shape}")
    return result


def normalize_to_uint8(array: np.ndarray, is_label: bool) -> np.ndarray:
    result = np.asarray(array)

    if is_label:
        unique_values = np.unique(result)
        mapping = {value: idx for idx, value in enumerate(unique_values)}
        indexed = np.vectorize(mapping.get)(result)
        if len(unique_values) == 1:
            return np.zeros_like(indexed, dtype=np.uint8)
        scaled = indexed.astype(np.float32) / float(len(unique_values) - 1)
        return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)

    result = result.astype(np.float32)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    min_value = float(result.min())
    max_value = float(result.max())

    if np.isclose(min_value, max_value):
        return np.zeros_like(result, dtype=np.uint8)

    normalized = (result - min_value) / (max_value - min_value)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def save_png(array: np.ndarray, output_path: Path, is_label: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = normalize_to_uint8(array, is_label=is_label)
    plt.imsave(output_path, image, cmap="gray", vmin=0, vmax=255)


def calculate_metrics(local_array: np.ndarray, server_array: np.ndarray, is_label: bool) -> dict[str, Any]:
    if local_array.shape != server_array.shape:
        return {
            "shape_equal": False,
            "local_shape": str(local_array.shape),
            "server_shape": str(server_array.shape),
            "mae": "",
            "max_abs_diff": "",
            "exact_match_ratio": "",
        }

    local_float = local_array.astype(np.float32)
    server_float = server_array.astype(np.float32)
    abs_diff = np.abs(local_float - server_float)

    metrics = {
        "shape_equal": True,
        "local_shape": str(local_array.shape),
        "server_shape": str(server_array.shape),
        "mae": float(abs_diff.mean()),
        "max_abs_diff": float(abs_diff.max()),
        "exact_match_ratio": float((local_array == server_array).mean()),
    }

    if is_label:
        metrics["local_unique"] = str(np.unique(local_array).tolist())
        metrics["server_unique"] = str(np.unique(server_array).tolist())
    else:
        metrics["local_min"] = float(local_float.min())
        metrics["local_max"] = float(local_float.max())
        metrics["server_min"] = float(server_float.min())
        metrics["server_max"] = float(server_float.max())

    return metrics


def save_comparison_figure(
    local_array: np.ndarray,
    server_array: np.ndarray,
    output_path: Path,
    title: str,
    is_label: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if local_array.shape != server_array.shape:
        figure = plt.figure(figsize=(8, 3))
        plt.text(
            0.02,
            0.5,
            f"Shape mismatch\nlocal: {local_array.shape}\nserver: {server_array.shape}",
            fontsize=12,
            va="center",
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        return

    local_img = normalize_to_uint8(local_array, is_label=is_label)
    server_img = normalize_to_uint8(server_array, is_label=is_label)
    diff_img = np.abs(local_img.astype(np.int16) - server_img.astype(np.int16)).astype(np.uint8)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(local_img, cmap="gray", aspect="auto")
    axes[0].set_title("Local")
    axes[0].axis("off")

    axes[1].imshow(server_img, cmap="gray", aspect="auto")
    axes[1].set_title("Server")
    axes[1].axis("off")

    axes[2].imshow(diff_img, cmap="gray", aspect="auto")
    axes[2].set_title("Absolute Diff")
    axes[2].axis("off")

    metrics = calculate_metrics(local_array, server_array, is_label=is_label)
    figure.suptitle(
        f"{title}\nMAE={metrics['mae']:.6f} | MaxDiff={metrics['max_abs_diff']:.6f} | ExactMatch={metrics['exact_match_ratio']:.6f}"
        if metrics["shape_equal"]
        else f"{title}\nShape mismatch",
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def process_side(
    root: Path,
    output_dir: Path,
    side_name: str,
) -> dict[str, np.ndarray]:
    extracted: dict[str, np.ndarray] = {}

    for category, filenames in TARGET_FILES.items():
        for filename in filenames:
            source_path = root / category / filename
            if not source_path.exists():
                print(f"[MISSING] {side_name}: {source_path}")
                continue

            raw = load_pickle(source_path)
            array = extract_array(raw, category=category)
            array = squeeze_for_display(array)
            extracted[f"{category}/{filename}"] = array

            png_name = source_path.with_suffix(".png").name
            png_path = output_dir / side_name / category / png_name
            save_png(array, png_path, is_label=(category == "labels"))
            print(f"[OK] Saved PNG: {png_path}")

    return extracted


def write_metrics_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in all_keys:
                all_keys.append(key)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert selected PKL files to PNG and compare local vs server.")
    parser.add_argument(
        "--local-root",
        type=Path,
        required=True,
        help="Path to local final_training_data folder",
    )
    parser.add_argument(
        "--server-root",
        type=Path,
        required=True,
        help="Path to downloaded server final_training_data folder",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save PNGs and comparison results",
    )
    args = parser.parse_args()

    if not args.local_root.exists():
        raise FileNotFoundError(f"Local root not found: {args.local_root}")
    if not args.server_root.exists():
        raise FileNotFoundError(f"Server root not found: {args.server_root}")

    local_arrays = process_side(args.local_root, args.output_dir, "local")
    server_arrays = process_side(args.server_root, args.output_dir, "server")

    metric_rows: list[dict[str, Any]] = []

    for category, filenames in TARGET_FILES.items():
        for filename in filenames:
            key = f"{category}/{filename}"
            local_array = local_arrays.get(key)
            server_array = server_arrays.get(key)

            if local_array is None or server_array is None:
                metric_rows.append(
                    {
                        "file_key": key,
                        "status": "missing_on_local_or_server",
                    }
                )
                continue

            comparison_name = Path(filename).with_suffix("").name + "_compare.png"
            comparison_path = args.output_dir / "comparisons" / category / comparison_name

            save_comparison_figure(
                local_array=local_array,
                server_array=server_array,
                output_path=comparison_path,
                title=key,
                is_label=(category == "labels"),
            )
            print(f"[OK] Saved comparison: {comparison_path}")

            row = {
                "file_key": key,
                "status": "compared",
            }
            row.update(calculate_metrics(local_array, server_array, is_label=(category == "labels")))
            metric_rows.append(row)

    metrics_csv_path = args.output_dir / "comparison_metrics.csv"
    write_metrics_csv(metric_rows, metrics_csv_path)
    print(f"[OK] Saved metrics CSV: {metrics_csv_path}")


if __name__ == "__main__":
    main()