from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x)
    except Exception:
        return x


def pick_image(data):
    if isinstance(data, dict):
        for key in ["image", "scan", "bmode", "data", "array", "input"]:
            if key in data:
                arr = to_numpy(data[key])
                if isinstance(arr, np.ndarray):
                    return arr, f"dict[{key}]"
        for key, value in data.items():
            arr = to_numpy(value)
            if isinstance(arr, np.ndarray):
                return arr, f"dict[{key}]"
        raise ValueError("No ndarray-like value found in scan pickle.")

    arr = to_numpy(data)
    if not isinstance(arr, np.ndarray):
        raise ValueError("Unsupported scan pickle format.")
    return arr, "root"


def pick_label(data):
    if isinstance(data, dict):
        for key in ["label", "mask", "segmentation", "target", "array"]:
            if key in data:
                arr = to_numpy(data[key])
                if isinstance(arr, np.ndarray):
                    return arr, f"dict[{key}]"
        for key, value in data.items():
            arr = to_numpy(value)
            if isinstance(arr, np.ndarray):
                return arr, f"dict[{key}]"
        return None, "none"

    arr = to_numpy(data)
    if isinstance(arr, np.ndarray):
        return arr, "root"
    return None, "none"


def squeeze_for_display(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def main():
    base = Path("C:/Users/CMME260629/Desktop/Synthetic_Data/super_resolution/final_training_data")
    scan_path = base / "scans" / "scan_benign_10.pkl"
    label_path = base / "labels" / "label_benign_10.pkl"
    out_path = Path("C:/Users/CMME260629/Desktop/benign_10_post_processing_results.png")

    scan_obj = load_pickle(scan_path)
    label_obj = load_pickle(label_path)

    scan_arr, scan_src = pick_image(scan_obj)
    label_arr, label_src = pick_label(label_obj)

    scan_arr = squeeze_for_display(scan_arr)
    if label_arr is not None:
        label_arr = squeeze_for_display(label_arr)

    fig, axes = plt.subplots(1, 2 if label_arr is not None else 1, figsize=(12, 5))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    axes[0].imshow(scan_arr, cmap="gray", aspect="auto")
    axes[0].set_title(f"Scan ({scan_src})")
    axes[0].axis("off")

    if label_arr is not None:
        axes[1].imshow(label_arr, cmap="gray", aspect="auto")
        axes[1].set_title(f"Label ({label_src})")
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()