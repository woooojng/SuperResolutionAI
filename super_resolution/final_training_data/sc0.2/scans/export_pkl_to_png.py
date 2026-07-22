from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent

if BASE_DIR.name == "scans":
    ROOT_DIR = BASE_DIR.parent
elif BASE_DIR.name == "labels":
    ROOT_DIR = BASE_DIR.parent
else:
    ROOT_DIR = BASE_DIR

SCAN_DIR = ROOT_DIR / "scans"
LABEL_DIR = ROOT_DIR / "labels"


def load_array(path: Path, kind: str) -> np.ndarray:
    with path.open("rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        keys = ["scan", "image", "bmode", "data", "array"] if kind == "scan" else ["label", "mask", "segmentation", "target", "array"]
        for key in keys:
            if key in data:
                data = data[key]
                break
        else:
            data = next(iter(data.values()))

    arr = np.asarray(data)
    while arr.ndim > 2:
        arr = arr[0]

    if arr.ndim != 2:
        raise ValueError(f"{path.name}: expected 2D array, got shape {arr.shape}")

    return arr


def normalize_scan(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32))
    mn, mx = float(arr.min()), float(arr.max())
    if np.isclose(mn, mx):
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - mn) / (mx - mn)
    return (arr * 255).clip(0, 255).astype(np.uint8)


def normalize_label(arr: np.ndarray) -> np.ndarray:
    uniq = np.unique(arr)
    mapping = {v: i for i, v in enumerate(uniq)}
    arr = np.vectorize(mapping.get)(arr)
    if len(uniq) == 1:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = arr.astype(np.float32) / (len(uniq) - 1)
    return (arr * 255).clip(0, 255).astype(np.uint8)


def export_folder(folder: Path, kind: str) -> None:
    if not folder.exists():
        print(f"[SKIP] missing folder: {folder}")
        return

    for pkl_path in sorted(folder.glob("*.pkl")):
        arr = load_array(pkl_path, kind)
        img = normalize_scan(arr) if kind == "scan" else normalize_label(arr)
        out_path = pkl_path.with_suffix(".png")
        plt.imsave(out_path, img, cmap="gray", vmin=0, vmax=255)
        print(f"[OK] {out_path}")


def main() -> None:
    export_folder(SCAN_DIR, "scan")
    export_folder(LABEL_DIR, "label")


if __name__ == "__main__":
    main()