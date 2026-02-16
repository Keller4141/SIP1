"""Task 4: Histogram-based processing (4.1, 4.2, 4.3, 4.4).

This script produces:
- CDF from histogram (4.1)
- C(I) mapping image (4.2)
- CDF pseudo-inverse (4.3)
- Histogram matching with images and CDF comparison (4.4)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def compute_cdf_from_histogram(hist: np.ndarray) -> np.ndarray:
    """Compute normalized CDF from a 1D histogram."""
    hist = np.asarray(hist, dtype=np.float64)
    if hist.ndim != 1:
        raise ValueError("Histogram must be a 1D array")

    total = hist.sum()
    if total <= 0:
        raise ValueError("Histogram sum must be > 0")

    return np.cumsum(hist) / total


def apply_cdf_to_image(image_uint8: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Compute C(I): map each pixel intensity i to C(i), output in [0,1]."""
    image_uint8 = np.asarray(image_uint8)
    cdf = np.asarray(cdf, dtype=np.float64)

    if image_uint8.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale image")
    if cdf.ndim != 1 or cdf.size != 256:
        raise ValueError("CDF must be a 1D array with 256 elements")

    return cdf[image_uint8.astype(np.uint8)]


def compute_cdf_pseudoinverse(cdf: np.ndarray, levels: int = 256) -> np.ndarray:
    """Compute C^{-1}(l) = min{s | C(s) >= l} for discretized l in [0,1]."""
    cdf = np.asarray(cdf, dtype=np.float64)
    if cdf.ndim != 1:
        raise ValueError("CDF must be a 1D array")
    if np.any(np.diff(cdf) < 0):
        raise ValueError("CDF must be non-decreasing")

    l_values = np.linspace(0.0, 1.0, levels)
    pseudo_inv = np.zeros(levels, dtype=np.uint8)

    for i, l in enumerate(l_values):
        idx = np.where(cdf >= l)[0]
        if idx.size == 0:
            pseudo_inv[i] = len(cdf) - 1
        else:
            pseudo_inv[i] = np.min(idx)

    return pseudo_inv


def histogram_match(source_img: np.ndarray, target_img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Histogram-match source image to target image using J = C2^{-1}(C1(I1))."""
    source_img = np.asarray(source_img, dtype=np.uint8)
    target_img = np.asarray(target_img, dtype=np.uint8)

    hist_src, _ = np.histogram(source_img.ravel(), bins=256, range=(0, 256))
    hist_tgt, _ = np.histogram(target_img.ravel(), bins=256, range=(0, 256))

    cdf_src = compute_cdf_from_histogram(hist_src)
    cdf_tgt = compute_cdf_from_histogram(hist_tgt)
    cdf_tgt_inv = compute_cdf_pseudoinverse(cdf_tgt, levels=256)

    c1_of_source = apply_cdf_to_image(source_img, cdf_src)
    idx = np.clip(np.round(c1_of_source * 255), 0, 255).astype(np.uint8)
    matched = cdf_tgt_inv[idx]

    hist_matched, _ = np.histogram(matched.ravel(), bins=256, range=(0, 256))
    cdf_matched = compute_cdf_from_histogram(hist_matched)

    return matched.astype(np.uint8), cdf_src, cdf_tgt, cdf_matched


def resolve_input_path(script_dir: Path, *relative_candidates: str) -> Path:
    """Find first existing input path across common roots."""
    roots = [script_dir, script_dir.parent, Path.cwd()]
    for rel in relative_candidates:
        for root in roots:
            p = (root / rel).resolve()
            if p.exists():
                return p
    raise FileNotFoundError(f"Could not find any of: {relative_candidates}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "outputs_task4"
    output_dir.mkdir(exist_ok=True)

    pout_path = resolve_input_path(script_dir, "Images/pout.tif")
    logo_path = resolve_input_path(script_dir, "Images/KU_Logo.png")

    img = np.array(Image.open(pout_path).convert("L"))

    # 4.1
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
    cdf = compute_cdf_from_histogram(hist)

    x = np.arange(256)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("pout.tif (grayscale)")
    axes[0].axis("off")

    axes[1].plot(x, hist, color="tab:blue")
    axes[1].set_title("Histogram")
    axes[1].set_xlabel("Intensity value")
    axes[1].set_ylabel("Pixel count")

    axes[2].plot(x, cdf, color="tab:red")
    axes[2].set_title("CDF (normalized)")
    axes[2].set_xlabel("Intensity value")
    axes[2].set_ylabel("Cumulative probability")
    axes[2].set_ylim(0, 1.02)

    fig.tight_layout()
    fig.savefig(output_dir / "task4_1_cdf_on_pout.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4.2
    ci = apply_cdf_to_image(img, cdf)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original: pout.tif")
    axes[0].axis("off")

    axes[1].imshow(ci, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Mapped image: C(I)")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "task4_2_c_of_i.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4.3
    cdf_pinv = compute_cdf_pseudoinverse(cdf)

    # 4.4
    i1 = img
    i2 = np.array(Image.open(logo_path).convert("L"))
    j_matched, cdf_i1, cdf_i2, cdf_j = histogram_match(i1, i2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(i1, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("I1: Source (pout.tif)")
    axes[0].axis("off")

    axes[1].imshow(i2, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("I2: Target (KU_Logo.png)")
    axes[1].axis("off")

    axes[2].imshow(j_matched, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("J: Histogram matched")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "task4_4_input_and_matched_images.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(x, cdf_i1, label="CDF original I1 (pout.tif)", linewidth=2)
    plt.plot(x, cdf_i2, label="CDF target I2 (KU_Logo.png)", linewidth=2)
    plt.plot(x, cdf_j, label="CDF matched J", linewidth=2, linestyle="--")
    plt.xlabel("Intensity value")
    plt.ylabel("Cumulative probability")
    plt.title("CDF comparison: original vs target vs matched")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "task4_4_cdf_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Task 4 completed.")
    print(f"Saved figures in: {output_dir}")
    print(f"CDF shape: {cdf.shape}")
    print(f"First 10 CDF values: {np.round(cdf[:10], 6)}")
    print(f"Last CDF value: {cdf[-1]}")
    print(f"C(I) dtype: {ci.dtype}")
    print(f"C(I) min/max: {float(ci.min())} {float(ci.max())}")
    print(f"Pseudo-inverse shape: {cdf_pinv.shape}")
    print(f"First 10 pseudo-inverse values: {cdf_pinv[:10]}")
    print(f"Last 10 pseudo-inverse values: {cdf_pinv[-10:]}")
    print(f"I1 shape: {i1.shape}, I2 shape: {i2.shape}, J shape: {j_matched.shape}")


if __name__ == "__main__":
    main()
