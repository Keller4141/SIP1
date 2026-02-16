"""Task 1: Gamma correction (1.1, 1.2, 1.3).

This script produces figures for:
- 1.1 Grayscale gamma transform on KU_Logo
- 1.2 RGB channel-wise gamma on autumn.tif
- 1.3 HSV V-channel gamma on autumn.tif
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import hsv2rgb, rgb2hsv


def gamma_transform_gray(gray_image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma transform to a grayscale image in range [0, 255]."""
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    img = gray_image.astype(np.float32) / 255.0
    transformed = np.power(img, gamma)
    return (transformed * 255.0).clip(0, 255).astype(np.uint8)


def apply_gamma_rgb_channelwise(rgb_image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply grayscale gamma correction independently to R, G, and B channels."""
    r_corr = gamma_transform_gray(rgb_image[:, :, 0], gamma)
    g_corr = gamma_transform_gray(rgb_image[:, :, 1], gamma)
    b_corr = gamma_transform_gray(rgb_image[:, :, 2], gamma)
    return np.stack([r_corr, g_corr, b_corr], axis=-1)


def apply_gamma_hsv_v(rgb_image: np.ndarray, gamma: float) -> np.ndarray:
    """Convert RGB to HSV, apply gamma on V channel, convert back to RGB."""
    rgb_float = rgb_image.astype(np.float32) / 255.0
    hsv = rgb2hsv(rgb_float)

    v_uint8 = (hsv[:, :, 2] * 255.0).clip(0, 255).astype(np.uint8)
    v_gamma_uint8 = gamma_transform_gray(v_uint8, gamma)
    hsv[:, :, 2] = v_gamma_uint8.astype(np.float32) / 255.0

    rgb_gamma = hsv2rgb(hsv)
    return (rgb_gamma * 255.0).clip(0, 255).astype(np.uint8)


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
    output_dir = script_dir / "outputs_task1"
    output_dir.mkdir(exist_ok=True)

    logo_path = resolve_input_path(script_dir, "Images/KU_Logo.png")
    autumn_path = resolve_input_path(script_dir, "Images/autumn.tif")

    gamma_values = [0.4, 0.8, 1.0, 1.6, 2.2]
    gamma_color = 0.8

    # Task 1.1
    gray = np.array(Image.open(logo_path).convert("L"))
    gray_results = [gamma_transform_gray(gray, g) for g in gamma_values]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()

    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original (grayscale)")
    axes[0].axis("off")

    for i, (g, img) in enumerate(zip(gamma_values, gray_results), start=1):
        axes[i].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[i].set_title(f"Gamma = {g}")
        axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "task1_1_grayscale_gamma.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Task 1.2
    autumn_rgb = np.array(Image.open(autumn_path).convert("RGB"))
    autumn_rgb_gamma = apply_gamma_rgb_channelwise(autumn_rgb, gamma_color)

    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(autumn_rgb)
    plt.title("Original autumn.tif")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(autumn_rgb_gamma)
    plt.title(f"RGB channel-wise gamma (gamma={gamma_color})")
    plt.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "task1_2_rgb_channel_gamma.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Task 1.3
    autumn_hsv_gamma = apply_gamma_hsv_v(autumn_rgb, gamma_color)

    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(autumn_rgb)
    plt.title("Original autumn.tif")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(autumn_hsv_gamma)
    plt.title(f"HSV V-channel gamma (gamma={gamma_color})")
    plt.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "task1_3_hsv_v_channel_gamma.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 1.2 vs 1.3 comparison
    fig = plt.figure(figsize=(16, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(autumn_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(autumn_rgb_gamma)
    plt.title("1.2 RGB channel-wise gamma")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(autumn_hsv_gamma)
    plt.title("1.3 HSV V-channel gamma")
    plt.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "task1_2_vs_1_3_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Task 1 completed.")
    print(f"Saved figures in: {output_dir}")


if __name__ == "__main__":
    main()
