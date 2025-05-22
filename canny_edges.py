#!/usr/bin/env python3
"""
glasswool_canny.py: Median-filtered Canny edge detection with configurable downsampling and semi-transparent red overlay.

Usage:
    python glasswool_canny.py input_image [--alpha ALPHA] [--scale SCALE] [--prefix PREFIX]

Outputs:
    <prefix>_overlay.png  - Full-resolution RGB image with masked pixels tinted red
    <prefix>_mask.png     - Logical Canny mask at full input resolution
"""
import argparse
import os
import numpy as np
import cv2
from skimage.feature import canny

def glasswool_canny(image: np.ndarray, alpha: float = 0.3, scale: float = 0.25):
    """
    Perform median-filtered Canny edge detection and overlay red tint on edges.

    Parameters:
        image: HxWx3 uint8 BGR image
        alpha: transparency of red overlay (0.0–1.0)
        scale: downsampling factor for processing (0.0 < scale <= 1.0)

    Returns:
        mask_full: boolean mask at full input resolution
        overlay: uint8 BGR image with red tint overlay
    """
    # Validate parameters
    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))
    scale = float(scale)
    if not (0 < scale <= 1.0):
        raise ValueError("Scale must be in the range (0, 1].")

    # Validate and convert image
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be an RGB image (HxWx3).")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Down-sample by given scale
    h, w = gray.shape
    new_w, new_h = int(w * scale), int(h * scale)
    small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 3x3 median filter
    small_med = cv2.medianBlur(small, 3)

    # Canny edge detection
    mask_down = canny(small_med.astype(float) / 255.0,
                      sigma=1.0,
                      #low_threshold=0.05,   # e.g. 5% of max gradient
                      #high_threshold=0.15   # e.g. 15% of max gradient
                      )

    # Upsample mask to full size
    mask_full = cv2.resize(mask_down.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    # Prepare overlay
    overlay = image.astype(np.float64)
    B, G, R = overlay[:, :, 0], overlay[:, :, 1], overlay[:, :, 2]

    # Apply red tint where mask is True
    R[mask_full] = (1 - alpha) * R[mask_full] + alpha * 255
    G[mask_full] = (1 - alpha) * G[mask_full]
    B[mask_full] = (1 - alpha) * B[mask_full]

    overlay = np.stack([B, G, R], axis=-1).astype(np.uint8)
    return mask_full, overlay

def main():
    parser = argparse.ArgumentParser(description="Glasswool Canny edge overlay script with adjustable scale.")
    parser.add_argument('input', help='Path to input image')
    parser.add_argument('--alpha', type=float, default=0.5, help='Overlay transparency (0.0–1.0)')
    parser.add_argument('--scale', type=float, default=0.25, help='Downsampling factor for edge detection (0.0–1.0)')
    parser.add_argument('--prefix', type=str, default=None, help='Output filename prefix')
    args = parser.parse_args()

    # Read image
    img = cv2.imread(args.input)
    if img is None:
        parser.error(f"Could not load image at '{args.input}'")

    # Run processing
    mask_full, overlay = glasswool_canny(img, args.alpha, args.scale)

    # Determine output filenames
    base = args.prefix or os.path.splitext(os.path.basename(args.input))[0]
    mask_path = f"{base}_mask.png"
    overlay_path = f"{base}_overlay.png"

    # Save results
    cv2.imwrite(mask_path, (mask_full.astype(np.uint8) * 255))
    cv2.imwrite(overlay_path, overlay)

    print(f"Saved mask to {mask_path}")
    print(f"Saved overlay to {overlay_path}")

if __name__ == '__main__':
    main()