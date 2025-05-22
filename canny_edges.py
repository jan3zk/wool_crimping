#!/usr/bin/env python3
"""
glasswool_canny.py: Median-filtered Canny edge detection with configurable downsampling and semi-transparent red overlay.

Usage:
    python glasswool_canny.py input_image [--alpha ALPHA] [--scale SCALE] \
        [--mask_path MASK_PATH] [--overlay_path OVERLAY_PATH]

Outputs:
    Specified by --mask_path and --overlay_path
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
    alpha = max(0.0, min(1.0, float(alpha)))
    scale = float(scale)
    if not (0 < scale <= 1.0):
        raise ValueError("Scale must be in the range (0, 1].")

    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be an RGB image (HxWx3).")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    new_w, new_h = int(w * scale), int(h * scale)
    small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    small_med = cv2.medianBlur(small, 3)

    mask_down = canny(small_med.astype(float) / 255.0,
                      sigma=1.0,
                      #low_threshold=0.05,   # e.g. 5% of max gradient
                      #high_threshold=0.15   # e.g. 15% of max gradient
                      )
    mask_full = cv2.resize(mask_down.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    overlay = image.astype(np.float64)
    B, G, R = overlay[:, :, 0], overlay[:, :, 1], overlay[:, :, 2]

    R[mask_full] = (1 - alpha) * R[mask_full] + alpha * 255
    G[mask_full] = (1 - alpha) * G[mask_full]
    B[mask_full] = (1 - alpha) * B[mask_full]

    overlay = np.stack([B, G, R], axis=-1).astype(np.uint8)
    return mask_full, overlay

def main():
    parser = argparse.ArgumentParser(description="Glasswool Canny edge overlay script with adjustable scale and output paths.")
    parser.add_argument('input', help='Path to input image')
    parser.add_argument('--alpha', type=float, default=0.5, help='Overlay transparency (0.0–1.0)')
    parser.add_argument('--scale', type=float, default=0.25, help='Downsampling factor (0.0–1.0]')
    parser.add_argument('--mask_path', type=str, required=True, help='Path to save the binary mask PNG')
    parser.add_argument('--overlay_path', type=str, required=True, help='Path to save the overlay image PNG')
    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        parser.error(f"Could not load image at '{args.input}'")

    mask_full, overlay = glasswool_canny(img, args.alpha, args.scale)

    cv2.imwrite(args.mask_path, (mask_full.astype(np.uint8) * 255))
    cv2.imwrite(args.overlay_path, overlay)

    print(f"Saved mask to {args.mask_path}")
    print(f"Saved overlay to {args.overlay_path}")

if __name__ == '__main__':
    main()
