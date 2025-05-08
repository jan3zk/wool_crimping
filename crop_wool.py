import cv2
import numpy as np
import os
import argparse
import glob

def crop_wool(image, downscale=0.25, border=(0,0,0,0)):
    border_top, border_bottom, border_left, border_right = border

    # 1) Downscale for speed
    small = cv2.resize(image, (0,0), fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    # 2) Grayscale + blur + Otsu threshold
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Foreground = bright regions
    mask = thresh
    # 3) Morphological cleanup (small-scale)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    # 4) Upscale mask
    mask_full = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    # 5) Find largest contour & crop
    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Apply additional border cropping
    x = max(0, x + border_left)
    y = max(0, y + border_top)
    w = max(1, w - border_left - border_right)
    h = max(1, h - border_top - border_bottom)

    return (x, y, w, h)

def crop_and_save(image_path, bbox, output_dir, prefix=""):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    x, y, w, h = bbox
    cropped_img = img[y:y+h, x:x+w]
    filename = prefix + os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, cropped_img)
    print(f"Saved: {output_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Crop wool samples from images')
    parser.add_argument('input_image', help='Path to the input image')
    parser.add_argument('--output', '-o', default='crop', help='Output directory path (default: "crop")')
    parser.add_argument('--downscale', '-d', type=float, default=0.25, help='Downscale factor for processing (default: 0.25)')
    parser.add_argument('--border', '-b', type=int, nargs=4, metavar=('TOP', 'BOTTOM', 'LEFT', 'RIGHT'),
                        default=(0, 0, 0, 0), help='Pixels to crop from each side (top bottom left right, default: 0 0 0 0)')
    parser.add_argument('--pattern', '-p', type=str, help='Glob pattern for batch cropping images (e.g., "L_*.png")')

    args = parser.parse_args()

    input_path = args.input_image
    input_dir = os.path.dirname(input_path)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not read image at {input_path}")
        return

    bbox = crop_wool(img, downscale=args.downscale, border=args.border)
    if bbox is None:
        print(f"Failed to detect sample in {input_path}")
        return

    if args.pattern:
        pattern_path = os.path.join(input_dir, args.pattern)
        for file_path in glob.glob(pattern_path):
            crop_and_save(file_path, bbox, args.output)
    else:
        crop_and_save(input_path, bbox, args.output)

if __name__ == "__main__":
    main()
