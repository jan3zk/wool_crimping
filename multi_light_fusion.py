import numpy as np
import cv2
import argparse
import sys
import os

def load_images(image_paths):
    images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in image_paths]
    if any(img is None for img in images):
        raise ValueError("One or more image paths are invalid or images failed to load.")
    return images

def fuse_images_min(images):
    return np.min(images, axis=0).astype(np.uint8)

def fuse_images_max(images):
    return np.max(images, axis=0).astype(np.uint8)

def fuse_images_mean(images):
    return np.mean(images, axis=0).astype(np.uint8)

def gradient_magnitude(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    return magnitude

def fuse_images_gradient(images):
    gradients = [gradient_magnitude(img) for img in images]
    gradients = np.stack(gradients, axis=-1)
    max_gradient_indices = np.argmax(gradients, axis=-1)

    fused_image = np.zeros_like(images[0])
    for i in range(fused_image.shape[0]):
        for j in range(fused_image.shape[1]):
            fused_image[i, j] = images[max_gradient_indices[i, j]][i, j]

    return fused_image

def main():
    parser = argparse.ArgumentParser(description='Multi-light fusion script.')
    parser.add_argument('mode', type=str, choices=['min', 'max', 'mean', 'gradient'], help='Fusion mode')
    parser.add_argument('images', nargs='+', help='Paths to input images')
    parser.add_argument('--output', type=str, default='fused_image.jpg', help='Output file name')

    args = parser.parse_args()

    images = load_images(args.images)

    if args.mode == 'min':
        fused_image = fuse_images_min(images)
    elif args.mode == 'max':
        fused_image = fuse_images_max(images)
    elif args.mode == 'mean':
        fused_image = fuse_images_mean(images)
    elif args.mode == 'gradient':
        fused_image = fuse_images_gradient(images)
    else:
        print(f"Invalid mode: {args.mode}")
        sys.exit(1)

    cv2.imwrite(args.output, fused_image)
    print(f"Fused image saved as {args.output}")

if __name__ == "__main__":
    main()
