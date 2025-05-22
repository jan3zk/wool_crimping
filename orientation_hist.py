import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from scipy.optimize import curve_fit


def gaussian_with_offset(x, amplitude, mean, stddev, offset):
    """Gaussian function with an offset."""
    return offset + amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def compute_orientation_histogram(image_path, edge_path, output_path, bin_size=10):
    # Load original image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Load edge map as binary image
    edges = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
    if edges is None:
        raise ValueError(f"Could not read edge map: {edge_path}")

    # Compute gradients in X and Y directions
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=7)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=7)

    # Compute gradient orientation (in degrees)
    angles = np.arctan2(grad_y, grad_x) * (180.0 / np.pi)
    angles = np.mod(angles, 180)  # Normalize to [0, 180)

    # Mask: keep angles only at edge pixels
    edge_mask = edges > 0
    edge_orientations = angles[edge_mask]

    # Compute histogram
    bins = np.arange(0, 181, bin_size)
    hist, bin_edges = np.histogram(edge_orientations, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit Gaussian to histogram
    try:
        amplitude_guess = np.max(hist) - np.min(hist)
        mean_guess = bin_centers[np.argmax(hist)]
        stddev_guess = 20
        offset_guess = np.min(hist)

        popt, _ = curve_fit(gaussian_with_offset, bin_centers, hist, 
                            p0=[amplitude_guess, mean_guess, stddev_guess, offset_guess])
        amplitude, mean, stddev, offset = popt
        print(f"Fitted Gaussian parameters:\n  Mean: {mean:.2f}°\n  Stddev: {stddev:.2f}°")
    except RuntimeError:
        print("Gaussian fit failed.")
        amplitude, mean, stddev = None, None, None

    # Plot histogram and fitted Gaussian
    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers, hist, width=bin_size, edgecolor='black', align='center', label='Histogram')
    if amplitude is not None:
        x_fit = np.linspace(0, 180, 1000)
        y_fit = gaussian_with_offset(x_fit, *popt)
        plt.plot(x_fit, y_fit, 'r--', label='Gaussian fit')
    plt.title('Histogram of Edge Orientations with Gaussian Fit')
    plt.xlabel('Orientation (degrees)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Histogram with Gaussian fit saved to: {output_path}")

    return mean, stddev

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute histogram of edge orientations from image and edge map, with Gaussian fit.")
    parser.add_argument("--image", required=True, help="Path to original grayscale image.")
    parser.add_argument("--edges", required=True, help="Path to binary edge map (from Canny).")
    parser.add_argument("--output", required=True, help="Output path to save histogram plot.")
    parser.add_argument("--bin_size", type=int, default=10, help="Orientation bin size in degrees (default: 10).")

    args = parser.parse_args()
    compute_orientation_histogram(args.image, args.edges, args.output, args.bin_size)
