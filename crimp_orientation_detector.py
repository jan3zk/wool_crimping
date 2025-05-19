import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob
import argparse


def analyze_fiber_orientation(image_path, save_dir=None, debug=False):
    """
    Analyze fiber orientation in mineral wool sample images
    
    Args:
        image_path: Path to the image file
        save_dir: Directory to save result images (optional)
        debug: Whether to save intermediate processing images for debugging
    
    Returns:
        histogram: Array of orientation counts (180 bins, 1-degree resolution)
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    
    # Create debug directory if needed
    debug_dir = None
    if debug and save_dir:
        base_name = Path(image_path).stem
        debug_dir = os.path.join(save_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save the original image
        cv2.imwrite(os.path.join(debug_dir, "01_original.png"), img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "02_grayscale.png"), gray)
    
    # Apply Gaussian blur to reduce noise
    #blurred = cv2.GaussianBlur(gray, (3, 3), 1)
    blurred = cv2.bilateralFilter(gray, d=5, sigmaColor=75, sigmaSpace=75)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "03_blurred.png"), blurred)
    #blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    
    # Edge detection using Canny
    v = np.median(blurred)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(blurred, lower, upper)
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "04_canny_edges.png"), edges)
    
    # Calculate gradients using Sobel
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    
    # Save debug images for Sobel gradients (normalize to 0-255 for visualization)
    if debug_dir:
        # Normalize sobelx for visualization
        sobelx_normalized = cv2.normalize(sobelx, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(os.path.join(debug_dir, "05_sobel_x.png"), sobelx_normalized)
        
        # Normalize sobely for visualization
        sobely_normalized = cv2.normalize(sobely, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(os.path.join(debug_dir, "06_sobel_y.png"), sobely_normalized)
    
    # Calculate magnitude and orientation
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    orientation = np.arctan2(sobely, sobelx) * (180 / np.pi) + 90  # +90 to get 0-180 range
    
    if debug_dir:
        # Save magnitude image (normalized to 0-255)
        magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(os.path.join(debug_dir, "07_gradient_magnitude.png"), magnitude_normalized)
        
        # Save orientation visualization (using HSV colormap)
        orientation_vis = np.zeros((*orientation.shape, 3), dtype=np.uint8)
        for i in range(orientation.shape[0]):
            for j in range(orientation.shape[1]):
                angle = orientation[i, j] % 180
                normalized_angle = angle / 180.0
                hsv = np.array([normalized_angle * 179, 255, 255], dtype=np.uint8)
                rgb = cv2.cvtColor(hsv.reshape(1, 1, 3), cv2.COLOR_HSV2BGR)[0, 0]
                orientation_vis[i, j] = rgb
        
        cv2.imwrite(os.path.join(debug_dir, "08_orientation_full.png"), orientation_vis)
    
    # Create a mask for significant edges (using Canny output)
    mask = edges > 0
    
    # Filter orientations by the mask
    valid_orientations = orientation[mask]
    
    # Convert to 0-179 range
    valid_orientations = valid_orientations % 180
    
    # Create histogram of orientations (1-degree bins)
    hist, bins = np.histogram(valid_orientations, bins=180, range=(0, 180))
    
    if save_dir:
        save_results(img, edges, orientation, mask, hist, image_path, save_dir, debug)
    
    return hist


def save_results(img, edges, orientation, mask, histogram, image_path, save_dir, debug=False):
    """Save visualization of results"""
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get base filename without extension
    base_name = Path(image_path).stem
    
    # Create figure with subplots and specify proper spacing
    fig = plt.figure(figsize=(16, 10))
    
    # Original image
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original Image')
    ax1.set_axis_off()
    
    # Edge detection result
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(edges, cmap='gray')
    ax2.set_title('Edge Detection')
    ax2.set_axis_off()
    
    # Orientation map (masked to show only edge pixels)
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Create a blank RGB image for visualization
    orientation_vis = np.zeros((*orientation.shape, 3), dtype=np.uint8)
    
    # We'll use HSV color space for angle representation
    # For each edge pixel, set a color based on its orientation
    for i in range(orientation.shape[0]):
        for j in range(orientation.shape[1]):
            if mask[i, j]:
                # Get angle and normalize to 0-1 range for color mapping
                angle = orientation[i, j] % 180
                normalized_angle = angle / 180.0
                
                # Convert HSV to RGB (hue=angle, full saturation and value)
                rgb = plt.cm.hsv(normalized_angle)[:3]
                
                # Set the pixel color (scale to 0-255 for uint8)
                orientation_vis[i, j] = (np.array(rgb) * 255).astype(np.uint8)
    
    im = ax3.imshow(orientation_vis)
    ax3.set_title('Orientation Map (at edges)')
    ax3.set_axis_off()
    
    # Add colorbar properly to the right of the orientation map
    norm = plt.Normalize(0, 180)
    sm = plt.cm.ScalarMappable(cmap='hsv', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3, orientation='vertical', ticks=[0, 45, 90, 135, 180], shrink=0.6)
    cbar.set_label('Angle (degrees)')
    
    # Histogram of orientations
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.bar(np.arange(180), histogram, width=1)
    ax4.set_xlabel('Orientation (degrees)')
    ax4.set_ylabel('Count')
    ax4.set_title('Orientation Histogram (1° resolution)')
    ax4.set_xlim(0, 180)
    ax4.grid(alpha=0.3)
    
    # Adjust layout with plt.tight_layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(save_dir, f"{base_name}_analysis.png"), dpi=300)
    plt.close()
    
    # Create and save the edge overlay image
    # Create a green overlay for edges
    green_edges = np.zeros_like(img)
    green_edges[mask] = [0, 255, 0]  # Green edges in BGR format for OpenCV
    
    # Blend the original image with the green edges
    alpha = 0.7  # Original image weight
    beta = 0.3   # Edge overlay weight
    overlay_img = cv2.addWeighted(img.copy(), alpha, green_edges, beta, 0)
    
    # Save the overlay image using OpenCV
    overlay_img_path = os.path.join(save_dir, f"{base_name}_edge_overlay.png")
    cv2.imwrite(overlay_img_path, overlay_img)
    
    if debug:
        # Save the masked orientation visualization
        debug_dir = os.path.join(save_dir, "debug", base_name)
        cv2.imwrite(os.path.join(debug_dir, "09_masked_orientation.png"), orientation_vis)
        
        # Save the edge overlay as debug image too
        cv2.imwrite(os.path.join(debug_dir, "10_edge_overlay.png"), overlay_img)
    
    # Also save the histogram data as CSV
    np.savetxt(os.path.join(save_dir, f"{base_name}_histogram.csv"), 
               np.column_stack((np.arange(180), histogram)),
               delimiter=',', header='Angle,Count', comments='')


def batch_process(input_dir, output_dir, pattern="*.jpg", debug=False):
    """
    Process all images matching the pattern in the input directory
    
    Args:
        input_dir: Directory containing the images
        output_dir: Directory to save the results
        pattern: Glob pattern to match image files
        debug: Whether to save intermediate processing images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all matching image files
    image_files = glob.glob(os.path.join(input_dir, pattern))
    
    if not image_files:
        print(f"No images found matching pattern {pattern} in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    all_histograms = {}
    for i, image_path in enumerate(image_files):
        print(f"Processing image {i+1}/{len(image_files)}: {image_path}")
        hist = analyze_fiber_orientation(image_path, output_dir, debug=debug)
        if hist is not None:
            all_histograms[os.path.basename(image_path)] = hist
    
    # Save combined results
    combined_output = os.path.join(output_dir, "combined_results.csv")
    with open(combined_output, 'w') as f:
        # Write header
        header = "Image," + ",".join([str(i) for i in range(180)])
        f.write(header + "\n")
        
        # Write data for each image
        for image_name, hist in all_histograms.items():
            line = image_name + "," + ",".join([str(count) for count in hist])
            f.write(line + "\n")
    
    print(f"Combined results saved to {combined_output}")


if __name__ == "__main__":
    # Example usage
    # For single image
    # analyze_fiber_orientation("path/to/image.jpg", "results")
    
    # For batch processing
    # batch_process("./images", "./results", "*.jpg")
    
    parser = argparse.ArgumentParser(description='Analyze fiber orientation in mineral wool samples')
    parser.add_argument('--input', required=True, help='Input image path or directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--pattern', default="*.jpg", help='File pattern for batch processing')
    parser.add_argument('--batch', action='store_true', help='Enable batch processing')
    parser.add_argument('--debug', action='store_true', help='Save intermediate processing images')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_process(args.input, args.output, args.pattern, debug=args.debug)
    else:
        analyze_fiber_orientation(args.input, args.output, debug=args.debug)