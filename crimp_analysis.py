import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse


def analyze_fiber_orientation(image_path, save_dir=None, debug=False, method="canny", min_edge_length=0):
    """
    Analyze fiber orientation in mineral wool sample images
    
    Args:
        image_path: Path to the image file
        save_dir: Directory to save result images (optional)
        debug: Whether to save intermediate processing images for debugging
        method: Edge detection method to use ("canny" or "adaptive")
        min_edge_length: Minimum length of edges to keep (in pixels), edges shorter than this will be filtered out
    
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
    
    # Process image based on selected method
    if method == "canny":
        # Method 1: Canny-based approach (from crimp_orientation_detector.py)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "03_blurred.png"), blurred)
            
        # Edge detection using Canny
        v = np.median(blurred)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        edges = cv2.Canny(blurred, lower, upper)
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "04_canny_edges.png"), edges)
        
    else:  # method == "adaptive"
        # Method 2: Adaptive threshold approach
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "02b_enhanced.png"), enhanced)
        
        # Apply stronger Gaussian blur to reduce fine details (increased kernel size)
        blurred = cv2.GaussianBlur(enhanced, (9, 9), 0)
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "03_blurred.png"), blurred)
        
        # Use adaptive thresholding instead of Canny for better fiber detection
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 5
        )
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "04a_adaptive_threshold.png"), binary)
        
        # Apply morphological operations to connect broken fibers and remove noise
        kernel = np.ones((3, 3), np.uint8)
        # Opening (erosion followed by dilation) to remove small isolated points
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "04b_opened.png"), opened)
        
        # Closing (dilation followed by erosion) to connect broken fibers
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "04c_closed.png"), closed)
        
        # Use the binary mask instead of Canny edges for orientation analysis
        edges = closed
    
    # Filter out edges that are too short using connected component analysis
    if min_edge_length > 0:
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
        
        # Create a new edge image with only components larger than min_edge_length
        filtered_edges = np.zeros_like(edges)
        
        # Start from 1 to skip the background (label 0)
        for i in range(1, num_labels):
            # Get the size (area) of the current component
            size = stats[i, cv2.CC_STAT_AREA]
            
            # Keep only components larger than the minimum size
            if size >= min_edge_length:
                filtered_edges[labels == i] = 255
        
        # Save debug image for filtered edges
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, "04d_filtered_edges.png"), filtered_edges)
        
        # Update edges with filtered version
        edges = filtered_edges
        
    # Calculate gradients using Sobel with larger kernel for smoother gradients
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=7)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=7)
    
    # Calculate magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Apply minimum threshold on gradient magnitude to remove weak edges
    min_magnitude = np.percentile(magnitude, 70)  # Only keep top 30% of gradient magnitudes
    strong_gradients = magnitude > min_magnitude
    
    # Create initial mask from edges
    mask = edges > 0
    
    # Combine the binary mask and strong gradients for final mask
    mask = mask & strong_gradients
        
    if debug_dir:
        # Visualize the final mask
        final_mask_img = np.zeros_like(gray)
        final_mask_img[mask] = 255
        cv2.imwrite(os.path.join(debug_dir, "09_final_mask.png"), final_mask_img)
    
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
        # Create debug directory if it doesn't exist
        debug_dir = os.path.join(save_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save the masked orientation visualization
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_masked_orientation.png"), orientation_vis)
        
        # Save the edge overlay as debug image too
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_edge_overlay.png"), overlay_img)
    
    # Also save the histogram data as CSV
    np.savetxt(os.path.join(save_dir, f"{base_name}_histogram.csv"), 
               np.column_stack((np.arange(180), histogram)),
               delimiter=',', header='Angle,Count', comments='')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze fiber orientation in mineral wool samples')
    parser.add_argument('--input', required=True, help='Input image path')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--debug', action='store_true', help='Save intermediate processing images')
    parser.add_argument('--method', choices=['canny', 'adaptive'], default='canny', 
                        help='Edge detection method: "canny" (basic Canny-based) or "adaptive" (morphological operations)')
    parser.add_argument('--min-edge-length', type=int, default=0,
                        help='Minimum length of edges to keep (in pixels). Edges shorter than this will be filtered out.')
    
    args = parser.parse_args()
    
    # Process a single image
    analyze_fiber_orientation(args.input, args.output, debug=args.debug, method=args.method, min_edge_length=args.min_edge_length)