import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from PIL import Image
import argparse

def compute_anisotropy_tensor(sub_image, alpha=0):
    """
    Compute the anisotropy tensor for a sub-image using the regularized correlation texture tensor 
    method described in the paper.
    """
    # Compute the 2D Fourier transform
    f_tilde = fftshift(fft2(sub_image))
    f_tilde_abs_squared = np.abs(f_tilde)**2
    
    # Get coordinates centered at zero
    ny, nx = sub_image.shape
    kx = np.fft.fftfreq(nx, d=1.0)
    ky = np.fft.fftfreq(ny, d=1.0)
    
    # Shift frequencies to match fftshift
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)
    
    # Create meshgrid
    kx, ky = np.meshgrid(kx, ky)
    
    # Compute |k|^2
    k_squared = kx**2 + ky**2
    # Avoid division by zero at DC component
    k_squared[ny//2, nx//2] = 1e-10
    
    # Initialize tensor
    tensor = np.zeros((2, 2))
    
    # Compute tensor components as per equation (1) in the paper
    # Iterate over all frequency components
    for i in range(ny):
        for j in range(nx):
            if k_squared[i, j] > 0:  # Skip DC
                weight = f_tilde_abs_squared[i, j] / (k_squared[i, j]**alpha)
                k_outer = np.array([[kx[i, j]**2, kx[i, j]*ky[i, j]], 
                                    [kx[i, j]*ky[i, j], ky[i, j]**2]])
                tensor += weight * k_outer
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    
    # The principal direction corresponds to the largest eigenvalue
    idx = np.argmax(eigenvalues)
    v = eigenvectors[:, idx]
    
    # Calculate anisotropy amplitude using the deviatoric part of tensor
    trace = np.trace(tensor)
    if trace > 0:
        A = tensor / trace - 0.5 * np.eye(2)
        anisotropy_amplitude = np.sqrt(np.sum(A**2))
    else:
        anisotropy_amplitude = 0.0
    
    # Orientation is the angle of the principal direction
    # IMPORTANT FIX: Add 90-degree rotation to align with fiber direction
    # rather than perpendicular to fibers
    orientation = np.arctan2(v[1], v[0]) + np.pi/2
    
    return tensor, anisotropy_amplitude, orientation, eigenvalues, v

def create_anisotropy_quiver_plot(image_path, output_path, zoi_size=32, skip=1, alpha=0, debug=False):
    """
    Create a quiver plot showing the anisotropy field superimposed on the original image.
    """
    # Load the image and convert to grayscale
    img = np.array(Image.open(image_path).convert('L')).astype(float)
    
    # Image dimensions
    height, width = img.shape
    print(f"Image dimensions: {width}x{height}")
    
    # Calculate number of ZOIs
    n_zoi_y = height // zoi_size
    n_zoi_x = width // zoi_size
    print(f"Number of ZOIs: {n_zoi_x}x{n_zoi_y}")
    
    # Initialize arrays for quiver plot
    X, Y = np.meshgrid(np.arange(n_zoi_x) * zoi_size + zoi_size/2, 
                       np.arange(n_zoi_y) * zoi_size + zoi_size/2)
    U = np.zeros_like(X, dtype=float)
    V = np.zeros_like(Y, dtype=float)
    A = np.zeros_like(X, dtype=float)  # Anisotropy amplitude
    
    # Process each ZOI
    orientations = []  # Store orientations for debugging
    
    for i in range(n_zoi_y):
        for j in range(n_zoi_x):
            # Extract the sub-image
            y_start = i * zoi_size
            y_end = (i + 1) * zoi_size
            x_start = j * zoi_size
            x_end = (j + 1) * zoi_size
            
            if y_end <= height and x_end <= width:
                sub_image = img[y_start:y_end, x_start:x_end]
                
                # Compute anisotropy
                tensor, anisotropy, orientation, eigenvalues, eigenvector = compute_anisotropy_tensor(sub_image, alpha)
                
                # Debug output for a few ZOIs
                if debug and i % 5 == 0 and j % 5 == 0:
                    print(f"ZOI ({i},{j}):")
                    print(f"  Tensor: {tensor}")
                    print(f"  Eigenvalues: {eigenvalues}")
                    print(f"  Eigenvector: {eigenvector}")
                    print(f"  Orientation: {orientation * 180/np.pi:.2f} degrees")
                    print(f"  Anisotropy: {anisotropy:.4f}")
                
                # Store results (make vectors point along fiber direction)
                U[i, j] = np.cos(orientation)
                V[i, j] = np.sin(orientation)
                A[i, j] = anisotropy
                orientations.append(orientation * 180/np.pi)  # Store in degrees
    
    # Debug: Check distribution of orientations
    if debug:
        orientations = np.array(orientations)
        print(f"Orientation stats: min={orientations.min():.2f}°, max={orientations.max():.2f}°, "
              f"mean={orientations.mean():.2f}°, std={orientations.std():.2f}°")
        # Plot histogram of orientations
        plt.figure(figsize=(8, 6))
        plt.hist(orientations, bins=36, range=(-180, 180))
        plt.title('Histogram of Orientations')
        plt.xlabel('Orientation (degrees)')
        plt.ylabel('Count')
        plt.savefig('orientation_histogram.png')
        plt.close()
    
    # Create the figure
    plt.figure(figsize=(12, 10))
    
    # Show the original image
    plt.imshow(img, cmap='gray')
    
    # Apply skip to reduce density of arrows if needed
    X_skip = X[::skip, ::skip]
    Y_skip = Y[::skip, ::skip]
    U_skip = U[::skip, ::skip]
    V_skip = V[::skip, ::skip]
    A_skip = A[::skip, ::skip]
    
    # Define arrow length - scale by anisotropy to emphasize strong orientations
    arrow_length = zoi_size * 0.5
    
    # Make arrows longer when anisotropy is higher
    arrow_lengths = arrow_length * (0.5 + A_skip)
    
    # Normalize direction vectors
    norm = np.sqrt(U_skip**2 + V_skip**2)
    norm[norm == 0] = 1.0  # Avoid division by zero
    
    U_scaled = U_skip / norm * arrow_lengths
    V_scaled = V_skip / norm * arrow_lengths
    
    # Draw arrows
    quiver = plt.quiver(X_skip, Y_skip, U_scaled, V_scaled, 
                      A_skip,  # Color by anisotropy
                      cmap='Reds',
                      angles='xy',  
                      scale=1,
                      scale_units='xy',
                      units='xy',
                      width=zoi_size/20,
                      headwidth=3,
                      headlength=5, 
                      headaxislength=4.5)
    
    # Add a colorbar for anisotropy
    cbar = plt.colorbar(quiver)
    cbar.set_label('Anisotropy Amplitude')
    
    plt.title('Anisotropy Field of the Studied Specimen')
    plt.xlabel('x (pixel)')
    plt.ylabel('y (pixel)')
    plt.tight_layout()
    
    # Save the figure to the output path
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Anisotropy quiver plot saved to {output_path}")
    
    # Close the figure to free memory
    plt.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Create an anisotropy quiver plot for a mineral wool image')
    parser.add_argument('input_image', help='Path to the input mineral wool image')
    parser.add_argument('--output', '-o', help='Path to save the output quiver plot (default: anisotropy_quiver.png)',
                        default='anisotropy_quiver.png')
    parser.add_argument('--zoi_size', '-z', type=int, default=32, help='Size of zones of interest (default: 32)')
    parser.add_argument('--skip', '-s', type=int, default=1, help='Show only every skip\'th arrow (default: 1)')
    parser.add_argument('--alpha', '-a', type=float, default=0, help='Regularization parameter alpha (default: 0)')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Create the quiver plot
    create_anisotropy_quiver_plot(args.input_image, args.output, args.zoi_size, args.skip, args.alpha, args.debug)

if __name__ == "__main__":
    main()