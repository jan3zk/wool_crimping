import cv2
import argparse

def undistort_image(image_path: str, output_path: str, calib_path: str = 'calibration.yaml') -> None:
    """
    Undistort an image using camera calibration data and save the result to a file.

    Parameters:
    - image_path: str, path to the input image.
    - output_path: str, path where the undistorted image will be saved.
    - calib_path: str, path to the calibration YAML file (default: 'calibration.yaml').
    """
    # Load calibration data
    fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    fs.release()

    # Load input image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Undistort
    h, w = img.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_mtx)

    # Save result
    cv2.imwrite(output_path, undistorted)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Undistort an image using calibration.yaml")
    parser.add_argument('image_path', help='Path to the image to undistort')
    parser.add_argument('--output', required=True, help='Path to save the undistorted image')
    parser.add_argument('--calib', default='calibration.yaml', help='Path to the calibration YAML file')
    args = parser.parse_args()

    # Call function
    undistort_image(args.image_path, args.output, args.calib)
