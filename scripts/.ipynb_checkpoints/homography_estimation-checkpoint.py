#!/usr/bin/env python3
"""
Homography Estimation using DLT Algorithm
Based on OpenCV tutorial: https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html

This script demonstrates:
1. Taking photos using the robot's camera
2. Detecting chessboard corners for homography estimation
3. Computing homography using DLT algorithm
4. Applying homography transformations
"""

import cv2
import numpy as np
import argparse
import time
import os
from arm_controller import ArmController
from Arm_Lib import Arm_Device

class HomographyEstimator:
    def __init__(self, chessboard_size=(6, 9), square_size=25.0):
        """
        Initialize homography estimator
        
        Args:
            chessboard_size: Tuple of (width, height) for chessboard pattern
            square_size: Size of chessboard square in mm
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # Generate object points for chessboard
        self.object_points = self._generate_object_points()
        
    def _generate_object_points(self):
        """Generate 3D object points for chessboard"""
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
    def take_chessboard_photo(self, filename=None):
        """
        Take a photo of a chessboard pattern using Windows-compatible camera access
        
        Args:
            filename: Optional filename to save the image (will be prefixed with timestamp)
            
        Returns:
            filename: Name of saved image file
        """
        # Generate timestamp for unique naming
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if filename is None:
            filename = f"chessboard_{timestamp}.jpg"
        else:
            # Add timestamp prefix to provided filename
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{timestamp}{ext}"
            
        print(f"Taking photo of chessboard pattern...")
        print("Make sure the chessboard is clearly visible in the camera view")
        input("Press Enter when ready to capture...")
        
        result = self._take_photo_windows(filename)
        if result:
            print(f"Photo saved as: {result}")
            return result
        else:
            raise RuntimeError("Failed to take photo")
    
    def _take_photo_windows(self, filename):
        """
        Windows-compatible photo capture function
        Returns the filename if successful, None if failed
        """
        try:
            # Try different camera indices (0, 1, 2, etc.)
            camera_index = 0
            max_attempts = 5
            
            for attempt in range(max_attempts):
                print(f"Attempting to open camera {camera_index}...")
                
                # Try to open camera
                camera = cv2.VideoCapture(camera_index)
                
                if camera.isOpened():
                    print(f"Successfully opened camera {camera_index}")
                    
                    # Set camera properties for better quality
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    
                    # Read a frame
                    print("Reading frame...")
                    ret, image = camera.read()
                    
                    if ret and image is not None:
                        print(f"Successfully captured a frame with shape: {image.shape}")
                        
                        # Save photo with provided filename
                        cv2.imwrite(filename, image)
                        
                        # Clean up
                        camera.release()
                        cv2.destroyAllWindows()
                        
                        print("Photo captured successfully.")
                        return filename
                    else:
                        print(f"Failed to read from camera {camera_index}")
                        camera.release()
                else:
                    print(f"Failed to open camera {camera_index}")
                
                camera_index += 1
            
            # If we get here, no camera worked
            raise RuntimeError("No working camera found after trying multiple indices")
            
        except Exception as e:
            print(f"Error taking photo: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def detect_chessboard_corners(self, image_path):
        """
        Detect chessboard corners in an image with enhanced preprocessing
        
        Args:
            image_path: Path to the image file
            
        Returns:
            corners: Detected corner points
            image: Loaded image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhanced preprocessing for better corner detection
        processed_gray = self._preprocess_image_for_chessboard(gray)
        
        # Try different chessboard sizes if the main one fails
        chessboard_sizes_to_try = [
            self.chessboard_size,
            (self.chessboard_size[1], self.chessboard_size[0]),  # Try swapped dimensions
            (self.chessboard_size[0] - 1, self.chessboard_size[1] - 1),  # Try smaller size
            (self.chessboard_size[0] + 1, self.chessboard_size[1] + 1),  # Try larger size
        ]
        
        corners = None
        best_size = None
        
        for size in chessboard_sizes_to_try:
            print(f"Trying chessboard size: {size}")
            
            # Find chessboard corners
            ret, detected_corners = cv2.findChessboardCorners(processed_gray, size, None)
            
            if ret:
                print(f"Successfully detected corners with size {size}")
                corners = detected_corners
                best_size = size
                break
        
        if corners is None:
            # If still no corners found, try with original grayscale image
            print("Trying with original grayscale image...")
            for size in chessboard_sizes_to_try:
                ret, detected_corners = cv2.findChessboardCorners(gray, size, None)
                if ret:
                    print(f"Successfully detected corners with original image, size {size}")
                    corners = detected_corners
                    best_size = size
                    break
        
        if corners is None:
            # Save diagnostic images for debugging
            cv2.imwrite("debug_original_gray.jpg", gray)
            cv2.imwrite("debug_processed_gray.jpg", processed_gray)
            raise RuntimeError(f"Could not find chessboard corners in {image_path}. Debug images saved.")
            
        # Refine corner detection with subpixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        print(f"Successfully detected {len(corners)} corners with size {best_size}")
        return corners, image
    
    def _preprocess_image_for_chessboard(self, gray_image):
        """
        Enhanced preprocessing for better chessboard corner detection
        Based on techniques from OpenCV forums and tutorials
        """
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray_image)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Apply adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def estimate_homography_dlt(self, src_points, dst_points):
        """
        Estimate homography using Direct Linear Transform (DLT) algorithm
        
        Args:
            src_points: Source points (Nx2 array)
            dst_points: Destination points (Nx2 array)
            
        Returns:
            H: Homography matrix (3x3)
        """
        if len(src_points) < 4:
            raise ValueError("Need at least 4 point correspondences for homography")
            
        # Normalize points for better numerical stability
        src_normalized, T1 = self._normalize_points(src_points)
        dst_normalized, T2 = self._normalize_points(dst_points)
        
        # Build the A matrix for Ah = 0
        A = []
        for i in range(len(src_normalized)):
            x, y = src_normalized[i]
            u, v = dst_normalized[i]
            
            A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
            A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
        
        A = np.array(A)
        
        # Solve using SVD
        U, S, Vt = np.linalg.svd(A)
        H_normalized = Vt[-1].reshape(3, 3)
        
        # Denormalize
        H = np.linalg.inv(T2) @ H_normalized @ T1
        
        # Normalize homography matrix
        H = H / H[2, 2]
        
        return H
    
    def _normalize_points(self, points):
        """
        Normalize points using Hartley's normalization
        
        Args:
            points: Nx2 array of points
            
        Returns:
            normalized_points: Normalized points
            T: Transformation matrix
        """
        points = np.array(points, dtype=np.float64)
        centroid = np.mean(points, axis=0)
        
        # Calculate mean distance from centroid
        distances = np.sqrt(np.sum((points - centroid) ** 2, axis=1))
        mean_distance = np.mean(distances)
        
        # Scale factor
        scale = np.sqrt(2) / mean_distance
        
        # Create transformation matrix
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])
        
        # Apply transformation
        homogeneous_points = np.column_stack([points, np.ones(len(points))])
        normalized_points = (T @ homogeneous_points.T).T[:, :2]
        
        return normalized_points, T
    
    def apply_homography(self, image, H, output_size=None):
        """
        Apply homography transformation to an image
        
        Args:
            image: Input image
            H: Homography matrix
            output_size: Output image size (width, height)
            
        Returns:
            warped_image: Transformed image
        """
        if output_size is None:
            output_size = (image.shape[1], image.shape[0])
            
        warped_image = cv2.warpPerspective(image, H, output_size)
        return warped_image
    
    def visualize_corners(self, image, corners, window_name="Chessboard Corners"):
        """Visualize detected chessboard corners"""
        # Draw corners
        cv2.drawChessboardCorners(image, self.chessboard_size, corners, True)
        
        # Display image
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def estimate_camera_pose_from_homography(self, H, camera_matrix, dist_coeffs=None):
        """
        Estimate camera pose from homography matrix
        Based on the OpenCV tutorial method
        
        Args:
            H: Homography matrix
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients (optional)
            
        Returns:
            rvec: Rotation vector
            tvec: Translation vector
        """
        # Normalize homography
        norm = np.sqrt(H[0, 0]**2 + H[1, 0]**2 + H[2, 0]**2)
        H_normalized = H / norm
        
        # Extract columns
        h1 = H_normalized[:, 0]
        h2 = H_normalized[:, 1]
        h3 = H_normalized[:, 2]
        
        # Compute rotation matrix
        r1 = h1
        r2 = h2
        r3 = np.cross(r1, r2)
        t = h3
        
        # Ensure orthogonality using SVD
        R = np.column_stack([r1, r2, r3])
        U, S, Vt = np.linalg.svd(R)
        R = U @ Vt
        
        # Ensure proper rotation matrix (det = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt
        
        # Convert to rotation vector
        rvec, _ = cv2.Rodrigues(R)
        
        return rvec, t

def main():
    parser = argparse.ArgumentParser(description='Homography estimation using DLT algorithm')
    parser.add_argument('--chessboard-size', nargs=2, type=int, default=[6, 9],
                       help='Chessboard pattern size (width height)')
    parser.add_argument('--square-size', type=float, default=25.0,
                       help='Chessboard square size in mm')
    parser.add_argument('--image1', type=str, help='Path to first image (optional)')
    parser.add_argument('--image2', type=str, help='Path to second image (optional)')
    parser.add_argument('--camera-matrix', type=str, help='Path to camera calibration file')
    parser.add_argument('--visualize', action='store_true', help='Show visualization windows')
    parser.add_argument('--take-photos', action='store_true', help='Take new photos using camera')
    
    args = parser.parse_args()
    
    # Initialize homography estimator
    estimator = HomographyEstimator(
        chessboard_size=tuple(args.chessboard_size),
        square_size=args.square_size
    )
    
    try:
        # Take photos if requested
        if args.take_photos:
            print("Taking first chessboard photo...")
            image1_path = estimator.take_chessboard_photo("chessboard1.jpg")
            
            input("Move the camera or chessboard to a different position, then press Enter...")
            
            print("Taking second chessboard photo...")
            image2_path = estimator.take_chessboard_photo("chessboard2.jpg")
        else:
            # Use provided image paths
            if not args.image1 or not args.image2:
                raise ValueError("Must provide --image1 and --image2 or use --take-photos")
            image1_path = args.image1
            image2_path = args.image2
        
        # Detect corners in both images
        print("Detecting corners in first image...")
        corners1, image1 = estimator.detect_chessboard_corners(image1_path)
        
        print("Detecting corners in second image...")
        corners2, image2 = estimator.detect_chessboard_corners(image2_path)
        
        # Visualize corners if requested
        if args.visualize:
            estimator.visualize_corners(image1.copy(), corners1, "Image 1 Corners")
            estimator.visualize_corners(image2.copy(), corners2, "Image 2 Corners")
        
        # Extract point correspondences
        src_points = corners1.reshape(-1, 2)
        dst_points = corners2.reshape(-1, 2)
        
        print(f"Using {len(src_points)} point correspondences for homography estimation")
        
        # Estimate homography using DLT
        print("Estimating homography using DLT algorithm...")
        H = estimator.estimate_homography_dlt(src_points, dst_points)
        
        print("Estimated homography matrix:")
        print(H)
        
        # Apply homography transformation
        print("Applying homography transformation...")
        warped_image = estimator.apply_homography(image1, H, (image2.shape[1], image2.shape[0]))
        
        # Save results
        cv2.imwrite("warped_image.jpg", warped_image)
        cv2.imwrite("image2_reference.jpg", image2)
        
        print("Results saved:")
        print("- warped_image.jpg: Image 1 transformed to Image 2's perspective")
        print("- image2_reference.jpg: Original Image 2 for comparison")
        
        # Estimate camera pose if camera matrix is provided
        if args.camera_matrix:
            print("Loading camera calibration...")
            fs = cv2.FileStorage(args.camera_matrix, cv2.FILE_STORAGE_READ)
            camera_matrix = fs.getNode("camera_matrix").mat()
            dist_coeffs = fs.getNode("distortion_coefficients").mat()
            fs.release()
            
            print("Estimating camera pose from homography...")
            rvec, tvec = estimator.estimate_camera_pose_from_homography(H, camera_matrix, dist_coeffs)
            
            print("Camera pose:")
            print(f"Rotation vector: {rvec.flatten()}")
            print(f"Translation vector: {tvec.flatten()}")
        
        # Calculate reprojection error
        print("Calculating reprojection error...")
        src_homogeneous = np.column_stack([src_points, np.ones(len(src_points))])
        predicted_dst = (H @ src_homogeneous.T).T
        predicted_dst = predicted_dst[:, :2] / predicted_dst[:, 2:]
        
        error = np.sqrt(np.mean(np.sum((predicted_dst - dst_points) ** 2, axis=1)))
        print(f"Mean reprojection error: {error:.2f} pixels")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 