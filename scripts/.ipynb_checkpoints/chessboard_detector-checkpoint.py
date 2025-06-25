#!/usr/bin/env python3
"""
Chessboard Detection and Robot Positioning using Homography
This script automatically positions the robot arm to view a chessboard,
takes a photo, and uses homography to determine the board's position.
"""

import cv2
import numpy as np
import time
import os
import argparse
from homography_estimation import HomographyEstimator
from arm_controller import ArmController

class ChessboardDetector:
    def __init__(self, chessboard_size=(9, 6), square_size=25.0):
        """
        Initialize the chessboard detector
        
        Args:
            chessboard_size: Tuple of (width, height) for chessboard pattern
            square_size: Size of chessboard square in mm
        """
        self.estimator = HomographyEstimator(chessboard_size, square_size)
        self.arm_controller = ArmController()
        
        # Joint mapping: descriptive names to joint IDs
        self.joint_mapping = {
            'base': 1,      # Base rotation
            'shoulder': 2,  # Arm bend low servo
            'elbow': 3,     # Arm bend med servo
            'wrist': 4,     # Arm bend high servo
            'wrist_rot': 5, # Wrist rotation
            'gripper': 6    # Gripper
        }
        
        # Optimal viewing angles for chessboard detection
        self.optimal_view_angles = {
            'base': 90,
            'shoulder': 100,
            'elbow': 10,
            'wrist': 0,
            'wrist_rot': 90,
            'gripper': 5
        }
        
        # Homography matrix from calibration (will be loaded)
        self.calibration_homography = None
        
    def convert_angles_to_joint_ids(self, angles_dict):
        """
        Convert descriptive angle dictionary to joint ID dictionary
        
        Args:
            angles_dict: Dictionary with descriptive keys like 'base', 'shoulder', etc.
            
        Returns:
            joint_dict: Dictionary with integer keys (1-6) for joint IDs
        """
        joint_dict = {}
        for joint_name, angle in angles_dict.items():
            if joint_name in self.joint_mapping:
                joint_id = self.joint_mapping[joint_name]
                joint_dict[joint_id] = angle
        return joint_dict
    
    def load_calibration_homography(self, calibration_image_path):
        """
        Load and establish homography from calibration image
        
        Args:
            calibration_image_path: Path to the calibration chessboard image
        """
        print(f"Loading calibration from: {calibration_image_path}")
        
        if not os.path.exists(calibration_image_path):
            raise FileNotFoundError(f"Calibration image not found: {calibration_image_path}")
        
        # Detect corners in calibration image
        print("Detecting corners in calibration image...")
        calibration_corners, calibration_image = self.estimator.detect_chessboard_corners(calibration_image_path)
        
        # Use the detected corners to create homography
        # We'll use the outer corners of the detected chessboard
        detected_corners = calibration_corners.reshape(-1, 2)
        
        # Find the bounding box of detected corners
        min_x, min_y = np.min(detected_corners, axis=0)
        max_x, max_y = np.max(detected_corners, axis=0)
        
        # Calculate the size of the detected chessboard
        board_width = max_x - min_x
        board_height = max_y - min_y
        
        # Create detected corner points (outer corners)
        detected_outer_corners = np.array([
            [min_x, min_y],  # Top-left
            [max_x, min_y],  # Top-right
            [max_x, max_y],  # Bottom-right
            [min_x, max_y]   # Bottom-left
        ], dtype=np.float32)
        
        # Create ideal square corners - use the larger dimension to make it square
        # This ensures we get a proper perspective correction
        ideal_size = max(board_width, board_height)
        ideal_corners = np.array([
            [0, 0],                    # Top-left
            [ideal_size, 0],           # Top-right
            [ideal_size, ideal_size],  # Bottom-right
            [0, ideal_size]            # Bottom-left
        ], dtype=np.float32)
        
        # Estimate homography from detected to ideal
        print("Estimating homography from detected to ideal chessboard...")
        self.calibration_homography = self.estimator.estimate_homography_dlt(
            detected_outer_corners, ideal_corners
        )
        
        print("Calibration homography matrix:")
        print(self.calibration_homography)
        
        # Save calibration info for debugging with corner markers
        debug_image = calibration_image.copy()
        # Draw the detected outer corners
        for i, corner in enumerate(detected_outer_corners):
            cv2.circle(debug_image, (int(corner[0]), int(corner[1])), 10, (0, 255, 0), -1)
            cv2.putText(debug_image, str(i+1), (int(corner[0])+15, int(corner[1])+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw the bounding rectangle
        cv2.rectangle(debug_image, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (255, 0, 0), 2)
        
        # Generate timestamp for file naming
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"calibration_debug_{timestamp}.jpg", debug_image)
        print(f"Calibration debug image saved as: calibration_debug_{timestamp}.jpg")
        
        return True
    
    def move_to_optimal_view(self):
        """
        Move robot arm to optimal viewing position for chessboard detection
        """
        print("Moving to optimal viewing position...")
        
        # Separate arm angles (joints 1-4) from wrist/gripper angles (joints 5-6)
        arm_angles = {
            'base': self.optimal_view_angles['base'],
            'shoulder': self.optimal_view_angles['shoulder'],
            'elbow': self.optimal_view_angles['elbow'],
            'wrist': self.optimal_view_angles['wrist']
        }
        
        wrist_gripper_angles = {
            'wrist_rot': self.optimal_view_angles['wrist_rot'],
            'gripper': self.optimal_view_angles['gripper']
        }
        
        # Convert arm angles to joint IDs
        joint_angles = self.convert_angles_to_joint_ids(arm_angles)
        
        # Move arm joints first
        print("Moving arm joints...")
        self.arm_controller.move_arm_only(joint_angles, 2000)
        time.sleep(1)
        
        # Move wrist and gripper separately
        print("Moving wrist and gripper...")
        self.arm_controller.move_wrist_gripper(
            wrist_angle=wrist_gripper_angles['wrist_rot'],
            gripper_angle=wrist_gripper_angles['gripper'],
            time_ms=1000
        )
        time.sleep(1)
        
        print("Robot positioned for chessboard detection")
    
    def take_current_photo(self, filename="current_chessboard.jpg"):
        """
        Take a photo of the current view
        
        Args:
            filename: Name of the file to save the photo (will be prefixed with timestamp)
            
        Returns:
            filename: Path to the saved photo
        """
        print("Taking current photo...")
        
        # Generate timestamp for unique naming
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        timestamped_filename = f"{name}_{timestamp}{ext}"
        
        # Take photo using the estimator's method
        result = self.estimator.take_chessboard_photo(timestamped_filename)
        
        if result:
            print(f"Photo saved as: {result}")
            return result
        else:
            raise RuntimeError("Failed to take photo")
    
    def verify_homography_correction(self, original_corners, corrected_corners, homography_matrix):
        """
        Verify that the homography correction is working properly
        
        Args:
            original_corners: Original detected corners
            corrected_corners: Ideal target corners
            homography_matrix: The homography matrix used
        """
        print("Verifying homography correction...")
        
        # Transform original corners using homography
        transformed_corners = []
        for corner in original_corners:
            corner_homogeneous = np.array([corner[0], corner[1], 1])
            transformed_homogeneous = homography_matrix @ corner_homogeneous
            transformed_corner = transformed_homogeneous[:2] / transformed_homogeneous[2]
            transformed_corners.append(transformed_corner)
        
        transformed_corners = np.array(transformed_corners)
        
        # Calculate error between transformed and ideal corners
        error = np.mean(np.linalg.norm(transformed_corners - corrected_corners, axis=1))
        print(f"Homography correction error: {error:.2f} pixels")
        
        # Check if the transformation is reasonable
        if error > 10.0:  # More than 10 pixels average error
            print("Warning: High homography error - correction may not be working properly")
            return False
        else:
            print("✓ Homography correction verified")
            return True
    
    def detect_chessboard_in_current_view(self, current_image_path):
        """
        Detect chessboard in the current view and correct perspective
        
        Args:
            current_image_path: Path to the current photo
            
        Returns:
            corrected_image: Perspective-corrected image
            board_center: Center coordinates of the detected board
            detection_success: Whether detection was successful
        """
        print("Detecting chessboard in current view...")
        
        try:
            # Detect corners in current image
            current_corners, current_image = self.estimator.detect_chessboard_corners(current_image_path)
            
            # Get the bounding box of detected corners
            detected_corners = current_corners.reshape(-1, 2)
            min_x, min_y = np.min(detected_corners, axis=0)
            max_x, max_y = np.max(detected_corners, axis=0)
            
            # Calculate board center
            board_center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
            
            # Create detected outer corners
            detected_outer_corners = np.array([
                [min_x, min_y],  # Top-left
                [max_x, min_y],  # Top-right
                [max_x, max_y],  # Bottom-right
                [min_x, max_y]   # Bottom-left
            ], dtype=np.float32)
            
            # Calculate the size of the detected chessboard
            board_width = max_x - min_x
            board_height = max_y - min_y
            
            # Create ideal square corners - use the larger dimension to make it square
            # This ensures we get a proper perspective correction
            ideal_size = max(board_width, board_height)
            ideal_corners = np.array([
                [0, 0],                    # Top-left
                [ideal_size, 0],           # Top-right
                [ideal_size, ideal_size],  # Bottom-right
                [0, ideal_size]            # Bottom-left
            ], dtype=np.float32)
            
            # Estimate homography from detected to ideal for current image
            current_homography = self.estimator.estimate_homography_dlt(
                detected_outer_corners, ideal_corners
            )
            
            # Verify the homography correction
            self.verify_homography_correction(detected_outer_corners, ideal_corners, current_homography)
            
            # Apply homography correction
            print("Applying homography correction...")
            corrected_image = self.estimator.apply_homography(
                current_image, current_homography, (int(ideal_size), int(ideal_size))
            )
            
            # Transform all chessboard corners using homography
            print("Transforming all chessboard corners...")
            transformed_corners = []
            for corner in current_corners.reshape(-1, 2):
                corner_homogeneous = np.array([corner[0], corner[1], 1])
                transformed_homogeneous = current_homography @ corner_homogeneous
                transformed_corner = transformed_homogeneous[:2] / transformed_homogeneous[2]
                transformed_corners.append(transformed_corner)
            
            transformed_corners = np.array(transformed_corners, dtype=np.float32).reshape(-1, 1, 2)
            
            # Transform board center using homography
            center_homogeneous = np.array([board_center[0], board_center[1], 1])
            corrected_center_homogeneous = current_homography @ center_homogeneous
            corrected_center = corrected_center_homogeneous[:2] / corrected_center_homogeneous[2]
            
            # Save corrected image with corner markers
            debug_corrected = corrected_image.copy()
            
            # Draw all chessboard corners using OpenCV's drawChessboardCorners
            cv2.drawChessboardCorners(debug_corrected, self.estimator.chessboard_size, transformed_corners, True)
            
            # Draw the center point
            cv2.circle(debug_corrected, (int(corrected_center[0]), int(corrected_center[1])), 8, (255, 0, 0), -1)
            cv2.putText(debug_corrected, "Center", (int(corrected_center[0])+10, int(corrected_center[1])+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Generate timestamp for file naming
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"corrected_chessboard_{timestamp}.jpg", debug_corrected)
            print(f"Corrected image saved as: corrected_chessboard_{timestamp}.jpg")
            
            # Also save the original image with detection markers
            debug_original = current_image.copy()
            
            # Draw all chessboard corners on original image
            cv2.drawChessboardCorners(debug_original, self.estimator.chessboard_size, current_corners, True)
            
            # Draw the detected outer corners
            for i, corner in enumerate(detected_outer_corners):
                cv2.circle(debug_original, (int(corner[0]), int(corner[1])), 10, (0, 255, 0), -1)
                cv2.putText(debug_original, str(i+1), (int(corner[0])+15, int(corner[1])+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw the bounding rectangle
            cv2.rectangle(debug_original, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (255, 0, 0), 2)
            
            # Draw the center point
            cv2.circle(debug_original, (int(board_center[0]), int(board_center[1])), 12, (255, 0, 0), -1)
            cv2.putText(debug_original, "Center", (int(board_center[0])+15, int(board_center[1])+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imwrite(f"detection_markers_{timestamp}.jpg", debug_original)
            print(f"Detection markers image saved as: detection_markers_{timestamp}.jpg")
            
            # Save additional debug info
            print(f"Original board dimensions: {board_width:.1f} x {board_height:.1f}")
            print(f"Ideal square size: {ideal_size:.1f}")
            print(f"Homography matrix determinant: {np.linalg.det(current_homography):.6f}")
            print(f"Number of corners detected: {len(current_corners)}")
            
            return corrected_image, corrected_center, True
                
        except Exception as e:
            print(f"Failed to detect chessboard: {str(e)}")
            return None, None, False
    
    def calculate_board_position_in_robot_space(self, board_center_image):
        """
        Calculate the board position in robot workspace coordinates
        
        Args:
            board_center_image: Board center coordinates in image space
            
        Returns:
            robot_position: Board position in robot workspace coordinates
        """
        if self.calibration_homography is None:
            print("Warning: No calibration homography available")
            return None
        
        # Convert image coordinates to robot workspace coordinates
        # This would require additional calibration between image and robot space
        # For now, we'll return the image coordinates as a placeholder
        
        print(f"Board detected at image coordinates: {board_center_image}")
        print("Note: Full robot space calibration would require additional setup")
        
        return board_center_image
    
    def run_detection_cycle(self, calibration_image_path="chessboardCalibration.jpg"):
        """
        Run complete chessboard detection cycle
        
        Args:
            calibration_image_path: Path to calibration image
        """
        try:
            print("=== Chessboard Detection Cycle ===")
            
            # Step 1: Load calibration homography
            print("\nStep 1: Loading calibration...")
            self.load_calibration_homography(calibration_image_path)
            
            # Step 2: Move to optimal viewing position
            print("\nStep 2: Positioning robot...")
            self.move_to_optimal_view()
            
            # Step 3: Take current photo
            print("\nStep 3: Taking photo...")
            current_photo = self.take_current_photo("current_chessboard.jpg")
            
            # Step 4: Detect chessboard and correct perspective
            print("\nStep 4: Detecting chessboard...")
            corrected_image, board_center, success = self.detect_chessboard_in_current_view(current_photo)
            
            if success:
                print(f"✓ Chessboard detected successfully!")
                print(f"Board center: {board_center}")
                
                # Step 5: Calculate position in robot space
                print("\nStep 5: Calculating robot space position...")
                robot_position = self.calculate_board_position_in_robot_space(board_center)
                
                # Step 6: Save results
                print("\nStep 6: Saving results...")
                cv2.imwrite("detection_result.jpg", corrected_image)
                print("Detection result saved as: detection_result.jpg")
                
                # Print summary
                print("\n=== Detection Summary ===")
                print(f"✓ Chessboard detected at image coordinates: {board_center}")
                print(f"✓ Perspective correction applied")
                print(f"✓ Results saved to files")
                
                return True, board_center, corrected_image
            else:
                print("✗ Failed to detect chessboard")
                return False, None, None
                
        except Exception as e:
            print(f"Error in detection cycle: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, None, None
        finally:
            # Return to home position
            print("\nReturning to home position...")
            self.arm_controller.move_home(2000)
            self.arm_controller.set_torque(False)

def main():
    """Main function to run chessboard detection"""
    parser = argparse.ArgumentParser(description='Chessboard detection using homography')
    parser.add_argument('--calibration-image', type=str, default='chessboardCalibration.jpg',
                       help='Path to calibration chessboard image')
    parser.add_argument('--chessboard-size', nargs=2, type=int, default=[9, 6],
                       help='Chessboard pattern size (width height)')
    parser.add_argument('--square-size', type=float, default=25.0,
                       help='Chessboard square size in mm')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ChessboardDetector(
        chessboard_size=tuple(args.chessboard_size),
        square_size=args.square_size
    )
    
    # Run detection cycle
    success, board_center, corrected_image = detector.run_detection_cycle(args.calibration_image)
    
    if success:
        print("\n✓ Detection cycle completed successfully!")
    else:
        print("\n✗ Detection cycle failed!")

if __name__ == "__main__":
    main() 