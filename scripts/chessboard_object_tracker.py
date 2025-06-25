#!/usr/bin/env python3
"""
Chessboard Object Tracker
This script detects objects on a chessboard, marks their center of mass,
and stores their locations relative to the chessboard coordinates.
"""

import cv2
import numpy as np
import time
import os
import argparse
import json
from homography_estimation import HomographyEstimator
from arm_controller import ArmController

class ChessboardObjectTracker:
    def __init__(self, chessboard_size=(9, 6), square_size=25.0):
        """
        Initialize the chessboard object tracker
        
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
        
        # Object detection parameters
        self.object_detection_params = {
            'min_area': 100,        # Minimum object area in pixels
            'max_area': 10000,      # Maximum object area in pixels
            'color_lower': np.array([0, 50, 50]),    # HSV lower bound for object detection
            'color_upper': np.array([180, 255, 255]), # HSV upper bound for object detection
            'contour_threshold': 0.7  # Threshold for contour filtering
        }
        
        # Chessboard coordinate system
        self.chessboard_coords = {
            'width_squares': chessboard_size[0] - 1,  # Number of squares horizontally
            'height_squares': chessboard_size[1] - 1, # Number of squares vertically
            'square_size_mm': square_size
        }
        
        # Detected objects storage
        self.detected_objects = []
        
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
    
    def detect_objects_on_chessboard(self, image_path):
        """
        Detect objects on the chessboard and correct perspective
        
        Args:
            image_path: Path to the current photo
            
        Returns:
            corrected_image: Perspective-corrected image
            objects_info: List of detected objects with their positions
            detection_success: Whether detection was successful
        """
        print("Detecting objects on chessboard...")
        
        try:
            # Detect corners in current image
            current_corners, current_image = self.estimator.detect_chessboard_corners(image_path)
            
            # Get the bounding box of detected corners
            detected_corners = current_corners.reshape(-1, 2)
            min_x, min_y = np.min(detected_corners, axis=0)
            max_x, max_y = np.max(detected_corners, axis=0)
            
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
            
            # Create ideal square corners
            ideal_size = max(board_width, board_height)
            ideal_corners = np.array([
                [0, 0],                    # Top-left
                [ideal_size, 0],           # Top-right
                [ideal_size, ideal_size],  # Bottom-right
                [0, ideal_size]            # Bottom-left
            ], dtype=np.float32)
            
            # Estimate homography from detected to ideal
            current_homography = self.estimator.estimate_homography_dlt(
                detected_outer_corners, ideal_corners
            )
            
            # Apply homography correction
            print("Applying homography correction...")
            corrected_image = self.estimator.apply_homography(
                current_image, current_homography, (int(ideal_size), int(ideal_size))
            )
            
            # Transform all chessboard corners using homography
            print("Transforming chessboard corners...")
            transformed_corners = []
            for corner in current_corners.reshape(-1, 2):
                corner_homogeneous = np.array([corner[0], corner[1], 1])
                transformed_homogeneous = current_homography @ corner_homogeneous
                transformed_corner = transformed_homogeneous[:2] / transformed_homogeneous[2]
                transformed_corners.append(transformed_corner)
            
            transformed_corners = np.array(transformed_corners, dtype=np.float32).reshape(-1, 1, 2)
            
            # Detect objects in the corrected image
            objects_info = self._detect_objects_in_image(corrected_image)
            
            # Convert object positions to chessboard coordinates
            for obj in objects_info:
                obj['chessboard_coords'] = self._pixel_to_chessboard_coords(
                    obj['center'], ideal_size
                )
            
            # Save corrected image with objects and chessboard corners
            debug_corrected = corrected_image.copy()
            
            # Draw all chessboard corners
            cv2.drawChessboardCorners(debug_corrected, self.estimator.chessboard_size, transformed_corners, True)
            
            # Draw detected objects
            for i, obj in enumerate(objects_info):
                center = obj['center']
                # Draw center of mass
                cv2.circle(debug_corrected, (int(center[0]), int(center[1])), 8, (0, 255, 255), -1)
                cv2.putText(debug_corrected, f"Obj{i+1}", (int(center[0])+10, int(center[1])+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Draw bounding box
                if 'bbox' in obj:
                    x, y, w, h = obj['bbox']
                    cv2.rectangle(debug_corrected, (x, y), (x+w, y+h), (255, 0, 255), 2)
            
            # Generate timestamp for file naming
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"objects_on_chessboard_{timestamp}.jpg", debug_corrected)
            print(f"Objects on chessboard image saved as: objects_on_chessboard_{timestamp}.jpg")
            
            # Save original image with detection markers
            debug_original = current_image.copy()
            cv2.drawChessboardCorners(debug_original, self.estimator.chessboard_size, current_corners, True)
            
            # Transform object positions back to original image coordinates
            for obj in objects_info:
                # Transform center back to original coordinates
                center_homogeneous = np.array([obj['center'][0], obj['center'][1], 1])
                inv_homography = np.linalg.inv(current_homography)
                original_center_homogeneous = inv_homography @ center_homogeneous
                original_center = original_center_homogeneous[:2] / original_center_homogeneous[2]
                
                # Draw on original image
                cv2.circle(debug_original, (int(original_center[0]), int(original_center[1])), 12, (0, 255, 255), -1)
            
            cv2.imwrite(f"objects_original_{timestamp}.jpg", debug_original)
            print(f"Objects on original image saved as: objects_original_{timestamp}.jpg")
            
            return corrected_image, objects_info, True
                
        except Exception as e:
            print(f"Failed to detect objects: {str(e)}")
            return None, None, False
    
    def _detect_objects_in_image(self, image):
        """
        Detect objects in the corrected chessboard image
        
        Args:
            image: Corrected chessboard image
            
        Returns:
            objects_info: List of detected objects
        """
        objects_info = []
        
        # Convert to HSV for better color-based detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for object detection (excluding chessboard pattern)
        # We'll use a simple approach: detect non-black, non-white regions
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create mask for potential objects (not too dark, not too light)
        _, thresh = cv2.threshold(gray, 30, 200, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.object_detection_params['min_area'] or area > self.object_detection_params['max_area']:
                continue
            
            # Calculate center of mass
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate additional properties
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Filter by circularity (objects should be somewhat circular)
            if circularity < self.object_detection_params['contour_threshold']:
                continue
            
            # Check if object is within chessboard bounds (with some margin)
            margin = 20
            if (cx < margin or cx > image.shape[1] - margin or 
                cy < margin or cy > image.shape[0] - margin):
                continue
            
            # Store object information
            obj_info = {
                'id': i,
                'center': (cx, cy),
                'area': area,
                'bbox': (x, y, w, h),
                'circularity': circularity,
                'perimeter': perimeter
            }
            
            objects_info.append(obj_info)
        
        print(f"Detected {len(objects_info)} objects")
        return objects_info
    
    def _pixel_to_chessboard_coords(self, pixel_coords, image_size):
        """
        Convert pixel coordinates to chessboard coordinates
        
        Args:
            pixel_coords: (x, y) pixel coordinates
            image_size: Size of the corrected image
            
        Returns:
            chessboard_coords: (square_x, square_y) chessboard coordinates
        """
        x, y = pixel_coords
        
        # Normalize coordinates to 0-1 range
        norm_x = x / image_size
        norm_y = y / image_size
        
        # Convert to chessboard square coordinates
        square_x = norm_x * self.chessboard_coords['width_squares']
        square_y = norm_y * self.chessboard_coords['height_squares']
        
        # Convert to physical coordinates (mm)
        mm_x = square_x * self.chessboard_coords['square_size_mm']
        mm_y = square_y * self.chessboard_coords['square_size_mm']
        
        return {
            'square_coords': (square_x, square_y),
            'mm_coords': (mm_x, mm_y),
            'pixel_coords': pixel_coords
        }
    
    def save_object_data(self, filename="detected_objects.json"):
        """
        Save detected object data to JSON file
        
        Args:
            filename: Name of the JSON file to save (will be prefixed with timestamp)
        """
        # Generate timestamp for unique naming
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        timestamped_filename = f"{name}_{timestamp}{ext}"
        
        data = {
            'timestamp': time.time(),
            'chessboard_info': self.chessboard_coords,
            'objects': self.detected_objects
        }
        
        with open(timestamped_filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Object data saved to: {timestamped_filename}")
    
    def run_object_tracking(self, calibration_image_path="chessboardCalibration.jpg"):
        """
        Run complete object tracking cycle
        
        Args:
            calibration_image_path: Path to calibration image
        """
        try:
            print("=== Chessboard Object Tracking ===")
            
            # Step 1: Move to optimal viewing position
            print("\nStep 1: Positioning robot...")
            self.move_to_optimal_view()
            
            # Step 2: Take current photo
            print("\nStep 2: Taking photo...")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            current_photo = self.take_current_photo(f"current_chessboard_{timestamp}.jpg")
            
            # Step 3: Detect objects on chessboard
            print("\nStep 3: Detecting objects...")
            corrected_image, objects_info, success = self.detect_objects_on_chessboard(current_photo)
            
            if success:
                print(f"✓ Object detection completed successfully!")
                print(f"Found {len(objects_info)} objects")
                
                # Store detected objects
                self.detected_objects = objects_info
                
                # Print object information
                print("\n=== Detected Objects ===")
                for i, obj in enumerate(objects_info):
                    print(f"Object {i+1}:")
                    print(f"  Center: {obj['center']}")
                    print(f"  Area: {obj['area']:.1f} pixels")
                    print(f"  Chessboard coords: {obj['chessboard_coords']['square_coords']}")
                    print(f"  Physical coords: {obj['chessboard_coords']['mm_coords']} mm")
                
                # Step 4: Save object data
                print("\nStep 4: Saving object data...")
                self.save_object_data()
                
                print("\n=== Tracking Summary ===")
                print(f"✓ {len(objects_info)} objects detected and tracked")
                print(f"✓ Object positions saved to JSON file")
                print(f"✓ Images saved with object markers")
                
                return True, objects_info
            else:
                print("✗ Failed to detect objects")
                return False, None
                
        except Exception as e:
            print(f"Error in object tracking: {str(e)}")
            import traceback
            traceback.print_exc()
            return False, None
        finally:
            # Return to home position
            print("\nReturning to home position...")
            self.arm_controller.move_home(2000)
            self.arm_controller.set_torque(False)

def main():
    """Main function to run object tracking"""
    parser = argparse.ArgumentParser(description='Object tracking on chessboard')
    parser.add_argument('--chessboard-size', nargs=2, type=int, default=[9, 6],
                       help='Chessboard pattern size (width height)')
    parser.add_argument('--square-size', type=float, default=25.0,
                       help='Chessboard square size in mm')
    parser.add_argument('--min-area', type=int, default=100,
                       help='Minimum object area in pixels')
    parser.add_argument('--max-area', type=int, default=10000,
                       help='Maximum object area in pixels')
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = ChessboardObjectTracker(
        chessboard_size=tuple(args.chessboard_size),
        square_size=args.square_size
    )
    
    # Update detection parameters
    tracker.object_detection_params['min_area'] = args.min_area
    tracker.object_detection_params['max_area'] = args.max_area
    
    # Run object tracking
    success, objects_info = tracker.run_object_tracking()
    
    if success:
        print("\n✓ Object tracking completed successfully!")
    else:
        print("\n✗ Object tracking failed!")

if __name__ == "__main__":
    main()
