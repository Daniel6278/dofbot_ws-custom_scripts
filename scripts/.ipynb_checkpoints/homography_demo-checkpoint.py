#!/usr/bin/env python3
"""
Demo script for homography estimation with robot arm positioning
This script shows how to use homography to relate camera views to robot arm positions
"""

import cv2
import numpy as np
import time
from homography_estimation import HomographyEstimator
from arm_controller import ArmController

class HomographyRobotDemo:
    def __init__(self):
        """Initialize the demo with homography estimator and arm controller"""
        self.estimator = HomographyEstimator(chessboard_size=(9, 6), square_size=25.0)
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
        
        # Define known robot positions for calibration
        self.calibration_positions = {
            'position1': {'base': 90, 'shoulder': 160, 'elbow': 20, 'wrist': 0},
            'position2': {'base': 120, 'shoulder': 160, 'elbow': 20, 'wrist': 0},
            'position3': {'base': 60, 'shoulder': 160, 'elbow': 20, 'wrist': 0},
            'position4': {'base': 90, 'shoulder': 140, 'elbow': 40, 'wrist': 0}
        }
    
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
        
    def calibrate_camera_to_robot(self):
        """
        Calibrate the relationship between camera view and robot arm positions
        This creates a homography between image coordinates and robot workspace coordinates
        """
        print("=== Camera to Robot Calibration ===")
        print("This will establish the relationship between camera view and robot workspace")
        
        # Take photos from different robot positions
        calibration_images = {}
        robot_positions = []
        
        for pos_name, angles in self.calibration_positions.items():
            print(f"\nMoving to {pos_name}...")
            
            # Convert descriptive angles to joint IDs
            joint_angles = self.convert_angles_to_joint_ids(angles)
            
            # Move robot to calibration position
            self.arm_controller.move_arm_only(joint_angles, 2000)
            time.sleep(2)
            
            # Take photo
            print(f"Taking photo from {pos_name}...")
            input("Press Enter when ready to capture...")
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_filename = f"calibration_{pos_name}_{timestamp}.jpg"
            result = self.estimator.take_chessboard_photo(image_filename)
            calibration_images[pos_name] = result
            
            # Store robot position (we'll use base angle as x-coordinate, shoulder as y-coordinate)
            robot_positions.append([angles['base'], angles['shoulder']])
        
        # Detect chessboard corners in all calibration images
        print("\nDetecting chessboard corners in calibration images...")
        image_corners = {}
        
        for pos_name, image_path in calibration_images.items():
            corners, _ = self.estimator.detect_chessboard_corners(image_path)
            # Use center of chessboard as reference point
            center_corner = np.mean(corners.reshape(-1, 2), axis=0)
            image_corners[pos_name] = center_corner
        
        # Create point correspondences for homography
        # Image coordinates (from camera)
        src_points = np.array([image_corners[pos] for pos in self.calibration_positions.keys()])
        
        # Robot workspace coordinates (normalized)
        dst_points = np.array(robot_positions)
        
        # Estimate homography from camera view to robot workspace
        print("Estimating homography from camera view to robot workspace...")
        self.camera_to_robot_homography = self.estimator.estimate_homography_dlt(src_points, dst_points)
        
        print("Camera to robot homography matrix:")
        print(self.camera_to_robot_homography)
        
        return True
    
    def image_point_to_robot_position(self, image_point):
        """
        Convert an image point to robot workspace coordinates using homography
        
        Args:
            image_point: [x, y] coordinates in image
            
        Returns:
            robot_pos: [base_angle, shoulder_angle] in robot workspace
        """
        # Convert to homogeneous coordinates
        point_homogeneous = np.array([image_point[0], image_point[1], 1])
        
        # Apply homography transformation
        robot_homogeneous = self.camera_to_robot_homography @ point_homogeneous
        
        # Convert back to 2D coordinates
        robot_pos = robot_homogeneous[:2] / robot_homogeneous[2]
        
        return robot_pos
    
    def move_to_image_point(self, image_point, approach_height=160):
        """
        Move robot arm to a position corresponding to an image point
        
        Args:
            image_point: [x, y] coordinates in image
            approach_height: Height to approach the target (shoulder angle)
        """
        # Convert image point to robot coordinates
        robot_pos = self.image_point_to_robot_position(image_point)
        
        print(f"Image point {image_point} -> Robot position {robot_pos}")
        
        # Move to the calculated position
        target_angles = {
            'base': int(robot_pos[0]),
            'shoulder': approach_height,
            'elbow': 20,
            'wrist': 0
        }
        
        # Convert to joint IDs
        joint_angles = self.convert_angles_to_joint_ids(target_angles)
        
        print(f"Moving to angles: {target_angles}")
        self.arm_controller.move_arm_only(joint_angles, 2000)
        
    def interactive_targeting_demo(self):
        """
        Interactive demo where user clicks on image and robot moves to corresponding position
        """
        print("\n=== Interactive Targeting Demo ===")
        print("Click on the image to move the robot to that position")
        print("Press 'q' to quit")
        
        # Take a current photo
        print("Taking current photo for targeting...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        current_image_path = self.estimator.take_chessboard_photo(f"targeting_image_{timestamp}.jpg")
        
        # Load and display the image
        image = cv2.imread(current_image_path)
        if image is None:
            print("Failed to load image")
            return
        
        # Create window and mouse callback
        window_name = "Click to target robot position"
        cv2.namedWindow(window_name)
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"\nClicked at image coordinates: ({x}, {y})")
                self.move_to_image_point([x, y])
        
        cv2.setMouseCallback(window_name, mouse_callback)
        
        # Display image and wait for clicks
        while True:
            cv2.imshow(window_name, image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def run_demo(self):
        """Run the complete homography demo"""
        try:
            print("=== Homography Robot Demo ===")
            print("This demo shows how to use homography to relate camera view to robot arm positions")
            
            # Step 1: Calibrate camera to robot relationship
            if not self.calibrate_camera_to_robot():
                print("Calibration failed!")
                return
            
            # Step 2: Interactive targeting demo
            self.interactive_targeting_demo()
            
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        except Exception as e:
            print(f"Error in demo: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Return to home position
            print("Returning to home position...")
            self.arm_controller.move_home(2000)
            self.arm_controller.set_torque(False)

def main():
    """Main function to run the homography demo"""
    demo = HomographyRobotDemo()
    demo.run_demo()

if __name__ == "__main__":
    main() 