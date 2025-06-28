#!/usr/bin/env python3
"""
Object Detection on Chessboard Area
This script positions the robot arm to look down at the chessboard area
and detects objects using computer vision techniques, placing bounding boxes
and center of mass points around detected objects.

# Run full detection cycle (position robot, take photo, detect objects)
python object_detection.py

# Analyze existing image
python object_detection.py --image path/to/image.jpg

# Adjust detection parameters
python object_detection.py --min-area 300 --max-area 30000
"""

import cv2
import numpy as np
import time
import os
import argparse
from arm_controller import ArmController

class ObjectDetector:
    def __init__(self):
        """
        Initialize the object detector
        """
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
        
        # Optimal viewing angles for object detection (looking down at chessboard area)
        self.optimal_view_angles = {
            'base': 90,
            'shoulder': 100,
            'elbow': 10,
            'wrist': 0,
            'wrist_rot': 90,
            'gripper': 5
        }
        
        # Object detection parameters (optimized for square blocks)
        self.min_object_area = 1000  # Minimum area for square block detection
        self.max_object_area = 30000  # Maximum area for square block detection
        self.contour_approximation_epsilon = 0.02  # Contour approximation parameter
        
        # Color ranges for square blocks (red, green, yellow, blue) - optimized HSV ranges
        self.color_ranges = {
            'red': [
                (np.array([0, 120, 70]), np.array([10, 255, 255])),      # Lower red range
                (np.array([170, 120, 70]), np.array([180, 255, 255]))    # Upper red range
            ],
            'green': [
                (np.array([35, 80, 80]), np.array([85, 255, 255])),      # Broader green range
                (np.array([40, 60, 60]), np.array([80, 255, 255])),      # Alternative green range
                (np.array([45, 100, 100]), np.array([75, 255, 255]))     # Narrower green range
            ],
            'blue': [
                (np.array([100, 80, 80]), np.array([130, 255, 255])),    # Broader blue range
                (np.array([110, 60, 60]), np.array([130, 255, 255])),    # Alternative blue range
                (np.array([100, 100, 100]), np.array([120, 255, 255]))   # Narrower blue range
            ],
            'yellow': [(np.array([20, 100, 100]), np.array([30, 255, 255]))]
        }
        
        # Expected square properties
        self.expected_square_ratio_range = (0.7, 1.3)  # Width/height ratio for squares
        self.min_square_vertices = 4  # Minimum vertices for square detection
        self.max_square_vertices = 8  # Maximum vertices (allowing for some approximation)
        
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
        Move robot arm to optimal viewing position for object detection
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
        
        print("Robot positioned for object detection")
    
    def take_photo(self, filename="object_detection.jpg"):
        """
        Take a photo using multiple methods (direct camera access and ROS camera topics)
        
        Args:
            filename: Name of the file to save the photo (will be prefixed with timestamp)
            
        Returns:
            filename: Path to the saved photo
        """
        print("Taking photo for object detection...")
        
        # Generate timestamp for unique naming
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(filename)
        timestamped_filename = f"{name}_{timestamp}{ext}"
        
        # Try direct camera access first
        print("Method 1: Trying direct camera access...")
        result = self._take_photo_direct(timestamped_filename)
        
        if result:
            print(f"Photo saved as: {result}")
            return result
        
        # Try ROS camera topic as fallback
        print("Method 2: Trying ROS camera topic...")
        result = self._take_photo_ros(timestamped_filename)
        
        if result:
            print(f"Photo saved as: {result}")
            return result
        
        # If both methods fail, raise error
        raise RuntimeError("Failed to take photo with both direct camera access and ROS camera topic")
    
    def _take_photo_direct(self, filename):
        """
        Robust camera access method for Jetson systems
        """
        camera = None
        try:
            print("Initializing camera...")
            
            # Try different camera backends and indices
            camera_configs = [
                (0, cv2.CAP_V4L2),      # V4L2 backend with index 0
                (0, cv2.CAP_ANY),       # Any backend with index 0
                (1, cv2.CAP_V4L2),      # V4L2 backend with index 1
                (1, cv2.CAP_ANY),       # Any backend with index 1
            ]
            
            for camera_index, backend in camera_configs:
                print(f"Trying camera index {camera_index} with backend {backend}")
                
                # Create camera object with specific backend
                camera = cv2.VideoCapture(camera_index, backend)
                
                if camera.isOpened():
                    print(f"Successfully opened camera with index {camera_index}")
                    
                    # Set camera properties for better compatibility
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Wait for camera to stabilize
                    print("Waiting for camera to stabilize...")
                    time.sleep(3)
                    
                    # Try to read multiple frames to ensure camera is working
                    successful_frames = 0
                    max_attempts = 10
                    
                    for attempt in range(max_attempts):
                        ret, image = camera.read()
                        
                        if ret and image is not None and image.size > 0:
                            print(f"Successfully captured frame {attempt + 1} with shape: {image.shape}")
                            successful_frames += 1
                            
                            # If we get a good frame, save it
                            if successful_frames >= 2:  # Require at least 2 good frames
                                cv2.imwrite(filename, image)
                                print("Photo captured successfully")
                                return filename
                        else:
                            print(f"Failed to read frame {attempt + 1}")
                            time.sleep(0.5)  # Wait before next attempt
                    
                    if successful_frames == 0:
                        print(f"Camera index {camera_index} opened but failed to read any frames")
                    else:
                        print(f"Camera index {camera_index} captured {successful_frames} frames but not enough for reliable capture")
                
                # Clean up and try next configuration
                if camera is not None:
                    camera.release()
                    camera = None
            
            # If we get here, no camera configuration worked
            print("All camera configurations failed")
            return None
            
        except Exception as e:
            print(f"Camera access error: {str(e)}")
            return None
        finally:
            # Ensure camera is released
            if camera is not None:
                camera.release()
                cv2.destroyAllWindows()
    
    def _take_photo_ros(self, filename):
        """
        Take photo using ROS camera topic (fallback method)
        """
        try:
            import rospy
            from sensor_msgs.msg import Image
            from cv_bridge import CvBridge
            
            print("Initializing ROS camera capture...")
            
            # Initialize ROS node if not already done
            try:
                rospy.init_node('object_detection_camera', anonymous=True)
            except:
                pass  # Node might already be initialized
            
            # Create CV bridge
            bridge = CvBridge()
            
            # Wait for camera topic to be available
            print("Waiting for camera topic...")
            try:
                # Try common camera topics
                camera_topics = [
                    '/usb_cam/image_raw',
                    '/camera/image_raw',
                    '/camera/color/image_raw',
                    '/camera/rgb/image_raw'
                ]
                
                image_msg = None
                for topic in camera_topics:
                    try:
                        print(f"Trying camera topic: {topic}")
                        image_msg = rospy.wait_for_message(topic, Image, timeout=5.0)
                        print(f"Successfully received image from {topic}")
                        break
                    except rospy.ROSException:
                        print(f"Topic {topic} not available")
                        continue
                
                if image_msg is None:
                    print("No camera topics available")
                    return None
                
                # Convert ROS message to OpenCV image
                cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
                
                if cv_image is not None and cv_image.size > 0:
                    print(f"Successfully converted ROS image with shape: {cv_image.shape}")
                    cv2.imwrite(filename, cv_image)
                    print("Photo captured successfully via ROS")
                    return filename
                else:
                    print("Failed to convert ROS image")
                    return None
                    
            except Exception as e:
                print(f"ROS camera error: {str(e)}")
                return None
                
        except ImportError:
            print("ROS or cv_bridge not available, skipping ROS camera method")
            return None
        except Exception as e:
            print(f"ROS camera access error: {str(e)}")
            return None
    
    def preprocess_image(self, image):
        """
        Preprocess image for object detection
        
        Args:
            image: Input image
            
        Returns:
            processed_image: Preprocessed image
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        return blurred
    
    def detect_objects_by_color(self, image, hsv_image):
        """
        Detect square blocks by color using HSV thresholds with RGB backup
        
        Args:
            image: Original BGR image
            hsv_image: HSV image for color detection
            
        Returns:
            detected_objects: List of detected square blocks with properties
        """
        detected_objects = []
        
        for color_name, color_ranges in self.color_ranges.items():
            # Create mask for each color range
            color_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
            
            for lower, upper in color_ranges:
                mask = cv2.inRange(hsv_image, lower, upper)
                color_mask = cv2.bitwise_or(color_mask, mask)
            
            # Find contours in the color mask
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Calculate contour area
                area = cv2.contourArea(contour)
                
                # Filter by area
                if self.min_object_area < area < self.max_object_area:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if it's approximately square (aspect ratio check)
                    aspect_ratio = w / h if h > 0 else 0
                    if not (self.expected_square_ratio_range[0] <= aspect_ratio <= self.expected_square_ratio_range[1]):
                        continue  # Skip non-square objects
                    
                    # Calculate center of mass (centroid)
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = x + w//2, y + h//2
                    
                    # Approximate contour to get shape information
                    epsilon = self.contour_approximation_epsilon * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Check if it has the right number of vertices for a square
                    num_vertices = len(approx)
                    if not (self.min_square_vertices <= num_vertices <= self.max_square_vertices):
                        continue  # Skip objects that don't have square-like vertices
                    
                    # Additional square validation: check if the contour is reasonably convex
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    # Square blocks should be mostly convex (solidity > 0.8)
                    if solidity < 0.8:
                        continue
                    
                    detected_objects.append({
                        'color': color_name,
                        'shape': 'square',
                        'area': area,
                        'bbox': (x, y, w, h),
                        'center': (cx, cy),
                        'contour': contour,
                        'approx_contour': approx,
                        'aspect_ratio': aspect_ratio,
                        'vertices': num_vertices,
                        'solidity': solidity
                    })
        
        return detected_objects
    
    def detect_objects_by_rgb_color(self, image):
        """
        Detect square blocks using RGB color space as backup for better blue/green detection
        
        Args:
            image: Original BGR image
            
        Returns:
            detected_objects: List of detected square blocks with RGB color detection
        """
        detected_objects = []
        
        # Convert BGR to RGB for easier color analysis
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Define RGB color ranges for blue and green (more robust)
        rgb_color_ranges = {
            'blue': [
                # Blue should have high blue component and low red/green
                (np.array([0, 0, 100]), np.array([100, 100, 255])),      # High blue, low red/green
                (np.array([0, 0, 80]), np.array([120, 120, 255])),       # Broader blue range
            ],
            'green': [
                # Green should have high green component and low red/blue
                (np.array([0, 100, 0]), np.array([100, 255, 100])),      # High green, low red/blue
                (np.array([0, 80, 0]), np.array([120, 255, 120])),       # Broader green range
            ]
        }
        
        for color_name, color_ranges in rgb_color_ranges.items():
            for lower, upper in color_ranges:
                # Create mask for RGB range
                mask = cv2.inRange(rgb_image, lower, upper)
                
                # Find contours in the mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Calculate contour area
                    area = cv2.contourArea(contour)
                    
                    # Filter by area
                    if self.min_object_area < area < self.max_object_area:
                        # Get bounding rectangle
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Check if it's approximately square (aspect ratio check)
                        aspect_ratio = w / h if h > 0 else 0
                        if not (self.expected_square_ratio_range[0] <= aspect_ratio <= self.expected_square_ratio_range[1]):
                            continue  # Skip non-square objects
                        
                        # Calculate center of mass (centroid)
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                        else:
                            cx, cy = x + w//2, y + h//2
                        
                        # Approximate contour to get shape information
                        epsilon = self.contour_approximation_epsilon * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        
                        # Check if it has the right number of vertices for a square
                        num_vertices = len(approx)
                        if not (self.min_square_vertices <= num_vertices <= self.max_square_vertices):
                            continue  # Skip objects that don't have square-like vertices
                        
                        # Additional square validation: check if the contour is reasonably convex
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        solidity = area / hull_area if hull_area > 0 else 0
                        
                        # Square blocks should be mostly convex (solidity > 0.8)
                        if solidity < 0.8:
                            continue
                        
                        detected_objects.append({
                            'color': color_name,
                            'shape': 'square',
                            'area': area,
                            'bbox': (x, y, w, h),
                            'center': (cx, cy),
                            'contour': contour,
                            'approx_contour': approx,
                            'aspect_ratio': aspect_ratio,
                            'vertices': num_vertices,
                            'solidity': solidity
                        })
        
        return detected_objects
    
    def detect_objects_by_edges(self, image):
        """
        Detect square blocks using edge detection (for blocks that might not be detected by color)
        
        Args:
            image: Input image
            
        Returns:
            detected_objects: List of detected square blocks
        """
        detected_objects = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # Filter by area
            if self.min_object_area < area < self.max_object_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if it's approximately square (aspect ratio check)
                aspect_ratio = w / h if h > 0 else 0
                if not (self.expected_square_ratio_range[0] <= aspect_ratio <= self.expected_square_ratio_range[1]):
                    continue  # Skip non-square objects
                
                # Calculate center of mass
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2
                
                # Approximate contour
                epsilon = self.contour_approximation_epsilon * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it has the right number of vertices for a square
                num_vertices = len(approx)
                if not (self.min_square_vertices <= num_vertices <= self.max_square_vertices):
                    continue  # Skip objects that don't have square-like vertices
                
                # Additional square validation: check if the contour is reasonably convex
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Square blocks should be mostly convex (solidity > 0.8)
                if solidity < 0.8:
                    continue
                
                detected_objects.append({
                    'color': 'unknown',  # Color not detected by edge method
                    'shape': 'square',
                    'area': area,
                    'bbox': (x, y, w, h),
                    'center': (cx, cy),
                    'contour': contour,
                    'approx_contour': approx,
                    'aspect_ratio': aspect_ratio,
                    'vertices': num_vertices,
                    'solidity': solidity
                })
        
        return detected_objects
    
    def draw_detection_results(self, image, detected_objects):
        """
        Draw bounding boxes and center points on the image for square blocks
        
        Args:
            image: Input image
            detected_objects: List of detected square blocks
            
        Returns:
            annotated_image: Image with detection annotations
        """
        annotated_image = image.copy()
        
        # Color mapping for square blocks (red, green, yellow, blue)
        color_map = {
            'red': (0, 0, 255),      # BGR format
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'unknown': (128, 128, 128)  # Gray for unknown color
        }
        
        for i, obj in enumerate(detected_objects):
            x, y, w, h = obj['bbox']
            cx, cy = obj['center']
            color = obj['color']
            area = obj['area']
            aspect_ratio = obj['aspect_ratio']
            vertices = obj['vertices']
            solidity = obj['solidity']
            
            # Get color for drawing
            draw_color = color_map.get(color, color_map['unknown'])
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), draw_color, 2)
            
            # Draw center point
            cv2.circle(annotated_image, (cx, cy), 5, draw_color, -1)
            cv2.circle(annotated_image, (cx, cy), 8, (255, 255, 255), 2)
            
            # Draw contour
            cv2.drawContours(annotated_image, [obj['contour']], -1, draw_color, 1)
            
            # Add label with square information
            label = f"{color}_square_{i+1}"
            cv2.putText(annotated_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)
            
            # Add square properties information
            properties_text = f"A:{area:.0f} R:{aspect_ratio:.2f} V:{vertices} S:{solidity:.2f}"
            cv2.putText(annotated_image, properties_text, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1)
        
        return annotated_image
    
    def detect_objects(self, image_path):
        """
        Main object detection function with enhanced color detection
        
        Args:
            image_path: Path to the image to analyze
            
        Returns:
            detected_objects: List of detected objects
            annotated_image: Image with detection annotations
        """
        print("Detecting objects in image...")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Could not load image: {image_path}")
        
        # Preprocess image
        hsv_image = self.preprocess_image(image)
        
        # Detect objects by HSV color
        hsv_objects = self.detect_objects_by_color(image, hsv_image)
        
        # Detect objects by RGB color (backup for blue/green)
        rgb_objects = self.detect_objects_by_rgb_color(image)
        
        # Detect objects by edges (for objects that might not be detected by color)
        edge_objects = self.detect_objects_by_edges(image)
        
        # Combine all detections
        all_objects = hsv_objects + rgb_objects + edge_objects
        
        # Remove duplicate detections (objects detected by multiple methods)
        unique_objects = self.remove_duplicate_detections(all_objects)
        
        # Prioritize HSV detections over RGB detections for the same object
        unique_objects = self.prioritize_color_detections(unique_objects, hsv_objects, rgb_objects)
        
        # Draw detection results
        annotated_image = self.draw_detection_results(image, unique_objects)
        
        return unique_objects, annotated_image
    
    def prioritize_color_detections(self, unique_objects, hsv_objects, rgb_objects):
        """
        Prioritize HSV color detections over RGB detections for the same object
        
        Args:
            unique_objects: List of unique detected objects
            hsv_objects: Objects detected by HSV method
            rgb_objects: Objects detected by RGB method
            
        Returns:
            prioritized_objects: List with prioritized color detections
        """
        prioritized_objects = []
        
        for obj in unique_objects:
            # Check if this object was detected by HSV method
            hsv_detected = any(self.is_same_object(obj, hsv_obj) for hsv_obj in hsv_objects)
            
            # Check if this object was detected by RGB method
            rgb_detected = any(self.is_same_object(obj, rgb_obj) for rgb_obj in rgb_objects)
            
            # If detected by both methods, prioritize HSV detection
            if hsv_detected and rgb_detected:
                # Find the HSV version of this object
                for hsv_obj in hsv_objects:
                    if self.is_same_object(obj, hsv_obj):
                        prioritized_objects.append(hsv_obj)
                        break
            else:
                # Keep the original object
                prioritized_objects.append(obj)
        
        return prioritized_objects
    
    def is_same_object(self, obj1, obj2, overlap_threshold=0.7):
        """
        Check if two objects are the same based on bounding box overlap
        
        Args:
            obj1: First object
            obj2: Second object
            overlap_threshold: Threshold for considering objects as the same
            
        Returns:
            bool: True if objects are the same
        """
        iou = self.calculate_iou(obj1['bbox'], obj2['bbox'])
        return iou > overlap_threshold
    
    def remove_duplicate_detections(self, objects, overlap_threshold=0.7):
        """
        Remove duplicate object detections based on bounding box overlap
        
        Args:
            objects: List of detected objects
            overlap_threshold: Threshold for considering objects as duplicates
            
        Returns:
            unique_objects: List of unique objects
        """
        if not objects:
            return []
        
        unique_objects = []
        
        for obj in objects:
            is_duplicate = False
            
            for unique_obj in unique_objects:
                # Calculate intersection over union (IoU)
                iou = self.calculate_iou(obj['bbox'], unique_obj['bbox'])
                
                if iou > overlap_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_objects.append(obj)
        
        return unique_objects
    
    def calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes
        
        Args:
            bbox1: First bounding box (x, y, w, h)
            bbox2: Second bounding box (x, y, w, h)
            
        Returns:
            iou: IoU value
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def run_detection_cycle(self):
        """
        Run complete object detection cycle
        """
        try:
            print("=== Object Detection Cycle ===")
            
            # Step 1: Move to optimal viewing position
            # print("\nStep 1: Positioning robot...")
            # self.move_to_optimal_view()
            
            # Step 2: Take photo
            print("\nStep 2: Taking photo...")
            photo_path = self.take_photo("object_detection.jpg")
            
            # Step 3: Detect objects
            print("\nStep 3: Detecting objects...")
            detected_objects, annotated_image = self.detect_objects(photo_path)
            
            # Step 4: Save results
            print("\nStep 4: Saving results...")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Save annotated image
            annotated_filename = f"object_detection_result_{timestamp}.jpg"
            cv2.imwrite(annotated_filename, annotated_image)
            
            # Save detection data
            detection_data = []
            for i, obj in enumerate(detected_objects):
                detection_data.append({
                    'object_id': i + 1,
                    'color': obj['color'],
                    'shape': obj['shape'],
                    'area': obj['area'],
                    'bbox': obj['bbox'],
                    'center': obj['center']
                })
            
            # Print detection summary
            print(f"\n=== Square Block Detection Summary ===")
            print(f"✓ {len(detected_objects)} square blocks detected")
            print(f"✓ Results saved as: {annotated_filename}")
            
            # Count blocks by color
            color_counts = {'red': 0, 'green': 0, 'blue': 0, 'yellow': 0, 'unknown': 0}
            for obj in detected_objects:
                color_counts[obj['color']] += 1
            
            print(f"\nColor distribution:")
            for color, count in color_counts.items():
                if count > 0:
                    print(f"  {color.capitalize()}: {count} block(s)")
            
            print(f"\nDetailed block information:")
            for i, obj in enumerate(detected_objects):
                print(f"  Block {i+1}: {obj['color']} square at {obj['center']} "
                      f"(area: {obj['area']:.0f}, ratio: {obj['aspect_ratio']:.2f}, "
                      f"vertices: {obj['vertices']}, solidity: {obj['solidity']:.2f})")
            
            return True, detected_objects, annotated_image
                
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

    def test_camera(self):
        """
        Test camera functionality and list available cameras
        """
        print("=== Camera Test ===")
        
        # Test direct camera access
        print("\n1. Testing direct camera access...")
        for i in range(4):  # Test indices 0-3
            print(f"\nTesting camera index {i}:")
            camera = cv2.VideoCapture(i)
            if camera.isOpened():
                print(f"  ✓ Camera {i} opened successfully")
                
                # Try to read a frame
                ret, frame = camera.read()
                if ret and frame is not None:
                    print(f"  ✓ Camera {i} can read frames (shape: {frame.shape})")
                else:
                    print(f"  ✗ Camera {i} opened but cannot read frames")
                
                camera.release()
            else:
                print(f"  ✗ Camera {i} failed to open")
        
        # Test ROS camera topics
        print("\n2. Testing ROS camera topics...")
        try:
            import rospy
            from sensor_msgs.msg import Image
            
            # Initialize ROS node
            try:
                rospy.init_node('camera_test', anonymous=True)
            except:
                pass
            
            # List available topics
            try:
                topics = rospy.get_published_topics()
                camera_topics = [topic[0] for topic in topics if 'image' in topic[0].lower() or 'camera' in topic[0].lower()]
                
                if camera_topics:
                    print("  Available camera-related topics:")
                    for topic in camera_topics:
                        print(f"    - {topic}")
                else:
                    print("  No camera-related topics found")
                    
            except Exception as e:
                print(f"  Error listing topics: {str(e)}")
                
        except ImportError:
            print("  ROS not available")
        except Exception as e:
            print(f"  ROS test error: {str(e)}")
        
        print("\n=== Camera Test Complete ===")

def main():
    """Main function to run object detection"""
    parser = argparse.ArgumentParser(description='Object detection on chessboard area')
    parser.add_argument('--min-area', type=int, default=500,
                       help='Minimum object area for detection')
    parser.add_argument('--max-area', type=int, default=50000,
                       help='Maximum object area for detection')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to existing image (if not provided, will take new photo)')
    parser.add_argument('--test-camera', action='store_true',
                       help='Test camera functionality without running detection')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ObjectDetector()
    
    # Update detection parameters
    detector.min_object_area = args.min_area
    detector.max_object_area = args.max_area
    
    if args.test_camera:
        # Test camera functionality
        detector.test_camera()
        return
    
    if args.image:
        # Analyze existing image
        print(f"Analyzing existing image: {args.image}")
        detected_objects, annotated_image = detector.detect_objects(args.image)
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        annotated_filename = f"object_detection_result_{timestamp}.jpg"
        cv2.imwrite(annotated_filename, annotated_image)
        
        print(f"Analysis complete. Results saved as: {annotated_filename}")
        print(f"Detected {len(detected_objects)} objects")
        
    else:
        # Run full detection cycle
        success, detected_objects, annotated_image = detector.run_detection_cycle()
        
        if success:
            print("\n✓ Detection cycle completed successfully!")
        else:
            print("\n✗ Detection cycle failed!")
            print("\nTroubleshooting tips:")
            print("1. Run with --test-camera to diagnose camera issues")
            print("2. Make sure camera is connected and not in use by another process")
            print("3. Try running 'ls /dev/video*' to see available cameras")
            print("4. Check if ROS camera topics are available with 'rostopic list'")

if __name__ == "__main__":
    main() 