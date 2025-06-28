#!/usr/bin/env python3
"""
Goal: Allow dofBot to identify all squares on the checkerboard grid,
        identify blocks of a solid color,
        calculate rotation to be on a linear plane with the block,
        use reverse 2D kinematics to calculate joint angles,
        pick up block and move it to pre-set location
"""

import cv2
import os
import numpy as np
from camera_calibration import find_chessboard_corners, create_object_points, convert_to_2d, calculate_homography
from object_detection import ObjectDetector

def detect_chessboard_and_create_grid_mapping(image_path, pattern_size=(6, 9), square_size=25.4):
    """
    Detect chessboard in image and create homography mapping from image to grid coordinates
    
    Args:
        image_path: Path to image containing chessboard
        pattern_size: (width (in), height (in)) of interior corners
        square_size: Size of each square in mm (25.4mm == 1in)
        
    Returns:
        homography_matrix: Transformation from image pixels to grid coordinates
        success: Boolean indicating if chessboard was detected
        0 = empty square
        1 = red block
        2 = green block
        3 = blue block
        4 = yellow block
        5 = unknown color
    """
    # Create object points for the chessboard
    object_points_3d = create_object_points(pattern_size, square_size)
    object_points_2d = convert_to_2d(object_points_3d)
    
    # Find chessboard corners in the image
    success, image_points, gray = find_chessboard_corners(image_path, pattern_size)
    
    if not success:
        print("Failed to detect chessboard in image")
        return None, False
    
    # Calculate homography from object points to image points
    homography = calculate_homography(object_points_2d, image_points)
    
    if homography is None:
        print("Failed to calculate homography")
        return None, False
    
    print("Successfully created grid mapping")
    return homography, True

def transform_pixel_to_grid(pixel_point, homography_matrix, grid_size=(6, 9)):
    """
    Transform pixel coordinates to grid coordinates using homography
    
    Args:
        pixel_point: (x, y) pixel coordinates
        homography_matrix: Homography transformation matrix
        grid_size: (width, height) of the grid
        
    Returns:
        grid_coords: (x, y) grid coordinates, or None if outside grid
    """
    if homography_matrix is None:
        return None
    
    # Convert pixel point to homogeneous coordinates
    pixel_homogeneous = np.array([[pixel_point[0]], [pixel_point[1]], [1]], dtype=np.float32)
    
    # Apply inverse homography to get grid coordinates
    grid_homogeneous = np.linalg.inv(homography_matrix) @ pixel_homogeneous
    
    # Convert back to 2D coordinates
    if grid_homogeneous[2] != 0:
        grid_x = grid_homogeneous[0] / grid_homogeneous[2]
        grid_y = grid_homogeneous[1] / grid_homogeneous[2]
    else:
        return None
    
    # Convert to discrete grid coordinates (round to nearest integer)
    grid_x_discrete = int(round(grid_x[0] / 25.4))  # Divide by square size to get grid units
    grid_y_discrete = int(round(grid_y[0] / 25.4))
    
    # Check if coordinates are within grid bounds
    if 0 <= grid_x_discrete < grid_size[0] and 0 <= grid_y_discrete < grid_size[1]:
        return (grid_x_discrete, grid_y_discrete)
    else:
        return None

def create_block_grid_matrix(detected_objects, homography_matrix, grid_size=(6, 9)):
    """
    Create a matrix representation of blocks on the chessboard grid
    
    Args:
        detected_objects: List of detected objects from object_detection.py
        homography_matrix: Homography transformation matrix
        grid_size: (width, height) of the grid
        
    Returns:
        grid_matrix: numpy array representing the grid with color codes
        block_positions: Dictionary mapping grid positions to block info
    """
    # Initialize grid matrix with zeros (empty squares)
    grid_matrix = np.zeros(grid_size, dtype=int)
    
    # Color mapping
    color_codes = {
        'red': 1,
        'green': 2,
        'blue': 3,
        'yellow': 4,
        'unknown': 5
    }
    
    # Dictionary to store detailed block information
    block_positions = {}
    
    for obj in detected_objects:
        # Get block center in pixel coordinates
        pixel_center = obj['center']
        
        # Transform to grid coordinates
        grid_coords = transform_pixel_to_grid(pixel_center, homography_matrix, grid_size)
        
        if grid_coords is not None:
            x, y = grid_coords
            color = obj['color']
            color_code = color_codes.get(color, 5)  # Default to 5 for unknown
            
            # Set grid matrix value
            grid_matrix[y, x] = color_code
            
            # Store detailed block information
            block_positions[grid_coords] = {
                'color': color,
                'color_code': color_code,
                'area': obj['area'],
                'pixel_center': pixel_center,
                'bbox': obj['bbox']
            }
            
            print(f"Block detected: {color} at grid position ({x+1}, {y+1})")
    
    return grid_matrix, block_positions

def analyze_chessboard_with_blocks(image_path, pattern_size=(6, 9), square_size=25.4):
    """
    Complete analysis function: detect chessboard, find blocks, create grid matrix
    
    Args:
        image_path: Path to image to analyze
        pattern_size: (width, height) of interior corners
        square_size: Size of each square in mm
        
    Returns:
        grid_matrix: Matrix representation of blocks on grid
        block_positions: Dictionary with detailed block information
        success: Boolean indicating overall success
    """
    print(f"Analyzing image: {image_path}")
    
    # Step 1: Detect chessboard and create grid mapping
    homography, chessboard_success = detect_chessboard_and_create_grid_mapping(
        image_path, pattern_size, square_size)
    
    if not chessboard_success:
        return None, None, False
    
    # Step 2: Detect objects in the image
    detector = ObjectDetector()
    detected_objects, annotated_image = detector.detect_objects(image_path)
    
    if not detected_objects:
        print("No objects detected in image")
        return np.zeros(pattern_size, dtype=int), {}, True
    
    # Step 3: Create grid matrix
    grid_matrix, block_positions = create_block_grid_matrix(
        detected_objects, homography, pattern_size)
    
    # Step 4: Print results
    print("\n=== Grid Analysis Results ===")
    print(f"Grid size: {pattern_size[0]} x {pattern_size[1]}")
    print(f"Blocks detected: {len(block_positions)}")
    
    print("\nGrid Matrix (0=empty, 1=red, 2=green, 3=blue, 4=yellow, 5=unknown):")
    for i in range(pattern_size[1]):
        row_str = " ".join([str(grid_matrix[i, j]) for j in range(pattern_size[0])])
        print(f"Row {i+1}: {row_str}")
    
    print("\nBlock positions:")
    for (x, y), info in block_positions.items():
        print(f"  Position ({x+1}, {y+1}): {info['color']} block (area: {info['area']:.0f})")
    
    return grid_matrix, block_positions, True

def get_block_at_position(grid_matrix, block_positions, x, y):
    """
    Get block information at specific grid position (1-indexed)
    
    Args:
        grid_matrix: Grid matrix from analyze_chessboard_with_blocks
        block_positions: Block positions dictionary
        x, y: Grid coordinates (1-indexed, e.g., (1,1) to (6,9))
        
    Returns:
        block_info: Dictionary with block info, or None if empty
    """
    # Convert to 0-indexed
    grid_x, grid_y = x - 1, y - 1
    
    # Check bounds
    if not (0 <= grid_x < grid_matrix.shape[1] and 0 <= grid_y < grid_matrix.shape[0]):
        print(f"Position ({x}, {y}) is out of bounds")
        return None
    
    # Check if position has a block
    if grid_matrix[grid_y, grid_x] == 0:
        return None
    
    # Return block info
    grid_coords = (grid_x, grid_y)
    return block_positions.get(grid_coords, None)

def display_result_image(image_path, grid_matrix, block_positions, pattern_size=(6, 9)):
    """
    Display the image with chessboard lines and detected blocks using OpenCV
    
    Args:
        image_path: Path to the original image
        grid_matrix: Grid matrix with block positions
        block_positions: Dictionary with detailed block information
        pattern_size: (width, height) of interior corners
    """
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    display_image = image.copy()
    
    # Step 1: Draw chessboard corners and grid lines
    success, corners, gray = find_chessboard_corners(image_path, pattern_size)
    
    if success:
        # Draw chessboard corners
        cv2.drawChessboardCorners(display_image, pattern_size, corners, True)
        print("✓ Chessboard corners drawn")
        
        # Draw grid lines connecting corners
        corners_reshaped = corners.reshape(pattern_size[1], pattern_size[0], 2)
        
        # Draw horizontal lines
        for i in range(pattern_size[1]):
            for j in range(pattern_size[0] - 1):
                pt1 = tuple(corners_reshaped[i, j].astype(int))
                pt2 = tuple(corners_reshaped[i, j + 1].astype(int))
                cv2.line(display_image, pt1, pt2, (0, 255, 0), 2)
        
        # Draw vertical lines
        for i in range(pattern_size[1] - 1):
            for j in range(pattern_size[0]):
                pt1 = tuple(corners_reshaped[i, j].astype(int))
                pt2 = tuple(corners_reshaped[i + 1, j].astype(int))
                cv2.line(display_image, pt1, pt2, (0, 255, 0), 2)
        
        print("✓ Grid lines drawn")
    
    # Step 2: Draw detected blocks
    detector = ObjectDetector()
    detected_objects, _ = detector.detect_objects(image_path)
    
    # Color mapping for visualization
    color_map = {
        'red': (0, 0, 255),      # BGR format
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'unknown': (128, 128, 128)
    }
    
    for (grid_x, grid_y), info in block_positions.items():
        # Draw bounding box around block
        pixel_center = info['pixel_center']
        color = info['color']
        draw_color = color_map.get(color, color_map['unknown'])
        
        # Find the corresponding detected object to get bbox
        for obj in detected_objects:
            if obj['center'] == pixel_center:
                x, y, w, h = obj['bbox']
                
                # Draw bounding box
                cv2.rectangle(display_image, (x, y), (x + w, y + h), draw_color, 3)
                
                # Draw center point
                cv2.circle(display_image, pixel_center, 8, draw_color, -1)
                cv2.circle(display_image, pixel_center, 12, (255, 255, 255), 2)
                
                # Add label with grid position
                label = f"{color} ({grid_x+1},{grid_y+1})"
                cv2.putText(display_image, label, (x, y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2)
                break
    
    print(f"✓ {len(block_positions)} block(s) highlighted")
    
    # Step 3: Display the image
    window_name = "Chessboard Detection and Block Analysis"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 800)  # Resize for better viewing
    cv2.imshow(window_name, display_image)
    
    print("\n=== Image Display Controls ===")
    print("- Press any key to close the image window")
    print("- Press 's' to save the annotated image")
    
    # Wait for key press
    key = cv2.waitKey(0) & 0xFF
    
    # Save image if 's' is pressed
    if key == ord('s'):
        output_filename = f"annotated_{os.path.basename(image_path)}"
        cv2.imwrite(output_filename, display_image)
        print(f"✓ Annotated image saved as: {output_filename}")
    
    cv2.destroyAllWindows()
    print("Image window closed")

def calculate_joint1_angle_to_block(block_grid_position, robot_position=(3.5, 9.5)):
    """
    Calculate the angle for joint 1 to rotate towards a block on the chessboard
    
    Args:
        block_grid_position: Tuple (x, y) of block position (1-indexed, e.g., (4, 7))
        robot_position: Tuple (x, y) of robot position in inches (default: (3.5, 9.5))
        
    Returns:
        angle_degrees: Angle in degrees for joint 1 rotation
        distance_info: Dictionary with calculation details
    """
    import math
    
    block_x, block_y = block_grid_position
    robot_x, robot_y = robot_position
    
    # Convert grid position to real-world coordinates (inches)
    # Grid squares are 1 inch each, block center is in middle of square
    block_real_x = block_x - 0.5  # Convert to 0-indexed then add 0.5 for center
    block_real_y = block_y - 0.5
    
    # Calculate distances from robot to block
    horizontal_distance = abs(block_real_x - robot_x)  # x-direction
    vertical_distance = abs(block_real_y - robot_y)    # y-direction (towards board)
    
    # Calculate angle using inverse tangent
    # theta = tan^-1(opposite/adjacent) = tan^-1(horizontal/vertical)
    if vertical_distance == 0:
        # Block is at same y-level as robot
        angle_radians = math.pi / 2 if horizontal_distance > 0 else 0
    else:
        angle_radians = math.atan(horizontal_distance / vertical_distance)
    
    # Convert to degrees
    angle_degrees = math.degrees(angle_radians)
    
    # Determine direction of rotation
    if block_real_x < robot_x:
        # Block is to the left of robot (negative x direction)
        angle_degrees = -angle_degrees
    
    # Calculate total distance for reference
    total_distance = math.sqrt(horizontal_distance**2 + vertical_distance**2)
    
    distance_info = {
        'block_real_coords': (block_real_x, block_real_y),
        'robot_coords': (robot_x, robot_y),
        'horizontal_distance': horizontal_distance,
        'vertical_distance': vertical_distance,
        'total_distance': total_distance,
        'angle_radians': angle_radians
    }
    
    return angle_degrees, distance_info

def calculate_angles_for_all_blocks(grid_matrix, block_positions, robot_position=(3.5, 9.5)):
    """
    Calculate joint 1 angles for all detected blocks
    
    Args:
        grid_matrix: Grid matrix from analyze_chessboard_with_blocks
        block_positions: Dictionary with block information
        robot_position: Tuple (x, y) of robot position in inches
        
    Returns:
        block_angles: Dictionary mapping block positions to angle calculations
    """
    block_angles = {}
    
    for (grid_x, grid_y), block_info in block_positions.items():
        # Convert 0-indexed to 1-indexed for calculation
        block_position = (grid_x + 1, grid_y + 1)
        
        angle, distance_info = calculate_joint1_angle_to_block(block_position, robot_position)
        
        block_angles[block_position] = {
            'color': block_info['color'],
            'angle_degrees': angle,
            'distance_info': distance_info
        }
    
    return block_angles

def print_angle_calculations(block_angles):
    """
    Print detailed angle calculations for all blocks
    
    Args:
        block_angles: Dictionary from calculate_angles_for_all_blocks
    """
    print("\n=== Joint 1 Angle Calculations ===")
    
    if not block_angles:
        print("No blocks detected for angle calculation")
        return
    
    for block_pos, info in block_angles.items():
        x, y = block_pos
        color = info['color']
        angle = info['angle_degrees']
        dist_info = info['distance_info']
        
        print(f"\nBlock at position ({x}, {y}) - {color.upper()}:")
        print(f"  Real coordinates: ({dist_info['block_real_coords'][0]:.1f}, {dist_info['block_real_coords'][1]:.1f}) inches")
        print(f"  Horizontal distance: {dist_info['horizontal_distance']:.1f} inches")
        print(f"  Vertical distance: {dist_info['vertical_distance']:.1f} inches")
        print(f"  Total distance: {dist_info['total_distance']:.2f} inches")
        print(f"  Joint 1 angle: {angle:.2f}°")
        
        # Show calculation step-by-step
        h_dist = dist_info['horizontal_distance']
        v_dist = dist_info['vertical_distance']
        print(f"  Calculation: θ = tan⁻¹({h_dist:.1f}/{v_dist:.1f}) = {angle:.2f}°")

def map_angle_to_joint1_servo(calculated_angle, robot_center_servo_value=90):
    """
    Map the calculated angle to the robot's joint 1 servo value
    
    Based on object_detection.py, joint 1 (base rotation) uses servo values:
    - 90° = center position (robot facing forward)
    - Values < 90 = rotate left (counterclockwise)  
    - Values > 90 = rotate right (clockwise)
    - Range appears to be 0-180 degrees
    
    Args:
        calculated_angle: Angle in degrees from calculate_joint1_angle_to_block
                         (positive = right, negative = left)
        robot_center_servo_value: Servo value when robot faces center (default: 90)
        
    Returns:
        servo_value: Servo value for joint 1 (0-180)
        rotation_info: Dictionary with mapping details
    """
    
    # Map calculated angle to servo value
    # Positive calculated angle = block is to the right = need to rotate right = servo > 90
    # Negative calculated angle = block is to the left = need to rotate left = servo < 90
    servo_value = robot_center_servo_value + calculated_angle
    
    # Ensure servo value is within valid range (0-180)
    servo_value = max(0, min(180, servo_value))
    
    # Determine rotation direction
    if calculated_angle > 0:
        direction = "right (clockwise)"
    elif calculated_angle < 0:
        direction = "left (counterclockwise)"
    else:
        direction = "center (no rotation)"
    
    rotation_info = {
        'calculated_angle': calculated_angle,
        'servo_value': servo_value,
        'rotation_direction': direction,
        'rotation_amount': abs(calculated_angle),
        'center_servo_value': robot_center_servo_value
    }
    
    return servo_value, rotation_info

def calculate_joint1_servo_for_all_blocks(block_angles, robot_center_servo=90):
    """
    Calculate joint 1 servo values for all detected blocks
    
    Args:
        block_angles: Dictionary from calculate_angles_for_all_blocks
        robot_center_servo: Servo value for center position (default: 90)
        
    Returns:
        block_servo_values: Dictionary with servo calculations for each block
    """
    block_servo_values = {}
    
    for block_pos, angle_info in block_angles.items():
        calculated_angle = angle_info['angle_degrees']
        
        servo_value, rotation_info = map_angle_to_joint1_servo(calculated_angle, robot_center_servo)
        
        block_servo_values[block_pos] = {
            'color': angle_info['color'],
            'calculated_angle': calculated_angle,
            'servo_value': servo_value,
            'rotation_info': rotation_info,
            'distance_info': angle_info['distance_info']
        }
    
    return block_servo_values

def print_servo_calculations(block_servo_values):
    """
    Print detailed servo calculations for all blocks
    
    Args:
        block_servo_values: Dictionary from calculate_joint1_servo_for_all_blocks
    """
    print("\n=== Joint 1 Servo Value Calculations ===")
    
    if not block_servo_values:
        print("No blocks detected for servo calculation")
        return
    
    for block_pos, info in block_servo_values.items():
        x, y = block_pos
        color = info['color']
        calc_angle = info['calculated_angle']
        servo_value = info['servo_value']
        rotation_info = info['rotation_info']
        
        print(f"\nBlock at position ({x}, {y}) - {color.upper()}:")
        print(f"  Calculated angle: {calc_angle:.2f}°")
        print(f"  Servo value: {servo_value:.0f}")
        print(f"  Rotation: {rotation_info['rotation_direction']}")
        print(f"  Rotation amount: {rotation_info['rotation_amount']:.2f}°")
        
        # Show mapping logic
        center_servo = rotation_info['center_servo_value']
        if calc_angle > 0:
            print(f"  Mapping: {center_servo} + {calc_angle:.2f}° = {servo_value:.0f} (rotate right)")
        elif calc_angle < 0:
            print(f"  Mapping: {center_servo} + ({calc_angle:.2f}°) = {servo_value:.0f} (rotate left)")
        else:
            print(f"  Mapping: {center_servo} + 0° = {servo_value:.0f} (no rotation needed)")

def get_servo_command_for_block(block_position, grid_matrix, block_positions, robot_position=(3.5, 9.5)):
    """
    Get the complete servo command to rotate joint 1 towards a specific block
    
    Args:
        block_position: Tuple (x, y) of block position (1-indexed)
        grid_matrix: Grid matrix from analyze_chessboard_with_blocks
        block_positions: Dictionary with block information
        robot_position: Tuple (x, y) of robot position in inches
        
    Returns:
        servo_command: Dictionary with all information needed to command the robot
    """
    # Calculate angle to block
    angle, distance_info = calculate_joint1_angle_to_block(block_position, robot_position)
    
    # Map to servo value
    servo_value, rotation_info = map_angle_to_joint1_servo(angle)
    
    # Get block info
    grid_x, grid_y = block_position[0] - 1, block_position[1] - 1  # Convert to 0-indexed
    block_info = block_positions.get((grid_x, grid_y), {})
    
    servo_command = {
        'block_position': block_position,
        'block_color': block_info.get('color', 'unknown'),
        'servo_value': servo_value,
        'calculated_angle': angle,
        'distance_info': distance_info,
        'rotation_info': rotation_info,
        'joint_id': 1,  # Joint 1 is the base rotation
        'command_ready': True
    }
    
    return servo_command

if __name__ == "__main__":
    print("=== Daniel Robot Movement Calculator Test ===")
    
    # Test with sample image paths
    test_images = [
        "object_detection_20250629_012603.jpg"
    ]
    
    # Try to find an existing image to test with
    test_image = None
    for img_path in test_images:
        if os.path.exists(img_path):
            test_image = img_path
            break
    
    if test_image is None:
        print("No test images found. Please ensure you have one of these files:")
        for img in test_images:
            print(f"  - {img}")
        print("\nTo test manually, run:")
        print("python daniel_robot_movement_calc.py")
        print("Then call: analyze_chessboard_with_blocks('your_image.jpg')")
    else:
        print(f"Testing with image: {test_image}")
        
        try:
            # Run the complete analysis
            grid_matrix, block_positions, success = analyze_chessboard_with_blocks(test_image)
            
            if success:
                print("\n✓ Analysis completed successfully!")
                
                # Test querying specific positions
                print("\n=== Testing Position Queries ===")
                test_positions = [(1, 1), (3, 5), (6, 9), (2, 4)]
                
                for x, y in test_positions:
                    block_info = get_block_at_position(grid_matrix, block_positions, x, y)
                    if block_info:
                        print(f"Position ({x}, {y}): {block_info['color']} block")
                    else:
                        print(f"Position ({x}, {y}): Empty")
                
                # Show grid statistics
                if len(block_positions) > 0:
                    print(f"\n=== Grid Statistics ===")
                    colors = {}
                    for pos, info in block_positions.items():
                        color = info['color']
                        colors[color] = colors.get(color, 0) + 1
                    
                    print("Color distribution:")
                    for color, count in colors.items():
                        print(f"  {color.capitalize()}: {count} block(s)")
                    
                    # Calculate joint 1 angles for all blocks
                    print("\n=== Calculating Joint 1 Angles ===")
                    block_angles = calculate_angles_for_all_blocks(grid_matrix, block_positions)
                    print_angle_calculations(block_angles)
                    
                    # Calculate servo values for joint 1
                    block_servo_values = calculate_joint1_servo_for_all_blocks(block_angles)
                    print_servo_calculations(block_servo_values)
                
                else:
                    print("\nNo blocks detected in the image.")
                    
            else:
                print("\n✗ Analysis failed - could not detect chessboard or process image")
                
        except Exception as e:
            print(f"\nError during testing: {str(e)}")
            print("This might be due to missing dependencies or image files.")
            print("\nTo test manually:")
            print("1. Take a photo with both chessboard and colored blocks")
            print("2. Run: grid_matrix, block_positions, success = analyze_chessboard_with_blocks('your_image.jpg')")
            print("3. Query positions: get_block_at_position(grid_matrix, block_positions, 1, 1)")

    
    # Optional image display
    # if test_image and success:
    #     print("\n=== Optional Image Display ===")
    #     display_choice = input("Would you like to display the image with chessboard lines and detected blocks? (y/n): ").lower().strip()
    #
    #     if display_choice in ['y', 'yes']:
    #         try:
    #             display_result_image(test_image, grid_matrix, block_positions)
    #         except Exception as e:
    #             print(f"Error displaying image: {str(e)}")
    #             print("Make sure you have a display available and OpenCV is properly installed.")
