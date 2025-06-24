#!/usr/bin/env python3
#!/usr/bin/env python3
import cv2
import numpy as np

def create_object_points(pattern_size, square_size):
    """
    Create the known real-world coordinates of chessboard corners
    Need for camera calibration
    Args:
        pattern_size: tuple (width, height) of interior corners
        square_size: size of each square in mm
        
    Returns:
        numpy array of 3D object points
    """
    # Create array to hold 3D coordinates
    object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    
    # Fill in the coordinates
    # This creates a grid of the corners on the chess board
    """
    [
    [0, 0, 0],      
    [0, 1, 0],
    [1, 0, 0],            
    [1, 1, 0],            
    [2, 0, 0],            
    [2, 1, 0]
    ]            
    """
    object_points[:, :2] = np.mgrid[0:pattern_size[0], 
                                   0:pattern_size[1]].T.reshape(-1, 2)
    
    # Scale by square size
    """
    Corner 0: (0.0,   0.0,   0)    ← Top-left corner
    Corner 1: (0.0,   25.4,  0)    ← One square down
    Corner 2: (0.0,   50.8,  0)    ← Two squares down
    ...
    Corner 9: (25.4,  0.0,   0)    ← One square right from top-left
    Corner 10:(25.4,  25.4,  0)    ← One right, one down
    """
    object_points[:, :2] *= square_size
    
    return object_points

def convert_to_2d(object_points_3d):
    """
    Convert 3D object points to 2D for homography (remove Z=0)
    Args:
        object_points_3d: 3D object points array
        
    Returns:
        2D object points array
    """
    return object_points_3d[:, :2].astype(np.float32)

def find_chessboard_corners(image_path, pattern_size):
    """
    Find chessboard corners in a single image
    
    Args:
        image_path: path to the image file
        pattern_size: tuple (width, height) of interior corners
        
    Returns:
        (success, refined_corners, gray_image)
    """

    # Needs img path
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return False, None, None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners in the image
    ret, image_points = cv2.findChessboardCorners(gray, pattern_size, None)
    
    if ret:
        # Refine corner positions for better accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        image_points_refined = cv2.cornerSubPix(gray, image_points, (11, 11), (-1, -1), criteria)
        
        print(f"Found {len(image_points_refined)} corners in {image_path}")
        return True, image_points_refined, gray
    else:
        print(f"Chessboard corners not found in {image_path}")
        return False, None, gray

def calculate_homography(object_points_2d, image_points):
    """
    Calculate homography matrix between object points and image points
    
    Args:
        object_points_2d: 2D object points in real world coordinates
        image_points: detected corner points in image coordinates
        
    Returns:
        homography matrix
    """
    homography, mask = cv2.findHomography(object_points_2d, 
                                        image_points.reshape(-1, 2), 
                                        cv2.RANSAC)
    return homography

def draw_corners_on_image(image_path, pattern_size, corners):
    """
    Draw detected corners on the image for visualization
    
    Args:
        image_path: path to the image file
        pattern_size: tuple (width, height) of interior corners
        corners: detected corner points
        
    Returns:
        image with corners drawn
    """
    img = cv2.imread(image_path)
    img_with_corners = cv2.drawChessboardCorners(img, pattern_size, corners, True)
    return img_with_corners

def process_single_image(image_path, object_points_3d, object_points_2d, pattern_size):
    """
    Process a single chessboard image and calculate homography
    
    Args:
        image_path: path to the image file
        object_points_3d: 3D object points
        object_points_2d: 2D object points  
        pattern_size: tuple (width, height) of interior corners
        
    Returns:
       (image_with_corners, homography_matrix)
    """
    # Find corners in the image
    success, image_points, gray = find_chessboard_corners(image_path, pattern_size)
    
    if not success:
        return None, None
    
    # Calculate homography
    homography = calculate_homography(object_points_2d, image_points)
    
    # Draw corners for visualization
    img_with_corners = draw_corners_on_image(image_path, pattern_size, image_points)
    
    print("First 5 Image Points:")
    print(image_points[:5].reshape(-1, 2))
    print("\nHomography Matrix:")
    print(homography)
    
    return img_with_corners, homography

def calibrate_camera_from_images(image_list, object_points_3d, pattern_size):
    """
    Calibrate camera using multiple chessboard images
    
    Args:
        image_list: list of image file paths
        object_points_3d: 3D object points
        pattern_size: tuple (width, height) of interior corners
        
    Returns:
        (camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors)
    """
    # Arrays to store object points and image points from all images
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane
    image_size = None
    
    for image_path in image_list:
        success, corners, gray = find_chessboard_corners(image_path, pattern_size)
        
        if success:
            obj_points.append(object_points_3d)
            img_points.append(corners)
            
            if image_size is None:
                image_size = gray.shape[::-1]  # (width, height)
    
    if len(obj_points) == 0:
        print("No valid chessboard images found for calibration")
        return None, None, None, None
    
    print(f"Using {len(obj_points)} images for calibration")
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None)
    
    if ret:
        print("Camera calibration successful!")
        print("Camera Matrix:")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(dist_coeffs)
        
        return camera_matrix, dist_coeffs, rvecs, tvecs
    else:
        print("Camera calibration failed")
        return None, None, None, None

def print_object_points_info(object_points_3d, object_points_2d):
    """
    Print information about the created object points
    
    Args:
        object_points_3d: 3D object points
        object_points_2d: 2D object points
    """
    print("3D Object Points (first 5):")
    print(object_points_3d[:5])
    print(f"3D Shape: {object_points_3d.shape}")
    
    print("\n2D Object Points (first 5):")
    print(object_points_2d[:5])
    print(f"2D Shape: {object_points_2d.shape}")
    print("="*40)

if __name__ == "__main__":
    # Chessboard parameters for 7x10 squares (6x9 interior corners), must be interior for correct detection
    board_width = 6   # interior corners (7 squares - 1)
    board_height = 9  # interior corners (10 squares - 1)
    pattern_size = (board_width, board_height)
    square_size = 25.4  # 1 inch = 25.4mm
    
    # Create the known object points - borard is 7inx10in
    object_points_3d = create_object_points(pattern_size, square_size)
    object_points_2d = convert_to_2d(object_points_3d)
    
    print_object_points_info(object_points_3d, object_points_2d)
    
    single_image_path = ''  # Replace with your image path
    img_with_corners, homography_matrix = process_single_image(
        single_image_path, object_points_3d, object_points_2d, pattern_size)
    
    if img_with_corners is not None:
        print("Single image processing successful!")
        cv2.imwrite('chessboard_with_corners.jpg', img_with_corners)
    
    """
    Multiple img processing, need to research further
    """
    # image_list = ['img1.jpg', 'img2.jpg', 'img3.jpg']  # Replace with img paths
    # camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera_from_images(
    #     image_list, object_points_3d, pattern_size)
    # np.savez('camera_calibration.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)