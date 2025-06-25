#!/usr/bin/env python3
"""
Debug script for chessboard detection
This script helps troubleshoot chessboard corner detection issues
"""

import cv2
import numpy as np
import argparse
import os

def debug_chessboard_detection(image_path, chessboard_size=(7, 10)):
    """
    Debug chessboard detection with multiple preprocessing techniques
    
    Args:
        image_path: Path to the image file
        chessboard_size: Expected chessboard size (width, height)
    """
    print(f"Debugging chessboard detection for image: {image_path}")
    print(f"Expected chessboard size: {chessboard_size}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Save original grayscale
    cv2.imwrite("debug_01_original_gray.jpg", gray)
    print("Saved: debug_01_original_gray.jpg")
    
    # Test 1: Original grayscale
    print("\n=== Test 1: Original grayscale ===")
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    print(f"Detection result: {ret}")
    if ret:
        print(f"Found {len(corners)} corners")
        # Draw corners
        img_with_corners = image.copy()
        cv2.drawChessboardCorners(img_with_corners, chessboard_size, corners, ret)
        cv2.imwrite("debug_02_original_detected.jpg", img_with_corners)
        print("Saved: debug_02_original_detected.jpg")
    
    # Test 2: CLAHE enhancement
    print("\n=== Test 2: CLAHE enhancement ===")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    cv2.imwrite("debug_03_clahe_enhanced.jpg", enhanced)
    print("Saved: debug_03_clahe_enhanced.jpg")
    
    ret, corners = cv2.findChessboardCorners(enhanced, chessboard_size, None)
    print(f"Detection result: {ret}")
    if ret:
        print(f"Found {len(corners)} corners")
        img_with_corners = image.copy()
        cv2.drawChessboardCorners(img_with_corners, chessboard_size, corners, ret)
        cv2.imwrite("debug_04_clahe_detected.jpg", img_with_corners)
        print("Saved: debug_04_clahe_detected.jpg")
    
    # Test 3: Gaussian blur
    print("\n=== Test 3: Gaussian blur ===")
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite("debug_05_gaussian_blur.jpg", blurred)
    print("Saved: debug_05_gaussian_blur.jpg")
    
    ret, corners = cv2.findChessboardCorners(blurred, chessboard_size, None)
    print(f"Detection result: {ret}")
    if ret:
        print(f"Found {len(corners)} corners")
        img_with_corners = image.copy()
        cv2.drawChessboardCorners(img_with_corners, chessboard_size, corners, ret)
        cv2.imwrite("debug_06_blur_detected.jpg", img_with_corners)
        print("Saved: debug_06_blur_detected.jpg")
    
    # Test 4: Adaptive thresholding
    print("\n=== Test 4: Adaptive thresholding ===")
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    cv2.imwrite("debug_07_adaptive_thresh.jpg", adaptive_thresh)
    print("Saved: debug_07_adaptive_thresh.jpg")
    
    ret, corners = cv2.findChessboardCorners(adaptive_thresh, chessboard_size, None)
    print(f"Detection result: {ret}")
    if ret:
        print(f"Found {len(corners)} corners")
        img_with_corners = image.copy()
        cv2.drawChessboardCorners(img_with_corners, chessboard_size, corners, ret)
        cv2.imwrite("debug_08_thresh_detected.jpg", img_with_corners)
        print("Saved: debug_08_thresh_detected.jpg")
    
    # Test 5: Combined preprocessing
    print("\n=== Test 5: Combined preprocessing ===")
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    
    cv2.imwrite("debug_09_combined_preprocessing.jpg", cleaned)
    print("Saved: debug_09_combined_preprocessing.jpg")
    
    ret, corners = cv2.findChessboardCorners(cleaned, chessboard_size, None)
    print(f"Detection result: {ret}")
    if ret:
        print(f"Found {len(corners)} corners")
        img_with_corners = image.copy()
        cv2.drawChessboardCorners(img_with_corners, chessboard_size, corners, ret)
        cv2.imwrite("debug_10_combined_detected.jpg", img_with_corners)
        print("Saved: debug_10_combined_detected.jpg")
    
    # Test 6: Different chessboard sizes
    print("\n=== Test 6: Testing different chessboard sizes ===")
    sizes_to_try = [
        chessboard_size,
        (chessboard_size[1], chessboard_size[0]),  # Swapped
        (chessboard_size[0] - 1, chessboard_size[1] - 1),  # Smaller
        (chessboard_size[0] + 1, chessboard_size[1] + 1),  # Larger
        (6, 9),  # Common alternative
        (8, 6),  # Another common size
    ]
    
    for size in sizes_to_try:
        print(f"Trying size {size}...")
        ret, corners = cv2.findChessboardCorners(gray, size, None)
        if ret:
            print(f"SUCCESS with size {size}!")
            img_with_corners = image.copy()
            cv2.drawChessboardCorners(img_with_corners, size, corners, ret)
            cv2.imwrite(f"debug_11_success_size_{size[0]}x{size[1]}.jpg", img_with_corners)
            print(f"Saved: debug_11_success_size_{size[0]}x{size[1]}.jpg")
            break
        else:
            print(f"Failed with size {size}")
    
    print("\n=== Debug Summary ===")
    print("Check the generated debug images to see which preprocessing method works best.")
    print("If none work, the issue might be:")
    print("1. Chessboard size is incorrect")
    print("2. Chessboard is not fully visible")
    print("3. Lighting is too poor")
    print("4. Chessboard is too blurry")
    print("5. Chessboard pattern is not standard")

def main():
    parser = argparse.ArgumentParser(description='Debug chessboard detection')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--chessboard-size', nargs=2, type=int, default=[7, 10],
                       help='Chessboard pattern size (width height)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file {args.image_path} does not exist")
        return
    
    debug_chessboard_detection(args.image_path, tuple(args.chessboard_size))

if __name__ == "__main__":
    main() 