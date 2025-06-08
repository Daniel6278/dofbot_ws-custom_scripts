#!/usr/bin/env python3
# get_color.py
import cv2
import sys
import argparse
import time
import numpy as np
from typing import Dict, Union, Tuple, Optional, Any

def resize_with_aspect_ratio(image, target_width=640, target_height=480):
    """
    Resize image while maintaining aspect ratio and adding padding if necessary.
    """
    # Get original dimensions
    h, w = image.shape[:2]
    
    # Calculate aspect ratios
    target_ratio = target_width / target_height
    image_ratio = w / h
    
    if image_ratio > target_ratio:
        # Image is wider than target
        new_width = target_width
        new_height = int(target_width / image_ratio)
    else:
        # Image is taller than target
        new_height = target_height
        new_width = int(target_height * image_ratio)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create black background
    background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Calculate position to paste resized image
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    
    # Paste resized image onto background
    background[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return background

def get_color_percentages(
    image_path: str,
    square: bool = False,
    debug: bool = False,
    show: bool = False,
    threshold: float = 0.0,
    square_size: int = 120
) -> Union[Dict[str, float], None]:
    """
    Analyzes an image file to determine the percentage of red, blue, green, and yellow colors.
    """
    try:
        # Read the image file
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Error: Could not read image file {image_path}")
            return None
            
        # Make a copy of the image for drawing
        img_display = img.copy()
        
        # Resize the image while maintaining aspect ratio
        img = resize_with_aspect_ratio(img)
        img_display = resize_with_aspect_ratio(img_display)
        
        # Convert to HSV color space for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define HSV color ranges for our basic colors
        # Red (wraps around the hue circle)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Blue range
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        
        # Green range
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        
        # Yellow range
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        
        # Create masks for each color
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Count pixels of each color
        total_pixels = img.shape[0] * img.shape[1]
        red_pixels = cv2.countNonZero(mask_red)
        blue_pixels = cv2.countNonZero(mask_blue)
        green_pixels = cv2.countNonZero(mask_green)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        
        # Calculate percentages
        red_percent = (red_pixels / total_pixels) * 100
        blue_percent = (blue_pixels / total_pixels) * 100
        green_percent = (green_pixels / total_pixels) * 100
        yellow_percent = (yellow_pixels / total_pixels) * 100
        
        # Create result dictionary
        color_percentages: Dict[str, float] = {
            "red": red_percent,
            "blue": blue_percent,
            "green": green_percent,
            "yellow": yellow_percent
        }
        
        # Only print debug info if debug is True
        if debug:
            print("\n--- COLOR DETECTION INFO ---")
            print(f"Total pixels analyzed: {total_pixels}")
            print(f"Color percentages:")
            print(f"  Red: {red_percent:.2f}%")
            print(f"  Blue: {blue_percent:.2f}%")
            print(f"  Green: {green_percent:.2f}%")
            print(f"  Yellow: {yellow_percent:.2f}%")
        
        # Create visualization of the detected colors
        color_vis = np.zeros_like(img)
        color_vis[mask_red > 0] = [0, 0, 255]   # Red in BGR
        color_vis[mask_blue > 0] = [255, 0, 0]  # Blue in BGR
        color_vis[mask_green > 0] = [0, 255, 0] # Green in BGR
        color_vis[mask_yellow > 0] = [0, 255, 255] # Yellow in BGR
        
        # Save debug images only if debug is True
        if debug:
            time_string = str(time.time())
            cv2.imwrite(f"debug_original-{time_string}.jpg", img)
            cv2.imwrite(f"debug_hsv-{time_string}.jpg", hsv)
            cv2.imwrite(f"debug_colors-{time_string}.jpg", color_vis)
        
        # Filter colors by threshold
        filtered_colors = {color: percentage for color, percentage in color_percentages.items() 
                          if percentage >= threshold}
        
        # If no colors meet the threshold, return the highest percentage color
        if not filtered_colors and total_pixels > 0:
            max_color = max(color_percentages.items(), key=lambda x: x[1])
            if max_color[1] > 0:  # Only return if there's at least some color detected
                filtered_colors = {max_color[0]: max_color[1]}
                if debug:
                    print(f"No colors met threshold. Using highest detected color: {max_color[0]} ({max_color[1]:.2f}%)")
        
        # Show debug information if requested
        if debug:
            print("\n--- DEBUG INFORMATION ---")
            print(f"Image dimensions: {img.shape[1]}x{img.shape[0]}")
            
            print("\nRaw pixel counts:")
            print(f"  Red pixels: {red_pixels} ({red_percent:.2f}%)")
            print(f"  Blue pixels: {blue_pixels} ({blue_percent:.2f}%)")
            print(f"  Green pixels: {green_pixels} ({green_percent:.2f}%)")
            print(f"  Yellow pixels: {yellow_pixels} ({yellow_percent:.2f}%)")
            
            # Calculate unclassified pixels
            classified_pixels = red_pixels + blue_pixels + green_pixels + yellow_pixels
            unclassified_pixels = total_pixels - classified_pixels
            unclassified_percent = (unclassified_pixels / total_pixels) * 100
            print(f"  Unclassified pixels: {unclassified_pixels} ({unclassified_percent:.2f}%)")
            
            print("\nHSV Color Ranges Used:")
            print(f"  Red (range 1): H:[{lower_red1[0]}-{upper_red1[0]}], S:[{lower_red1[1]}-{upper_red1[1]}], V:[{lower_red1[2]}-{upper_red1[2]}]")
            print(f"  Red (range 2): H:[{lower_red2[0]}-{upper_red2[0]}], S:[{lower_red2[1]}-{upper_red2[1]}], V:[{lower_red2[2]}-{upper_red2[2]}]")
            print(f"  Blue: H:[{lower_blue[0]}-{upper_blue[0]}], S:[{lower_blue[1]}-{upper_blue[1]}], V:[{lower_blue[2]}-{upper_blue[2]}]")
            print(f"  Green: H:[{lower_green[0]}-{upper_green[0]}], S:[{lower_green[1]}-{upper_green[1]}], V:[{lower_green[2]}-{upper_green[2]}]")
            print(f"  Yellow: H:[{lower_yellow[0]}-{upper_yellow[0]}], S:[{lower_yellow[1]}-{upper_yellow[1]}], V:[{lower_yellow[2]}-{upper_yellow[2]}]")
            
            # Show the visualization if requested
        # if show:
        #     print("-SHOW-")
        #     try:
        #         cv2.imshow('Original Image', img)
        #         cv2.imshow('HSV Image', hsv)
        #         cv2.imshow('Color Detection', color_vis)
        #         print("Press any key to close the image windows")
        #         cv2.waitKey(0)
        #         cv2.destroyAllWindows()
        #     except Exception as e:
        #         print(f"Window display failed: {e}")
            
        return filtered_colors
            
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get percentage of red, blue, green, and yellow in an image.')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--show', action='store_true', help='Show the image with detection visualization')
    parser.add_argument('--debug', action='store_true', help='Show detailed debug information')
    parser.add_argument('--square', action='store_true', help='Analyze only the center square of the image')
    parser.add_argument('--threshold', type=float, default=0.0, help='Minimum percentage to report a color (default: 0.0)')
    parser.add_argument('--square-size', type=int, default=120, help='Size of the center square in pixels (default: 120) (max: 480)')
    args = parser.parse_args()
    
    color_percentages = get_color_percentages(
        args.image_path, 
        args.square, 
        args.debug, 
        args.show, 
        args.threshold,
        args.square_size
    )
    
    if color_percentages:
        print("\nFinal color percentages detected:")
        for color, percentage in color_percentages.items():
            print(f"  {color.upper()}: {percentage:.2f}%")