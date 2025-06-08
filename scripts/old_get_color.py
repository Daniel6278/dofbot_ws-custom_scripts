#!/usr/bin/env python3
# get_color.py
import cv2
import sys
import argparse
import numpy as np
from typing import Dict, Union, Tuple, Optional, Any

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
    
    Args:
        image_path: Path to the image file
        square: If True, analyze only the center square region
        debug: If True, return and show debug information
        show: If True, show the visualization window
        threshold: Minimum percentage to include a color in results
        square_size: Size of the square if square=True (default 120)
        
    Returns:
        Dictionary with filtered color percentages or None if an error occurs
    """
    if (square_size > 480):
        print("max square-size == 480")
        square_size = 480
        
    try:
        # Read the image file
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Error: Could not read image file {image_path}")
            return None
            
        # Make a copy of the image for drawing
        img_display = img.copy()
        
        # Resize the image
        img = cv2.resize(img, (640, 480))
        img_display = cv2.resize(img_display, (640, 480))
        
        
        # If square is True, analyze only the center region
        if square:
            # Define the center square
            center_x, center_y = 640 // 2, 480 // 2
            rect_size = square_size
            x1 = center_x - rect_size // 2
            y1 = center_y - rect_size // 2
            x2 = center_x + rect_size // 2
            y2 = center_y + rect_size // 2
            
            # Draw the center rectangle on the display image
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Extract the center region for analysis
            roi = img[y1:y2, x1:x2]
            cv2.imwrite("square.jpg", img_display)
        else:
            # Analyze the entire image
            roi = img
            x1, y1, x2, y2 = 0, 0, img.shape[1], img.shape[0]  # Full image coordinates
            
        # Convert to HSV color space for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define HSV color ranges for our basic colors
        # Red (wraps around the hue circle)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Blue range
        lower_blue = np.array([100, 70, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Green range
        lower_green = np.array([40, 70, 50])
        upper_green = np.array([80, 255, 255])
        
        # Yellow range
        lower_yellow = np.array([20, 70, 50])
        upper_yellow = np.array([35, 255, 255])
        
        # Create masks for each color
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Count pixels of each color
        total_pixels = roi.shape[0] * roi.shape[1]
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
        filtered_colors = {color: percentage for color, percentage in color_percentages.items() 
                          if percentage >= threshold}
        # Create visualization and print total pixles and HSV ranges if debug is enabled
        if debug or show:
            
            print("\n--- DEBUG INFORMATION ---")
            print(f"Image dimensions: {roi.shape[1]}x{roi.shape[0]}")
            print(f"Total pixels analyzed: {total_pixels}")
            
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
            
            # Analysis area information
            if square:
                print(f"\nAnalyzing center square: ({x1},{y1}) to ({x2},{y2}), Size: {square_size}x{square_size}")
            else:
                print(f"\nAnalyzing entire image: ({x1},{y1}) to ({x2},{y2}), Size: {x2-x1}x{y2-y1}")
            
            # Add histogram info for HSV values in the ROI (optional but useful)
            h_values = hsv[:,:,0].flatten()
            s_values = hsv[:,:,1].flatten()
            v_values = hsv[:,:,2].flatten()
            
            print("\nHSV Distribution:")
            print(f"  Hue - Min: {np.min(h_values)}, Max: {np.max(h_values)}, Mean: {np.mean(h_values):.1f}")
            print(f"  Saturation - Min: {np.min(s_values)}, Max: {np.max(s_values)}, Mean: {np.mean(s_values):.1f}")
            print(f"  Value - Min: {np.min(v_values)}, Max: {np.max(v_values)}, Mean: {np.mean(v_values):.1f}")
            
            print("--- END DEBUG INFO ---\n")
            # total pixles
            print("Pixles information:")
            print(f"Total pixels: {total_pixels}")
            print(f"Red pixels: {red_pixels}")
            print(f"Blue pixels: {blue_pixels}")
            print(f"Green pixels: {green_pixels}")
            print(f"Yellow pixels: {yellow_pixels}")
            
            # Create a visualization of the detected colors
            color_vis = np.zeros_like(roi)
            
            # Apply colors to the mask regions
            color_vis[mask_red > 0] = [0, 0, 255]   # Red in BGR
            color_vis[mask_blue > 0] = [255, 0, 0]  # Blue in BGR
            color_vis[mask_green > 0] = [0, 255, 0] # Green in BGR
            color_vis[mask_yellow > 0] = [0, 255, 255] # Yellow in BGR
            
        # Show the visualization if requested
        if show:
            try:
                cv2.imshow('Color Detection', img_display)
                cv2.imshow('Color Mask', color_vis)
                print("Press any key to close the image window")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Window display failed: {e}")
        
        # Filter colors by threshold
        # filtered_colors = {color: percentage for color, percentage in color_percentages.items() 
        #                   if percentage >= threshold}
            
        return filtered_colors
            
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    
    """
    Main function to handle command-line arguments and process image
    """
    print(f"Command line arg options:\nimage_path: 'Path to the image file\n--show: Show the image with detection visualization\n--debug: Show detailed debug information\n--square: Analyze only the center square of the image\n--threshold: Minimum percentage to report a color, default: 1.0\n--square-size: Size of the center square in pixels, default: 120, max: 480")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Get percentage of red, blue, green, and yellow in an image.')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--show', action='store_true', help='Show the image with detection visualization')
    parser.add_argument('--debug', action='store_true', help='Show detailed debug information')
    parser.add_argument('--square', action='store_true', help='Analyze only the center square of the image')
    parser.add_argument('--threshold', type=float, default=0.0, help='Minimum percentage to report a color (default: 0.0)')
    parser.add_argument('--square-size', type=int, default=120, help='Size of the center square in pixels (default: 120) (max: 480)')
    # need to implement check to make sure square-size is not too big
    args = parser.parse_args()
    
    # Process the image with the combined function
    color_percentages = get_color_percentages(
        args.image_path, 
        args.square, 
        args.debug, 
        args.show, 
        args.threshold,
        args.square_size
    )
    
    if color_percentages:
        # Print detection results
        print("\nColor percentages detected:")
        for color, percentage in color_percentages.items():
            print(f"  {color.upper()}: {percentage:.2f}%")