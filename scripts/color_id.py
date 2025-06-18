#!/usr/bin/env python3
# color_id.py
# Daniel was here
import cv2
import sys
import argparse
import numpy as np

def detect_red_block(img, debug=False):
    """
    Detect if there's a red block in the image with improved sensitivity
    """
    # Resize the image
    img = cv2.resize(img, (640, 480))
    
    # Calculate the center detection area (larger - 120x120 pixels)
    center_x, center_y = 640 // 2, 480 // 2
    rect_size = 120
    x1 = center_x - rect_size // 2
    y1 = center_y - rect_size // 2
    x2 = center_x + rect_size // 2
    y2 = center_y + rect_size // 2
    
    # Draw the center rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Extract the center region for analysis
    center_roi = hsv[y1:y2, x1:x2]
    
    # Broadened red HSV ranges for better detection
    lower_red1 = np.array([0, 50, 50])     # More sensitive lower threshold
    upper_red1 = np.array([15, 255, 255])  # Broader upper threshold
    lower_red2 = np.array([150, 50, 50])   # More sensitive lower threshold
    upper_red2 = np.array([180, 255, 255]) # Standard upper threshold
    
    # Create masks for red detection
    mask1 = cv2.inRange(center_roi, lower_red1, upper_red1)
    mask2 = cv2.inRange(center_roi, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Count red pixels
    red_pixel_count = cv2.countNonZero(red_mask)
    total_pixels = rect_size * rect_size
    red_percentage = (red_pixel_count / total_pixels) * 100
    
    # Debug information
    if debug:
        # Visualize the mask on the image
        debug_img = img.copy()
        debug_mask = np.zeros((480, 640), dtype=np.uint8)
        debug_mask[y1:y2, x1:x2] = red_mask
        
        # Find contours in the mask
        contours, _ = cv2.findContours(debug_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug_img, contours, -1, (0, 255, 255), 2)
        
        # Collect HSV statistics from center roi
        h_values = center_roi[:,:,0].flatten()
        s_values = center_roi[:,:,1].flatten()
        v_values = center_roi[:,:,2].flatten()
        
        h_stats = {
            'min': np.min(h_values),
            'max': np.max(h_values), 
            'mean': np.mean(h_values),
            'median': np.median(h_values)
        }
        
        s_stats = {
            'min': np.min(s_values),
            'max': np.max(s_values),
            'mean': np.mean(s_values),
            'median': np.median(s_values)
        }
        
        v_stats = {
            'min': np.min(v_values),
            'max': np.max(v_values),
            'mean': np.mean(v_values),
            'median': np.median(v_values)
        }
        
        return debug_img, red_percentage > 10, red_percentage, h_stats, s_stats, v_stats
    
    return img, red_percentage > 10, red_percentage, None, None, None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect red blocks in an image.')
    parser.add_argument('image_path', help='Path to the JPG image file')
    parser.add_argument('--show', action='store_true', help='Show the image with detection rectangle')
    parser.add_argument('--debug', action='store_true', help='Show detailed debug information')
    parser.add_argument('--threshold', type=float, default=10.0, 
                       help='Percentage threshold for red detection (default: 10.0)')
    args = parser.parse_args()
    
    try:
        # Read the image file
        img = cv2.imread(args.image_path)
        
        if img is None:
            print(f"Error: Could not read image file {args.image_path}")
            sys.exit(1)
            
        # Detect red block with debug info
        img_result, is_red, red_percentage, h_stats, s_stats, v_stats = detect_red_block(img, args.debug)
        
        # Print detection results
        if is_red and red_percentage > args.threshold:
            print(f"RED BLOCK DETECTED")
            print(f"Red pixel percentage: {red_percentage:.2f}%")
        else:
            print("No red block detected in the center")
            print(f"Red percentage: {red_percentage:.2f}% (below threshold of {args.threshold}%)")
        
        # Print debug information if requested
        if args.debug and h_stats and s_stats and v_stats:
            print("\nDebug Information:")
            print(f"Hue (H) stats: {h_stats}")
            print(f"Saturation (S) stats: {s_stats}")
            print(f"Value (V) stats: {v_stats}")
            
        # Optionally show the image with detection rectangle
        if args.show or args.debug:
            cv2.imshow('Red Block Detection', img_result)
            print("Press any key to close the image window")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
