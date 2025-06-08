#!/usr/bin/env python3
# camera_test.py
# Import the USBCamera class
from jetcam.jetcam.usb_camera import USBCamera
import cv2
import signal
import sys
import time
import subprocess

def check_if_camera_in_use(validPID):
    result = subprocess.run(["./check_if_camera_in_use.sh"], capture_output=True, text=True)
    # If exit code is 1, camera is in use
    if result.returncode == 1:
        print(f"(check_if_camera_in_use){result.stdout.strip()}")  # Print the message about camera being in use with PIDs
        sys.exit(1)  # Exit the script
    
# Handle termination cleanly
def signal_handler(sig, frame):
    print('Exiting gracefully...')
    sys.exit(0)
    
signal.signal(signal.SIGINT, signal_handler)

def take_photo(validPID):
    # returns true if it was successful
    try:
        # Check if camera is in use
        check_if_camera_in_use(validPID) 
        
        # Initialize camera with larger output size
        print("Initializing camera...")
        camera = USBCamera(capture_device=0, width=1920, height=1080)
        
        # Read a frame
        print("Reading frame...")
        image = camera.read() #takes photo
        
        print(f"Successfully captured a frame with shape: {image.shape}")
        
        # Save photo
        #name = input("Enter the name of the image: ")
        name = "test1"
        
        cv2.imwrite(f'{name}.jpg', image) 
        print("Test completed successfully.")
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        # Show full stack trace
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    take_photo()