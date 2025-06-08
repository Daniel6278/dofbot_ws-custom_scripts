#!/usr/bin/env python3
# camera_test.py
import cv2
import signal
import sys
import time
import subprocess
import glob
import os

def find_camera_device():
    """
    Find the first available camera device that works.
    Returns the device path (e.g. '/dev/video1') or None if no camera is found.
    """
    # Get list of all video devices
    video_devices = glob.glob('/dev/video*')
    
    if not video_devices:
        print("No video devices found")
        return None
        
    # Try each device until we find one that works
    for device in video_devices:
        try:
            # Try to open the device with a simple pipeline
            test_pipeline = f"v4l2src device={device} ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=1 drop=true"
            cap = cv2.VideoCapture(test_pipeline, cv2.CAP_GSTREAMER)
            
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    print(f"Found working camera at {device}")
                    return device
        except Exception as e:
            print(f"Failed to open {device}: {str(e)}")
            continue
    
    print("No working camera found")
    return None

def check_if_camera_in_use(valid_pid=None):
    result = subprocess.run(["./check_if_camera_in_use.sh"], capture_output=True, text=True)
    # If exit code is 1, camera is in use
    if result.returncode == 1:
        # Get the PIDs from the output
        pids = result.stdout.strip().split(": ")[1].split()
        # Filter out the valid PID if provided
        if valid_pid is not None:
            pids = [pid for pid in pids if int(pid) != valid_pid]
            if not pids:  # If no other PIDs remain, camera is not in use by other processes
                return
        print(f"Camera is in use by process ID(s): {' '.join(pids)}")
        sys.exit(1)  # Exit the script
    
# Handle termination cleanly
def signal_handler(sig, frame):
    print('Exiting gracefully...')
    sys.exit(0)
    
signal.signal(signal.SIGINT, signal_handler)

def take_photo(valid_pid=None):
    # returns the filename if successful, None if failed
    try:
        # Check if camera is in use, excluding valid_pid if provided
        check_if_camera_in_use(valid_pid)
        
        # Find working camera device
        camera_device = find_camera_device()
        if not camera_device:
            raise RuntimeError("No working camera found")
        
        # Initialize camera with GStreamer pipeline
        print(f"Initializing camera at {camera_device}...")
        # Modified pipeline to use detected device and suppress position query warning
        pipeline = f"v4l2src device={camera_device} ! video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink max-buffers=1 drop=true"
        camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if not camera.isOpened():
            raise RuntimeError("Failed to open camera")
        
        # Read a frame
        print("Reading frame...")
        ret, image = camera.read()
        if not ret:
            raise RuntimeError("Failed to read from camera")
        
        print(f"Successfully captured a frame with shape: {image.shape}")
        
        # Save photo with timestamp
        filename = "blockIMG-" + str(time.time()) + ".jpg"
        cv2.imwrite(filename, image)
        
        # Clean up
        camera.release()
        cv2.destroyAllWindows()
        
        print("Test completed successfully.")
        return filename
        
    except Exception as e:
        print(f"Error: {str(e)}")
        # Show full stack trace
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    take_photo()