def take_photo(filename="simple_photo.jpg"):
    """
    Simple function to take a photo using camera

    Args:
        filename: Name of the file to save the photo

    Returns:
        filename: Path to the saved photo or None if failed
    """
    import cv2
    import time
    import os

    print("Taking photo...")

    # Generate timestamp for unique naming
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(filename)
    timestamped_filename = f"{name}_{timestamp}{ext}"

    camera = None
    try:
        print("Initializing camera...")

        # Use camera index 1 with V4L2 backend (backend 200)
        camera_index = 1
        backend = cv2.CAP_V4L2
        
        print(f"Trying camera index {camera_index} with backend {backend}")

        # Create camera object with specific backend
        camera = cv2.VideoCapture(camera_index, backend)

        if camera.isOpened():
            print(f"Successfully opened camera with index {camera_index}")

            # Set camera properties
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
                        cv2.imwrite(timestamped_filename, image)
                        print("Photo captured successfully")
                        print(f"Photo saved as: {timestamped_filename}")
                        return timestamped_filename
                else:
                    print(f"Failed to read frame {attempt + 1}")
                    time.sleep(0.5)  # Wait before next attempt

            if successful_frames == 0:
                print(f"Camera index {camera_index} opened but failed to read any frames")
            else:
                print(f"Camera index {camera_index} captured {successful_frames} frames but not enough for reliable capture")
        else:
            print(f"Failed to open camera index {camera_index}")

        # If we get here, camera failed to work
        print("Camera configuration failed")
        return None

    except Exception as e:
        print(f"Camera access error: {str(e)}")
        return None
    finally:
        # Ensure camera is released
        if camera is not None:
            camera.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    name = input("Enter a filename for photo: ")
    photo_path = take_photo(name)
    if photo_path:
        print(f"Success! Photo saved to: {photo_path}")
    else:
        print("Failed to take photo")
