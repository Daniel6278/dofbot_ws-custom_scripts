# Homography Estimation with DLT Algorithm

This implementation provides homography estimation using the Direct Linear Transform (DLT) algorithm, based on the OpenCV tutorial: https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html

## Files

- `homography_estimation.py` - Main homography estimation class and functions
- `homography_demo.py` - Demo script showing robot arm integration
- `README_homography.md` - This documentation file

## Requirements

- OpenCV 4.x
- NumPy
- Robot arm hardware (Dofbot)
- Chessboard pattern (9x6 recommended)

## Basic Usage

### 1. Simple Homography Estimation

```bash
# Take new photos and estimate homography
python3 homography_estimation.py --take-photos --visualize

# Use existing images
python3 homography_estimation.py --image1 chessboard1.jpg --image2 chessboard2.jpg --visualize
```

### 2. Robot Arm Integration Demo

```bash
# Run the interactive demo
python3 homography_demo.py
```

## Command Line Arguments

### homography_estimation.py

- `--chessboard-size W H`: Chessboard pattern size (default: 9 6)
- `--square-size SIZE`: Chessboard square size in mm (default: 25.0)
- `--image1 PATH`: Path to first image
- `--image2 PATH`: Path to second image
- `--camera-matrix PATH`: Path to camera calibration file
- `--visualize`: Show visualization windows
- `--take-photos`: Take new photos using camera

### Example Usage

```bash
# Basic homography estimation with custom chessboard
python3 homography_estimation.py --take-photos --chessboard-size 8 6 --square-size 20.0

# With camera calibration
python3 homography_estimation.py --take-photos --camera-matrix camera_calibration.yml --visualize
```

## How It Works

### 1. DLT Algorithm Implementation

The Direct Linear Transform (DLT) algorithm solves for the homography matrix H that satisfies:

```
[x']   [h11 h12 h13] [x]
[y'] = [h21 h22 h23] [y]
[1 ]   [h31 h32 h33] [1]
```

The implementation includes:
- **Point normalization** using Hartley's method for numerical stability
- **SVD-based solution** for the linear system Ah = 0
- **Reprojection error calculation** for quality assessment

### 2. Camera Integration

The script uses the same camera interface as `scan_and_replicate.py`:
- Automatic camera device detection
- GStreamer pipeline for video capture
- Timestamped image saving

### 3. Robot Arm Integration

The demo script shows how to:
- Calibrate camera view to robot workspace using homography
- Convert image coordinates to robot joint angles
- Interactive targeting by clicking on images

## Chessboard Requirements

For best results, use a chessboard pattern with:
- **Size**: 9x6 internal corners (recommended)
- **Square size**: 25mm (adjustable)
- **Material**: Rigid, flat surface
- **Lighting**: Good, even illumination
- **Visibility**: Entire pattern should be visible in camera view

## Output Files

The script generates:
- `chessboard1.jpg`, `chessboard2.jpg` - Input images
- `warped_image.jpg` - Transformed image
- `image2_reference.jpg` - Reference image for comparison

## Error Handling

The script includes comprehensive error handling for:
- Camera connection issues
- Chessboard detection failures
- Insufficient point correspondences
- Numerical instability in homography estimation

## Advanced Features

### Camera Pose Estimation

If a camera calibration file is provided, the script can estimate camera pose from homography:

```bash
python3 homography_estimation.py --take-photos --camera-matrix calibration.yml
```

### Interactive Demo

The `homography_demo.py` script provides:
- Camera-to-robot calibration
- Interactive image clicking for robot positioning
- Real-time coordinate transformation

## Troubleshooting

### Common Issues

1. **Chessboard not detected**: Ensure good lighting and the entire pattern is visible
2. **Camera not found**: Check USB connections and permissions
3. **Poor homography quality**: Use more calibration points or improve image quality
4. **Robot arm errors**: Ensure proper initialization and check servo connections

### Debug Mode

Use the `--visualize` flag to see intermediate results:
```bash
python3 homography_estimation.py --take-photos --visualize
```

## Mathematical Background

The DLT algorithm solves the linear system:

```
[x_i * h31 - h11] [x_i * h32 - h12] [x_i * h33 - h13] [0] [0] [0] [-x_i * x'_i] [-x_i * y'_i] [-x_i]   [h11]
[0] [0] [0] [y_i * h31 - h21] [y_i * h32 - h22] [y_i * h33 - h23] [-y_i * x'_i] [-y_i * y'_i] [-y_i] * [h12] = 0
                                                                                                    [h13]
                                                                                                    [h21]
                                                                                                    [h22]
                                                                                                    [h23]
                                                                                                    [h31]
                                                                                                    [h32]
                                                                                                    [h33]
```

Where (x_i, y_i) and (x'_i, y'_i) are corresponding points in the two images.

## References

- OpenCV Homography Tutorial: https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
- Hartley, R. "In Defense of the Eight-Point Algorithm." IEEE Transactions on Pattern Analysis and Machine Intelligence, 1997.
- Multiple View Geometry in Computer Vision, Hartley & Zisserman 