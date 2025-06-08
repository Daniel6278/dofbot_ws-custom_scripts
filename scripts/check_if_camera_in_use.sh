#!/bin/bash
# check_if_camera_in_use.sh
# Check if camera is in use and print PID
camera_check=$(sudo lsof /dev/video* 2>/dev/null)
if [ -n "$camera_check" ]; then
    # Extract PIDs and remove duplicates
    pids=$(echo "$camera_check" | awk 'NR>1 {print $2}' | sort -u | tr '\n' ' ')
    echo "Camera is in use by process ID(s): $pids"
    exit 1
else
    echo "Camera is not in use."
    exit 0
fi