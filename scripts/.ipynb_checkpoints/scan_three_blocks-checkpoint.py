#!/usr/bin/env python3
# scan_three_blocks.py
import cv2
import numpy as np
import time
from time import sleep
import os
import signal
import subprocess
from arm_controller import ArmController
from camera_test import take_photo
from get_color import get_color_percentages
import argparse

class BlockScanner:
    def __init__(self):
        self.arm = ArmController()
        # Define scanning positions (left, center, right)
        self.scan_positions = {
            'right': {1: 45, 2: 50, 3: 60, 4: 0,} ,
            'center': {1: 90, 2: 75, 3: 25, 4: 0,} ,
            'left': {1: 125, 2: 50, 3: 60, 4: 0,}
        }
        self.current_pid = os.getpid()
        print(f"Current PID: {self.current_pid}")
        # Ensure torque is enabled at start
        self.arm.set_torque(True)
        sleep(0.5)

    def ensure_torque(self):
        """Ensure torque is enabled before any movement"""
        self.arm.set_torque(True)
        sleep(0.5)

    def detect_color(self, debug=False):
        #Returns: (color_name, confidence)
        try:
            # Take photo, pass current PID to allow camera usage
            filename = take_photo(valid_pid=self.current_pid)
            if not filename:
                return None, 0

            # Get color confidence percentages
            color_percentages = get_color_percentages(
                filename,
                square=False,
                debug=debug,
                show=debug,
                threshold=10.0,  # Minimum 10% confidence
                square_size=120
            )

            if not color_percentages:
                return None, 0

            # Get the color with highest confidence
            max_color = max(color_percentages.items(), key=lambda x: x[1])
            return max_color[0], max_color[1]
        except Exception as e:
            print(f"Error in detect_color: {str(e)}")
            return None, 0

    def scan_three_blocks(self, debug=False):
        """
        Scan three blocks in a horizontal line and return their colors
        Returns: list of detected colors [left, center, right]
        """
        detected_colors = [None, None, None]
        positions = ['left', 'center', 'right']
        
        try:
            # Ensure we start from a known state
            self.ensure_torque()
            self.arm.move_home(time_ms=2000)
            sleep(1.8)  # Reduced from 2.0 since movement is 2000ms
            
            for i, pos in enumerate(positions):
                # Ensure torque is enabled before each movement
                self.ensure_torque()
                
                # Move to scanning position
                self.arm.move_arm_only(self.scan_positions[pos], time_ms=2000)
                sleep(1.8)  # Reduced from 2.0 since movement is 2000ms
                
                # Detect color at current position
                color, confidence = self.detect_color(debug)
                detected_colors[i] = color
                
                if debug:
                    print(f"Position {pos}: Detected {color} with {confidence:.1f}% confidence")
                
                sleep(0.8)  # Reduced from 1.0 since we just need a small delay between positions
            
            return detected_colors
            
        finally:
            # Ensure torque is enabled before returning home
            self.ensure_torque()
            # Return to home position
            self.arm.move_home(time_ms=2000)
            sleep(1.8)  # Reduced from 2.0 since movement is 2000ms

    #~~~~~~~~~~~~~~~~~End of scanning auxiliary functions~~~~~~~~~~~~~~~~~~#

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~Recreate scene~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    def close_gripper(self):
        self.arm.move_wrist_gripper(gripper_angle=135, time_ms=1000)
        #sleep(0.8)  # Reduced from 1.0 since gripper movement is quick
        
    def open_gripper(self):
        self.arm.move_wrist_gripper(gripper_angle=30, time_ms=1500)
        #sleep(1.2)  # Reduced from 1.5 since gripper movement is quick

    def reach_down_blue(self):
        current = self.arm.read_current_state()
        self.arm.move_all_joints({1: 145, 2: 50, 3: 40, 4: 25, 5: 66, 6: current[6]}, time_ms=2000)
        #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms
    
    def reach_down_red(self):
        current = self.arm.read_current_state()
        self.arm.move_all_joints({1: 25, 2: 55, 3: 30, 4: 30, 5: 110, 6: current[6]}, time_ms=2000)
        #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms
    
    def reach_down_green(self):
        current = self.arm.read_current_state()
        self.arm.move_arm_only({1: 179, 2: 53, 3: 37, 4: 16}, time_ms=2000)
        #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms
    
    def reach_down_yellow(self):
        current = self.arm.read_current_state()
        self.arm.move_arm_only({1: 0, 2: 53, 3: 28, 4: 22}, time_ms=2000)
        #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms
        
    def reach_down_left(self):
        current = self.arm.read_current_state()
        self.arm.move_all_joints({1: 130, 2: 36, 3: 50, 4: 45, 5: 125, 6: current[6]}, time_ms=3000)
        #sleep(2.8)  # Reduced from 3.0 since movement is 3000ms
        
    def reach_down_middle(self):
        self.arm.move_arm_only({1: 90, 2: 45, 3: 45, 4: 20}, time_ms=2000)
        #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms
        
    def reach_down_right(self):
        current = self.arm.read_current_state()
        self.arm.move_all_joints({1: 44, 2: 36, 3: 60, 4: 30, 5: 45, 6: current[6]}, time_ms=3000)
        #sleep(2.8)  # Reduced from 3.0 since movement is 3000ms

    def reach_up(self):
        current = self.arm.read_current_state()
        self.arm.move_arm_only({
            1: current[1],  # Maintain current rotation
            2: 160,         # Lift arm angles
            3: 20,
            4: 0
        }, time_ms=2000)
        #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms

    def set_left(self):
        print("Placing into left position")
        current = self.arm.read_current_state()
        self.arm.move_single_joint(1, 135, time_ms=2000)
        #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms
        self.arm.move_all_joints({1: 129, 2: 37, 3: 58, 4: 27, 5: 128, 6: current[6]}, time_ms=2000)
        #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms
        self.open_gripper()
        self.reach_up()
        self.arm.move_home(time_ms=2000)
        #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms

    def set_middle(self):
        print("Placing into middle position")
        current = self.arm.read_current_state()
        self.arm.move_single_joint(1, 90, time_ms=2000)
        #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms
        self.arm.move_all_joints({1: 90, 2: 65, 3: 20, 4: 30, 5: 90, 6: current[6]}, time_ms=2000)
        #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms
        self.open_gripper()
        self.reach_up()
        self.arm.move_home(time_ms=2000)
        #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms
        
    def set_right(self):
        print("Placing into right position")
        current = self.arm.read_current_state()
        self.arm.move_single_joint(1, 45, time_ms=2000)
        #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms
        self.arm.move_all_joints({1: 44, 2: 50, 3: 33, 4: 40, 5: 51, 6: current[6]}, time_ms=2000)
        #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms
        self.open_gripper()
        self.reach_up()
        self.arm.move_home(time_ms=2000)
        #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms
    
    def grab_red(self, locationID):
        print("Picking up red block")
        self.arm.move_single_joint(1, 30, time_ms=2000)
        #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms
        self.open_gripper()
        self.reach_down_red()
        self.close_gripper()
        self.reach_up()
        
        if locationID == 0:
            self.set_left()
        elif locationID == 1:
            self.set_middle()
        elif locationID == 2:
            self.set_right()
        else:
            raise ValueError(f"Invalid location ID for red block: {locationID}. Must be 0, 1, or 2.")
        
    def grab_green(self, locationID):
        print("Picking up green block")
        self.arm.move_single_joint(1, 180, time_ms=2000)
        #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms
        self.open_gripper()
        self.reach_down_green()
        self.close_gripper()
        self.reach_up()
        
        if locationID == 0:
            self.set_left()
        elif locationID == 1:
            self.set_middle()
        elif locationID == 2:
            self.set_right()
        else:
            raise ValueError(f"Invalid location ID for green block: {locationID}. Must be 0, 1, or 2.")
                
    def grab_blue(self, locationID):
        print("Picking up blue block")
        self.arm.move_single_joint(1, 150, time_ms=2000)
        #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms
        self.open_gripper()
        self.reach_down_blue()
        self.close_gripper()
        self.reach_up()
        
        if locationID == 0:
            self.set_left()
        elif locationID == 1:
            self.set_middle()
        elif locationID == 2:
            self.set_right()
        else:
            raise ValueError(f"Invalid location ID for blue block: {locationID}. Must be 0, 1, or 2.")
                
    def grab_yellow(self, locationID):
        print("Picking up yellow block")
        self.arm.move_single_joint(1, 0, time_ms=2000)
        #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms
        self.open_gripper()
        self.reach_down_yellow()
        self.close_gripper()
        self.reach_up()
        
        if locationID == 0:
            self.set_left()
        elif locationID == 1:
            self.set_middle()
        elif locationID == 2:
            self.set_right()
        else:
            raise ValueError(f"Invalid location ID for yellow block: {locationID}. Must be 0, 1, or 2.")
                
    def move_block(self, colorID, locationID):
        #call scene recreation functions based on parameters
        if colorID == "red":
            self.grab_red(locationID)
        elif colorID == "green":
            self.grab_green(locationID)
        elif colorID == "blue":
            self.grab_blue(locationID)
        elif colorID == "yellow":
            self.grab_yellow(locationID)
        else:
            raise ValueError(f"Invalid color ID: {colorID}. Must be one of: red, green, blue, yellow")
    
    def create_scene(self, colors):
        # Validate that we have exactly 3 colors
        if not isinstance(colors, list) or len(colors) != 3:
            raise ValueError("create_scene requires exactly 3 colors")
        
        # Validate that all colors are valid
        valid_colors = ["red", "green", "blue", "yellow"]
        for color in colors:
            if color not in valid_colors:
                raise ValueError(f"Invalid color detected: {color}. Must be one of: {valid_colors}")
        
        try:
            # Ensure torque is enabled before starting scene creation
            self.ensure_torque()
            self.arm.move_home(time_ms=2000)
            #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms
            
            #move blocks of colors(0), colors(1), colors(2) to their respective locations
            for i, color in enumerate(colors):
                # Ensure torque is enabled before each block movement
                self.ensure_torque()
                self.move_block(color, i)
                #sleep(0.8)  # Reduced from 1.0 since we just need a small delay between blocks
                
        finally:
            # Ensure torque is enabled before final home position
            self.ensure_torque()
            self.arm.move_home(time_ms=2000)
            #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Scan three blocks and detect their colors.')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--show', action='store_true', help='Show debug images')
    args = parser.parse_args()
    
    scanner = BlockScanner()
    try:
        #ensure arm starts at an appropriate position
        scanner.ensure_torque()
        scanner.arm.move_home(time_ms=2000)
        #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms

        #Identify 3 blocks
        colors = scanner.scan_three_blocks(debug=args.debug)
        print("\nDetected colors:")
        print(f"Left: {colors[0]}")
        print(f"Center: {colors[1]}")
        print(f"Right: {colors[2]}")

        # recreate scene if 3 valid colors were identified
        if all(color is not None for color in colors):
            input("Press enter to recreate scene of detected colors...")
            scanner.create_scene(colors)
        else:
            print("Error: Could not detect all three colors. Skipping scene recreation.")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Ensure torque is enabled before final home position
        scanner.ensure_torque()
        scanner.arm.move_home(time_ms=2000)
        #sleep(1.8)  # Reduced from 2.0 since movement is 2000ms

if __name__ == "__main__":
    main() 