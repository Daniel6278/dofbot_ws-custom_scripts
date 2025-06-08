#!/usr/bin/env python3
# place_three_blocks.py
import time
from time import sleep
import os
from arm_controller import ArmController
import argparse

class BlockMover:
    def __init__(self):
        self.arm = ArmController()
        self.current_pid = os.getpid()
        print(f"Current PID: {self.current_pid}")

    def close_gripper(self):
        self.arm.move_wrist_gripper(gripper_angle=135, time_ms=1000)
        
    def open_gripper(self):
        self.arm.move_wrist_gripper(gripper_angle=30, time_ms=1500)


    
    def reach_down_blue(self): #{1: 146, 2: 46, 3: 41, 4: 24, 5: 66, 6: 124}
        current = self.arm.read_current_state()
        self.arm.move_all_joints({1: 145, 2: 50, 3: 40, 4: 25, 5: 66, 6: current[6]}, time_ms=2000)
    
    def reach_down_red(self): #{1: 25, 2: 50, 3: 30, 4: 33, 5: 110, 6: 108}
        current = self.arm.read_current_state()
        self.arm.move_all_joints({1: 25, 2: 55, 3: 30, 4: 30, 5: 110, 6: current[6]}, time_ms=2000)
    
    def reach_down_green(self):
        current = self.arm.read_current_state()
        self.arm.move_arm_only({1: 179, 2: 53, 3: 37, 4: 16}, time_ms=2000)
    
    def reach_down_yellow(self):
        current = self.arm.read_current_state()
        self.arm.move_arm_only({1: 0, 2: 53, 3: 28, 4: 22}, time_ms=2000)
        
    def reach_down_left(self): #{1: 129, 2: 28, 3: 90, 4: 35, 5: 133, 6: 126}
                                #{1: 128, 2: 32, 3: 47, 4: 48, 5: 120, 6: 31}
        current = self.arm.read_current_state()
        self.arm.move_all_joints({1: 130, 2: 36, 3: 50, 4: 45, 5: 125, 6: current[6]}, time_ms=3000)
        
    def reach_down_middle(self): #{1: 88, 2: 44, 3: 47, 4: 22, 5: 90, 6: 128}
        self.arm.move_arm_only({1: 90, 2: 45, 3: 45, 4: 20}, time_ms=2000)
        
    def reach_down_right(self): #{1: 44, 2: 33, 3: 62, 4: 27, 5: 45, 6: 5}
        current = self.arm.read_current_state()
        self.arm.move_all_joints({1: 44, 2: 36, 3: 60, 4: 30, 5: 45, 6: current[6]}, time_ms=3000)

    def reach_up(self):
        current = self.arm.read_current_state()
        self.arm.move_arm_only({
            1: current[1],  # Maintain current rotation
            2: 160,         # Lift arm angles
            3: 20,
            4: 0
        }, time_ms=2000)


    
    def set_left(self):
        print("Placing into left position")
        current = self.arm.read_current_state()
        #self.arm.move_single_joint(1, 135, time_ms=2000)
        sleep(2.0)
        self.arm.move_all_joints({1: 129, 2: 37, 3: 58, 4: 27, 5: 128, 6: current[6]}, time_ms=2000)
        sleep(2.0)
        self.open_gripper()
        sleep(1.5)
        self.reach_up()
        sleep(2.0)
        self.arm.move_home(time_ms=2000)
        sleep(2.0)

    def set_middle(self):
        print("Placing into middle position")
        current = self.arm.read_current_state()
        #self.arm.move_single_joint(1, 90, time_ms=2000)
        sleep(2.0)
        self.arm.move_all_joints({1: 90, 2: 65, 3: 20, 4: 30, 5: 90, 6: current[6]}, time_ms=2000)
        sleep(2.0)
        self.open_gripper()
        sleep(1.5)
        self.reach_up()
        sleep(2.0)
        self.arm.move_home(time_ms=2000)
        sleep(2.0)
        
    def set_right(self):
        print("Placing into right position")
        current = self.arm.read_current_state()
        #self.arm.move_single_joint(1, 45, time_ms=2000)
        sleep(2.0)
        self.arm.move_all_joints({1: 44, 2: 50, 3: 33, 4: 40, 5: 51, 6: current[6]}, time_ms=2000)
        sleep(2.0)
        self.open_gripper()
        sleep(1.5)
        self.reach_up()
        sleep(2.0)
        self.arm.move_home(time_ms=2000)
        sleep(2.0)

    def set_left2(self):
        print("Placing into left2 position")
        current = self.arm.read_current_state()
        self.arm.move_single_joint(1, 135, time_ms=2000)
        sleep(2.0)
        self.arm.move_all_joints({1: 144, 2: 50, 3: 40, 4: 20, 5: 155, 6: current[6]}, time_ms=2000)
        sleep(2.0)
        self.open_gripper()
        sleep(1.5)
        self.reach_up()
        sleep(2.0)
        self.arm.move_home(time_ms=2000)
        sleep(2.0)

    def set_middle2(self): #{1: 91, 2: 88, 3: 2, 4: 2, 5: 87, 6: 125}
        print("Placing into middle2 position")
        current = self.arm.read_current_state()
        self.arm.move_single_joint(1, 90, time_ms=2000)
        sleep(2.0)
        self.arm.move_all_joints({1: 90, 2: 85, 3: 5, 4: 5, 5: 90, 6: current[6]}, time_ms=2000)
        sleep(2.0)
        self.open_gripper()
        sleep(1.5)
        self.reach_up()
        sleep(2.0)
        self.arm.move_home(time_ms=2000)
        sleep(2.0)
        
    def set_right2(self):
        print("Placing into right2 position")
        current = self.arm.read_current_state()
        self.arm.move_single_joint(1, 45, time_ms=2000)
        sleep(2.0)
        self.arm.move_all_joints({1: 26, 2: 50, 3: 40, 4: 20, 5: 35, 6: current[6]}, time_ms=2000)
        sleep(2.0)
        self.open_gripper()
        sleep(1.5)
        self.reach_up()
        sleep(2.0)
        self.arm.move_home(time_ms=2000)
        sleep(2.0)

    def move_block(self, color, location):
        """Move a block of specified color to the specified location"""
        print(f"Moving {color} block to {location}")
        
        # Set rotation based on color
        color_angles = {
            "red": 30,
            "green": 180,
            "blue": 150,
            "yellow": 0
        }
        
        if color not in color_angles:
            raise ValueError(f"Invalid color: {color}. Must be one of: red, green, blue, yellow")
            
        # Move to color position and pick up
        self.arm.move_single_joint(1, color_angles[color], time_ms=2000)
        sleep(2.0)
        self.open_gripper()
        sleep(1.5)
        
        # Pick up block based on color
        if color == "red":
            self.reach_down_red()
        elif color == "green":
            self.reach_down_green()
        elif color == "blue":
            self.reach_down_blue()
        elif color == "yellow":
            self.reach_down_yellow()
            
        sleep(2.0)
        self.close_gripper()
        sleep(1.0)
        self.reach_up()
        sleep(2.0)
        
        # Place block
        if location == "left":
            self.set_left()
        elif location == "middle":
            self.set_middle()
        elif location == "right":
            self.set_right()
            
    def move_left_back(self):
        self.arm.move_single_joint(1, 130, time_ms=2000)
        sleep(2.0)
        self.open_gripper()
        sleep(1.5)
        self.reach_down_left()
        sleep(3.0)
        self.close_gripper()
        sleep(1.0)
        self.reach_up()
        sleep(2.0)
        self.set_left2()
        sleep(2.0)
        
    def move_middle_back(self):
        self.arm.move_single_joint(1, 90, time_ms=2000)
        sleep(2.0)
        self.open_gripper()
        sleep(1.5)
        self.reach_down_middle()
        sleep(2.0)
        self.close_gripper()
        sleep(1.0)
        self.reach_up()
        sleep(2.0)
        self.set_middle2()
        sleep(2.0)

    def move_right_back(self):
        self.arm.move_single_joint(1, 45, time_ms=2000)
        sleep(2.0)
        self.open_gripper()
        sleep(1.5)
        self.reach_down_right()
        sleep(3.0)
        self.close_gripper()
        sleep(1.0)
        self.reach_up()
        sleep(2.0)
        self.set_right2()
        sleep(2.0)

    def move_block_to_second_position(self, location):
        """Move a block from its first position to its second position"""
        if location == "left":
            self.move_left_back()
        elif location == "middle":
            self.move_middle_back()
        elif location == "right":
            self.move_right_back()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Place three blocks in specified positions')
    parser.add_argument('left_color', choices=['red', 'green', 'blue', 'yellow'], help='Color for left position')
    parser.add_argument('middle_color', choices=['red', 'green', 'blue', 'yellow'], help='Color for middle position')
    parser.add_argument('right_color', choices=['red', 'green', 'blue', 'yellow'], help='Color for right position')
    args = parser.parse_args()
    
    mover = BlockMover()
    try:
        # Ensure arm starts at an appropriate position
        mover.arm.move_home(time_ms=2000)
        sleep(2)
        
        """mover.reach_down_right()
        sleep(3)
        mover.close_gripper()
        sleep(3)
        return 0"""
    
        # Place blocks in first positions
        print("\nPlacing blocks in first positions...")
        mover.move_block(args.left_color, "left")
        mover.move_block(args.middle_color, "middle")
        mover.move_block(args.right_color, "right")
        
        # Wait for user input
        input("\nPress Enter to move blocks to second positions...")
        
        # Move blocks to second positions
        print("\nMoving blocks to second positions...")
        mover.move_block_to_second_position("left")
        mover.move_block_to_second_position("middle")
        mover.move_block_to_second_position("right")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 