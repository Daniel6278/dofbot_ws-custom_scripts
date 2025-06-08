#!/usr/bin/env python3
from Arm_Lib import Arm_Device
from time import sleep
import argparse
from arm_controller import ArmController

"""
Given two command line arguments, this script picks up the block at the position defined
by the first argument and places it down at the position defined by the second argument

usage: $ python3 pick_place_custom string1 string2
        - where string1 and string2 are colors: red, green, blue, yellow, white
"""

class ArmActions:
    def __init__(self, controller: ArmController):
        self.controller = controller
    
    # Auxiliary functions from separate file
    def ready_position(self):
        # Move to pre-grab position
        self.controller.move_arm_only({1: 90, 2: 160, 3: 20, 4: 0}, time_ms=1500)
    
    def rotate_to_middle(self):
        self.controller.move_single_joint(1, 90, 1500)
    
    def rotate_to_blue(self):
        self.controller.move_single_joint(1, 150, 1500)
    
    def rotate_to_green(self):
        self.controller.move_single_joint(1, 180, 1500)
    
    def rotate_to_red(self):
        self.controller.move_single_joint(1, 30, 1500)
    
    def rotate_to_yellow(self):
        self.controller.move_single_joint(1, 0, 1500)

    def reach_down(self):
        """Universal reach down using current base rotation"""
        current = self.controller.read_current_state()
        self.controller.move_arm_only({
            1: current[1],  # Maintain current rotation
            2: 40,          # Consistent arm angles for picking
            3: 60,
            4: 0
        }, time_ms=1500)

    def reach_up(self):
        """Return to ready position after picking/placing"""
        current = self.controller.read_current_state()
        self.controller.move_arm_only({
            1: current[1],  # Maintain current rotation
            2: 160,         # Lift arm angles
            3: 20,
            4: 0
        }, time_ms=1500)

    def close_gripper(self):
        self.controller.move_wrist_gripper(gripper_angle=135, time_ms=500)
        
    def open_gripper(self):
        self.controller.move_wrist_gripper(gripper_angle=30, time_ms=500)
    
    # Color handling functions
    def rotate_to_color(self, color):
        """Rotate base to specified color position"""
        color = color.lower()
        {
            'red': self.rotate_to_red,
            'green': self.rotate_to_green,
            'blue': self.rotate_to_blue,
            'yellow': self.rotate_to_yellow,
            'white': self.rotate_to_middle
        }[color]()
    
    def pick_and_place(self, source_color, destination_color):
        """Complete pick/place sequence with consolidated movements"""
        print(f"Moving block from {source_color} to {destination_color}")
        
        # Initialize sequence
        self.open_gripper()
        sleep(1.5)
        self.ready_position()
        sleep(1.5)
        
        # Pick sequence
        self.rotate_to_color(source_color)
        sleep(1.5)
        self.reach_down()
        sleep(1.5)
        self.close_gripper()
        sleep(1.5)
        self.reach_up()
        sleep(1.5)
        
        # Place sequence
        self.rotate_to_color(destination_color)
        sleep(1.5)
        self.reach_down()
        sleep(1.5)
        self.open_gripper()
        sleep(1.5)
        self.reach_up()
        sleep(1.5)
        
        # Return home
        self.controller.move_home(2000)
        sleep(2)
        print("Operation completed successfully")

if __name__ == "__main__":
    arm_controller = ArmController()
    actions = ArmActions(arm_controller)

    try:
        parser = argparse.ArgumentParser(description="Robot Arm Pick/Place Control")
        parser.add_argument('source', choices=['red', 'green', 'blue', 'yellow', 'white'])
        parser.add_argument('destination', choices=['red', 'green', 'blue', 'yellow', 'white'])
        args = parser.parse_args()
        
        actions.pick_and_place(args.source.lower(), args.destination.lower())
        
    except KeyboardInterrupt:
        print("\nEmergency shutdown!")
        actions.controller.move_home(2000)
        sleep(2.5)
        print("Arm safely homed.")
        
    finally:
        print(actions.controller.set_torque(False))
        print("Script terminated.")
