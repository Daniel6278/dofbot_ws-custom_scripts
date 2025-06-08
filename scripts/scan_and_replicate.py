#!/usr/bin/env python3
from Arm_Lib import Arm_Device
from time import sleep
from arm_controller import ArmController
from get_color import get_color_percentages
from camera_test import take_photo
import argparse

"""
basic outline for a custom script using arm_controller.py
"""
"""
Block scan position servo angles: {1: 90, 2: 65, 3: 70, 4: 90}
"""

class ArmActions:
    def __init__(self, controller: ArmController):
        self.controller = controller
    #DEFINE FILE-SPECIFIC FUNCTIONS HERE

    #arm position functions
    def scan_position(self): #prepare for block scan
        """Universal reach down using current base rotation"""
        current = self.controller.read_current_state()
        self.controller.move_arm_only({
            1: 90,
            2: 75,
            3: 20,
            4: 0
        }, time_ms=3000)
        
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
        
    def ready_position(self):
        self.controller.move_arm_only({1: 90, 2: 160, 3: 20, 4: 0}, time_ms=1500)
    
    #arm rotation functions
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
        
    #gripper control functions    
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
    
    #move to scanned block color location to grab similar block
    def grab_block(self, block_color): #block_color: scanned block color
        # Initialize sequence
        self.open_gripper()
        sleep(1.5)
        self.ready_position()
        sleep(1.5)
        
        # Pick sequence
        self.rotate_to_color(block_color)
        sleep(1.5)
        self.reach_down()
        sleep(1.5)
        self.close_gripper()
        sleep(1.5)
        self.reach_up()
        sleep(1.5)

        #place sequence
        self.rotate_to_middle()
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
    parser = argparse.ArgumentParser(description='Get percentage of red, blue, green, and yellow in an image.')
    parser.add_argument('--show', action='store_true', help='Show the image with detection visualization')
    parser.add_argument('--debug', action='store_true', help='Show detailed debug information')
    parser.add_argument('--square', action='store_true', help='Analyze only the center square of the image')
    parser.add_argument('--threshold', type=float, default=0.0, help='Minimum percentage to report a color (default: 0.0)')
    parser.add_argument('--square-size', type=int, default=120, help='Size of the center square in pixels (default: 120) (max: 480)')
    # need to implement check to make sure square-size is not too big
    args = parser.parse_args()

    # Initialize hardware connection
    arm_controller = ArmController()
    actions = ArmActions(arm_controller)

    try:
        #move to scan block position
        print("Moving to scan position")
        actions.scan_position() #move to scan position
        sleep(3.1)

        #scan block
        input("Press enter to scan block...")
        print("scanning...")
        take_photo()

        #extract highest confidence block color
        img = "test1.jpg"
        color_percentages = get_color_percentages(
            img,
            args.square,
            args.debug,
            args.show,
            args.threshold,
            args.square_size
        )
        found_block_color = max(color_percentages, key=color_percentages.get)
        
        print(f"Colors hashmap:{color_percentages}")
        print(f"Highest confidence color: {found_block_color}")
        
        ##Temporary##
        scanned_block_color = "blue"
        sleep(4)
        
        print(f"Scan done.")
        input(f"Press enter to pick up {found_block_color} block, ctrl+c to exit program...")
        
        #grab similar block/place in center
        actions.grab_block(found_block_color)
        
    except KeyboardInterrupt:
        print("\nEmergency shutdown initiated!")
        actions.controller.move_home(2000)
        sleep(2.5)
        print("Arm safely homed.")
        
    finally:
        print(actions.controller.set_torque(False))
        print("Script completed.")