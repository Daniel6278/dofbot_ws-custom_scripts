#!/usr/bin/env python3
from Arm_Lib import Arm_Device
from time import sleep
from arm_controller import ArmController

""" 
A Demo of arm_controller class with custom functions making use of the
functions defined inside arm_controller
"""

class ArmActions:
    def __init__(self, controller: ArmController):
        self.controller = controller
        
    def inspection_scan(self):
        for angle in range(0, 195, 15):
            self.controller.move_arm_only({1: angle}, 250)
            sleep(0.5)
            print("Current Servo State:",self.controller.read_current_state())
            sleep(0.5)
    def grab_position(self):
        """Close gripper while maintaining current positions"""
        current = self.controller.read_current_state()
        self.controller.move_wrist_gripper(
            wrist_angle=current[5],
            gripper_angle=30,
            time_ms=500
        )
    def ready_position(self):
        #move to pre-grab position
        current = self.controller.read_current_state()
        self.controller.move_arm_only({1: 90, 2: 160, 3: 20, 4: 0}, time_ms=1500)
    
    def rotate_to_middle(self): #rotate arm towards the middle square
        current = self.controller.read_current_state()
        self.controller.move_single_joint(1,90,1500)
    def rotate_to_blue(self): #rotate arm towards the blue square
        current = self.controller.read_current_state()
        self.controller.move_single_joint(1,150,1500)
    def rotate_to_green(self): #rotate arm towards the green square
        current = self.controller.read_current_state()
        self.controller.move_single_joint(1,180,1500)
    def rotate_to_red(self): #rotate arm towards the red square
        current = self.controller.read_current_state()
        self.controller.move_single_joint(1,30,1500)
    def rotate_to_yellow(self): #rotate arm towards the yellow square
        current = self.controller.read_current_state()
        self.controller.move_single_joint(1,0,1500)
    
    def reach_down_middle(self): #reach down to middle space
        current = self.controller.read_current_state()
        self.controller.move_arm_only({1: 90, 2: 40, 3: 60, 4: 0}, time_ms=1500)
    def reach_down_blue(self): #reach down to blue space
        current = self.controller.read_current_state()
        self.controller.move_arm_only({1: 150, 2: 40, 3: 60, 4: 0}, time_ms=1500)
    def reach_down_green(self): #reach down to green space
        current = self.controller.read_current_state()
        self.controller.move_arm_only({1: 180, 2: 40, 3: 60, 4: 0}, time_ms=1500)
    def reach_down_red(self): #reach down to red space
        current = self.controller.read_current_state()
        self.controller.move_arm_only({1: 30, 2: 40, 3: 60, 4: 0}, time_ms=1500)
    def reach_down_yellow(self): #reach down to yellow space
        current = self.controller.read_current_state()
        self.controller.move_arm_only({1: 0, 2: 40, 3: 60, 4: 0}, time_ms=1500)

    def reach_up(self): #reach back up to ready position (for use after reach_down)
        current = self.controller.read_current_state()
        base_rotation = current[1]
        self.controller.move_arm_only({1: base_rotation, 2: 160, 3: 20, 4: 0}, time_ms=1500)

    def close_gripper(self):
        #close gripper
        current = self.controller.read_current_state()
        self.controller.move_wrist_gripper(
            wrist_angle=current[5],
            gripper_angle=135,
            time_ms=500
        )
        
    def open_gripper(self):
        #close gripper
        current = self.controller.read_current_state()
        self.controller.move_wrist_gripper(
            wrist_angle=current[5],
            gripper_angle=30,
            time_ms=500
        )

if __name__ == "__main__":
    # Initialize hardware connection
    arm_controller = ArmController()
    actions = ArmActions(arm_controller)
    
    try:
        #function call directly from arm_controller.py
        #print("Starting sequence, moving to home position...")
        #actions.controller.move_home(2000)
        #sleep(2)
        
        #function call from ArmActions class declared above
        #print("Starting inspection scan...")
        #actions.inspection_scan()
        #sleep(1)
        #input("Press enter to continue")
        
        print("Moving to ready position...")
        actions.ready_position()
        sleep(1.5)

        print("Moving to grab middle position...")
        actions.reach_down_middle()
        sleep(1.5)

        print("Closing gripper...")
        actions.close_gripper()
        sleep(1.5)

        print("Moving back to ready position...")
        actions.reach_up()
        sleep(1.5)
        actions.ready_position()
        sleep(1.5)
        
        print("Moving to grab blue position...")
        actions.rotate_to_red()
        sleep(1.5)
        actions.reach_down_red()
        sleep(1.5)
        
        print("Opening gripper...")
        actions.open_gripper()
        sleep(1.5)

        print("Returning to ready position...")
        actions.reach_up()
        sleep(1.5)
        actions.ready_position()
        sleep(1.5)
        
        print("Returning to home position...")
        actions.controller.move_home()
        sleep(1.5)

    except KeyboardInterrupt:
        print("\nEmergency shutdown initiated!")
        actions.controller.move_home(2000)
        sleep(2.5)
        print("Arm safely homed.")
    
    finally:
        print(actions.controller.set_torque(False))
        print("sequence completed.")