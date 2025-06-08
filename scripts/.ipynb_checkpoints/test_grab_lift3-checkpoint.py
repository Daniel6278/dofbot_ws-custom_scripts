#!/usr/bin/env python3
from Arm_Lib import Arm_Device
from time import sleep

from arm_controller import ArmController

""" A simple demo of the arm_controller class """

def main():
    controller = ArmController()
    print("Enabling Torque... ",controller.set_torque(True))
    
    print("Moving arm to pick position...")
    controller.move_arm_only({
        1: 45,   # Base rotation
        2: 120,  # Shoulder
        3: 60,   # Elbow
        4: 100   # Wrist pitch
    }, time_ms=1500)

    sleep(2)

    print("Adjusting wrist and gripper only...")
    controller.move_wrist_gripper(
        wrist_angle=180,  # Rotate wrist
        gripper_angle=30, # Open gripper
        time_ms=1000
    )
    
    sleep(1)

    print("Performing complex maneuver...")
    # Move arm while simultaneously closing gripper
    controller.move_arm_only({1: 90, 2: 80, 3: 100, 4: 85})
    controller.move_wrist_gripper(gripper_angle=70)

    current_state = controller.read_current_state()
    print(f"Current wrist rotation: {current_state[5]}°")

    print("Resetting to default position...")
    controller.move_home(time_ms=2000)
    sleep(2.1)
    print("Turning off torque...",controller.set_torque(False))
    
if __name__ == "__main__":
    main()

