#!/usr/bin/env python3
from Arm_Lib import Arm_Device
from time import sleep
from arm_controller import ArmController

"""
basic outline for a custom script using arm_controller.py
"""

class ArmActions:
    def __init__(self, controller: ArmController):
        self.controller = controller
    #DEFINE FILE-SPECIFIC FUNCTIONS HERE

if __name__ == "__main__":
    # Initialize hardware connection
    arm_controller = ArmController()
    actions = ArmActions(arm_controller)

    try:
        #ROBOT ACTIONS HERE
    except KeyboardInterrupt:
        print("\nEmergency shutdown initiated!")
        actions.controller.move_home(2000)
        sleep(2.5)
        print("Arm safely homed.")
        
    finally:
        print(actions.controller.set_torque(False))
        print("Script completed.")