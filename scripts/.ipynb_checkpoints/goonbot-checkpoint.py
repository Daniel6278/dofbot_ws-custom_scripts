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
    def prepare(self):
        self.controller.move_arm_only({1: 90, 2: 110, 3: 50, 4: 30}, time_ms=2000)
        
    def goon(self, strokes=5):
        for i in range(strokes):
            self.controller.move_arm_only({1: 90, 2: 110, 3: 50, 4: 30}, time_ms=250)
            sleep(0.300)
            self.controller.move_arm_only({1: 90, 2: 105, 3: 45, 4: 25}, time_ms=250)
            sleep(0.300)

if __name__ == "__main__":
    # Initialize hardware connection
    arm_controller = ArmController()
    actions = ArmActions(arm_controller)

    try:
        #ROBOT ACTIONS HERE
        actions.prepare()
        sleep(2)
        actions.goon()
        sleep(5)
        
    except KeyboardInterrupt:
        print("\nEmergency shutdown initiated!")
        actions.controller.move_home(2000)
        sleep(2.5)
        print("Arm safely homed.")
        
    finally:
        print(actions.controller.set_torque(False))
        print("Goon completed.")