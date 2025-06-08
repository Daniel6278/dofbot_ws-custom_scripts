#!/usr/bin/env python3
# arm_controller.py
from Arm_Lib import Arm_Device
import time

class ArmController:
    def __init__(self, ic2_bus=7):
        self.arm = Arm_Device(ic2_bus)
        self.arm.Arm_serial_set_torque(1)  # Enable torque on initialization
        
        self.current_state = {
            1: 90,  # Base rotation
            2: 90,  # Arm bend low servo
            3: 90,  # Arm bend med servo
            4: 90,  # Arm bend high servo
            5: 90,  # Wrist rotation (0-270 degrees)
            6: 90,  # Gripper
        }
        
    def set_torque(self, enable=True):
        #Control servo torque state (enabled/disabled)
        self.arm.Arm_serial_set_torque(1 if enable else 0)
        return f"Torque {'enabled' if enable else 'disabled'}"
        
    def read_current_state(self): 
        #return an array of the current angle of all servos (1-7)
        for joint_id in range(1,7):
            angle = self.arm.Arm_serial_servo_read(joint_id)
            if angle is not None:
                self.current_state[joint_id] = angle
        return self.current_state
        
    def move_all_joints(self, positions, time_ms):
        """
        Move all joints to specified positions.
        Args:
            positions: Dict mapping joint IDs (1-6) to angles
            time_ms: Movement time in milliseconds (recommended: 1500-2500ms for smooth movement)
        """
        # Read actual current state of all servos
        current_state = self.read_current_state()
        
        if isinstance(positions, dict): #extract angles from dict.
            angles = [positions.get(i, current_state[i]) for i in range(1, 7)]
        else:
            angles = positions
            
        print(f"Current state before movement: {current_state}")
        print(f"Target angles: {angles}")
        self.arm.Arm_serial_servo_write6_array(angles, time_ms) #move servos

        # Wait for movement to complete
        time.sleep(time_ms / 1000.0)  # Convert ms to seconds
        
        # Update current state with actual values
        for joint_id, angle in positions.items():
            self.current_state[joint_id] = angle

    def move_arm_only(self, arm_positions, time_ms):
        """
        Move arm joints (servos 1-4), maintain current state of wrist/gripper.
        Args:
            arm_positions: Dict mapping joint IDs (1-4) to angles
            time_ms: Movement time in milliseconds (recommended: 1500-2500ms for smooth movement)
        """
        # Read actual current state of all servos
        current_state = self.read_current_state()
        
        positions = current_state.copy() #create new dict using current state
        for joint_id, angle in arm_positions.items(): #update servo states using arm_positions
            if 1 <= joint_id <= 4: #update positions dict. using arm_positions
                positions[joint_id] = angle
            else:
                raise ValueError(f"move_arm_only only accepts joint IDs 1-4, got {joint_id}")
        
        angles = [positions[i] for i in range(1, 7)] #extract angles from dict
        print(f"Current state before movement: {current_state}")
        print(f"Target angles: {angles}")
        self.arm.Arm_serial_servo_write6_array(angles, time_ms) #move servos

        # Wait for movement to complete
        time.sleep(time_ms / 1000.0)  # Convert ms to seconds

        # Update current state with actual values
        for joint_id, angle in arm_positions.items():
            self.current_state[joint_id] = angle
            
    def move_wrist_gripper(self, wrist_angle=None, gripper_angle=None, time_ms=1000):
        """
        Move wrist/gripper joints (servos 5-6), maintain current state of arm joints.
        Args:
            wrist_angle: Angle for wrist rotation (0-270 degrees)
            gripper_angle: Angle for gripper (0-180 degrees)
            time_ms: Movement time in milliseconds (recommended: 1000-1500ms for smooth movement)
        """
        # Read actual current state of all servos
        current_state = self.read_current_state()
        
        # Create new positions dict with current state
        positions = current_state.copy()
        
        if wrist_angle is not None: #update dict. using parameters
            if 0 <= wrist_angle <= 270:
                positions[5] = wrist_angle
            else: #error trap
                raise ValueError(f"Wrist angle must be between 0 and 270, got {wrist_angle}")
        if gripper_angle is not None: #update dict. using parameters
            if 0 <= gripper_angle <= 180:
                positions[6] = gripper_angle
            else: #error trap
                raise ValueError(f"Gripper angle must be between 0 and 180, got {gripper_angle}")
        
        angles = [positions[i] for i in range(1, 7)] #extract angles from dict
        print(f"Current state before movement: {current_state}")
        print(f"Target angles: {angles}")
        self.arm.Arm_serial_servo_write6_array(angles, time_ms) #move servos

        # Wait for movement to complete
        time.sleep(time_ms / 1000.0)  # Convert ms to seconds

        # Update current state with actual values
        if wrist_angle is not None:
            self.current_state[5] = wrist_angle
        if gripper_angle is not None:
            self.current_state[6] = gripper_angle
            
    def move_single_joint(self, joint_id, angle, time_ms):
        """
        Move specified joint to specified angle, maintain current state of all other joints.
        Args:
            joint_id: Joint ID (1-6)
            angle: Target angle
            time_ms: Movement time in milliseconds (recommended: 1000-1500ms for smooth movement)
        """
        if 1 <= joint_id <= 6:
            # Use the direct single-servo control function
            self.arm.Arm_serial_servo_write(joint_id, angle, time_ms) #move servo
            
            # Wait for movement to complete
            time.sleep(time_ms / 1000.0)  # Convert ms to seconds
            
            self.current_state[joint_id] = angle #update current state
        else: #error trap
            raise ValueError(f"Invalid joint ID: {joint_id}. Must be between 1 and 6.")
            
    def move_home(self, time_ms=2000):
        """
        Return robot to home state (erect).
        Args:
            time_ms: Movement time in milliseconds (recommended: 1500-2500ms for smooth movement)
        """
        default_positions = {
            1: 90,  # Base rotation
            2: 160,  # Shoulder
            3: 20,  # Elbow
            4: 0,  # Wrist pitch
            5: 90,  # Wrist rotation
            6: 0,  # Gripper
        }
        
        # Convert positions to array in correct order
        angles = [default_positions[i] for i in range(1, 7)]
        
        # Move directly to home position
        self.arm.Arm_serial_servo_write6_array(angles, time_ms)
        
        # Wait for movement to complete
        time.sleep(time_ms / 1000.0)
        
        # Update current state with home positions
        self.current_state = default_positions.copy()