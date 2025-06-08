#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
from dofbot_msgs.msg import ArmJoint
from Arm_Lib import Arm_Device
import sys
import signal
import atexit
import logging

class Joint:
    def __init__(self, name, joint_id, min_angle, max_angle, default_angle=90):
        self.name = name
        self.id = joint_id
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.default_angle = default_angle
        self.current_angle = default_angle
        self.logger = logging.getLogger(f'Joint_{name}')
        
    def move(self, angle, duration=1.0):
        """Move the joint to a specific angle"""
        try:
            # Validate angle
            if angle < self.min_angle or angle > self.max_angle:
                self.logger.error(f"Angle {angle}° is outside limits ({self.min_angle}° to {self.max_angle}°)")
                return False
            
            # Convert duration to milliseconds
            speed = int(duration * 1000)
            
            # Move the joint
            self.current_angle = angle
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to move joint: {e}")
            return False
    
    def get_current_angle(self):
        """Get the current angle of the joint"""
        return self.current_angle
    
    def reset(self):
        """Reset the joint to its default position"""
        return self.move(self.default_angle)

class BaseJoint(Joint):
    def __init__(self):
        super().__init__('base', 1, -90, 90)

class ShoulderJoint(Joint):
    def __init__(self):
        super().__init__('shoulder', 2, 0, 180)

class ElbowJoint(Joint):
    def __init__(self):
        super().__init__('elbow', 3, 0, 180)

class WristJoint(Joint):
    def __init__(self):
        super().__init__('wrist', 4, -90, 90)

class GripperJoint(Joint):
    def __init__(self):
        super().__init__('gripper', 5, 0, 180)
        self.open_angle = 180
        self.closed_angle = 0
    
    def open(self, duration=1.0):
        """Open the gripper"""
        return self.move(self.open_angle, duration)
    
    def close(self, duration=1.0):
        """Close the gripper"""
        return self.move(self.closed_angle, duration)

class DofbotControl:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('dofbot_control', anonymous=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('DofbotControl')
        
        # Initialize the Dofbot arm
        self.arm = Arm_Device()
        
        # Flag to prevent multiple cleanups
        self.cleanup_done = False
        
        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, lambda signum, frame: self.cleanup(True))
        signal.signal(signal.SIGTERM, lambda signum, frame: self.cleanup(True))
        
        # Ensure torque is enabled at start
        self.arm.Arm_serial_set_torque(1)
        time.sleep(0.1)
        
        # Initialize joints
        self.base = BaseJoint()
        self.shoulder = ShoulderJoint()
        self.elbow = ElbowJoint()
        self.wrist = WristJoint()
        self.gripper = GripperJoint()
        
        # Set initial speed
        self.speed = 1000  # Default speed (time in ms)
        
        # Setup publisher
        self.arm_pub = rospy.Publisher('/arm_joints', ArmJoint, queue_size=10)
        
        self.logger.info("Dofbot Control initialized successfully")

    def release_bus(self):
        """Explicitly release the I2C bus"""
        try:
            if hasattr(self.arm, 'bus'):
                self.arm.bus.close()
            self.logger.info("I2C bus released")
        except Exception as e:
            self.logger.error(f"Error releasing I2C bus: {e}")

    def cleanup(self, signal_received=False):
        """Enhanced cleanup function with guard against multiple calls"""
        if self.cleanup_done:
            return
        
        try:
            self.logger.info("Starting cleanup...")
            # First move to home position
            self.home_position()
            time.sleep(2)  # Wait for movement
            
            # Then disable torque
            self.arm.Arm_serial_set_torque(0)
            self.logger.info("Torque disabled")
            
            # Finally release the I2C bus
            self.release_bus()
            
            self.cleanup_done = True  # Mark cleanup as done
            self.logger.info("Cleanup completed")
            
            if signal_received:
                sys.exit(0)
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            self.cleanup_done = True  # Mark as done even if there was an error
            if signal_received:
                sys.exit(1)

    def move_joint(self, joint, angle, duration=1.0):
        """
        Move a single joint to a specific angle
        
        Args:
            joint: Joint object to move
            angle (float): Target angle in degrees
            duration (float): Time to complete the movement in seconds
        """
        try:
            # Validate angle
            if angle < joint.min_angle or angle > joint.max_angle:
                self.logger.error(f"Angle {angle}° is outside limits for {joint.name} ({joint.min_angle}° to {joint.max_angle}°)")
                return False
            
            # Convert duration to milliseconds
            speed = int(duration * 1000)
            
            # Move the joint
            self.arm.Arm_serial_servo_write(joint.id, angle, speed)
            joint.current_angle = angle
            self.logger.info(f"Moved {joint.name} to {angle}°")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to move {joint.name}: {e}")
            return False

    def move_all_joints(self, angles, duration=1.0):
        """
        Move all joints to specified angles simultaneously
        
        Args:
            angles (dict): Dictionary of joint names and their target angles
            duration (float): Time to complete the movement in seconds
        """
        try:
            # Get angles in the correct order, use current position if not specified
            s1 = angles.get('base', self.base.get_current_angle())
            s2 = angles.get('shoulder', self.shoulder.get_current_angle())
            s3 = angles.get('elbow', self.elbow.get_current_angle())
            s4 = angles.get('wrist', self.wrist.get_current_angle())
            s5 = angles.get('gripper', self.gripper.get_current_angle())
            s6 = 90  # We don't use the 6th servo, set to neutral position
            
            # Convert duration to milliseconds
            time_ms = int(duration * 1000)
            
            # Move all servos
            self.arm.Arm_serial_servo_write6(s1, s2, s3, s4, s5, s6, time_ms)
            
            # Update current angles
            self.base.current_angle = s1
            self.shoulder.current_angle = s2
            self.elbow.current_angle = s3
            self.wrist.current_angle = s4
            self.gripper.current_angle = s5
            
            time.sleep(duration)  # Wait for movement to complete
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to move joints: {e}")
            return False

    def home_position(self):
        """Move all joints to their home positions"""
        return self.move_all_joints({
            'base': self.base.default_angle,
            'shoulder': self.shoulder.default_angle,
            'elbow': self.elbow.default_angle,
            'wrist': self.wrist.default_angle,
            'gripper': self.gripper.default_angle
        }, 2.0)

    def pick_and_place(self):
        """Perform a pick and place operation"""
        try:
            # Reach down
            self.logger.info("Reach Down")
            self.move_all_joints({
                'base': 90,
                'shoulder': 90,
                'elbow': 45,
                'wrist': 45,
                'gripper': 90
            }, 0.6)
            
            # Grab item
            self.logger.info("Grab")
            self.move_all_joints({
                'base': 90,
                'shoulder': 45,
                'elbow': 45,
                'wrist': 45,
                'gripper': 180
            }, 0.6)
            
            # Return to home position
            self.logger.info("Lift")
            self.move_all_joints({
                'base': 90,
                'shoulder': 90,
                'elbow': 90,
                'wrist': 0,
                'gripper': 90
            }, 0.6)
            
            self.logger.info("End of pick and place operation")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during pick and place: {e}")
            return False

def main():
    # Create robot control instance
    robot = DofbotControl()
    
    try:
        # Example 1: Move to home position
        
        robot.logger.info("Moving to home position...")
        robot.home_position()
        
        # Example 2: Move individual joints
        
        robot.logger.info("Moving individual joints...")
        robot.move_joint(robot.base, 45)  # Move base to 45 degrees
        robot.move_joint(robot.shoulder, 120)  # Move shoulder to 120 degrees
        robot.move_joint(robot.gripper, 180)  # Open gripper
        
        # Example 3: Perform pick and place
        
        # robot.logger.info("Performing pick and place operation...")
        # robot.pick_and_place()
        
        # Return to home position at the end
        robot.logger.info("Returning to home position...")
        robot.home_position()
        
    except Exception as e:
        robot.logger.error(f"An error occurred: {e}")
        raise  # Re-raise the exception to trigger cleanup
    finally:
        if not robot.cleanup_done:  # Only call cleanup if it hasn't been done yet
            robot.cleanup()

if __name__ == '__main__':
    main() 