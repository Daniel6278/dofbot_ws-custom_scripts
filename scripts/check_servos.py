#!/usr/bin/env python3
from Arm_Lib import Arm_Device
import time

# Initialize the arm
arm = Arm_Device()

def check_servos():
    print("=== DOFBOT Servo Diagnostic Test ===")
    
    # First disable then enable torque to ensure clean state
    print("\n1. Resetting torque state...")
    arm.Arm_serial_set_torque(0)
    time.sleep(1)
    print("Torque disabled - servos should now move freely by hand")
    input("Try moving any servo by hand - it should move freely. Press Enter to continue...")
    
    print("\nEnabling torque...")
    arm.Arm_serial_set_torque(1)
    time.sleep(1)
    print("Torque enabled - servos should now resist movement")
    input("Try moving any servo by hand - it should resist movement. Press Enter to continue...")
    
    # Test each servo with distinct movements
    print("\n2. Testing each servo individually...")
    for servo_id in range(1, 7):
        print(f"\nTesting Servo {servo_id}:")
        try:
            # Move to 45 degrees first
            print(f"  Moving servo {servo_id} to 45 degrees...")
            arm.Arm_serial_servo_write(servo_id, 45, 1000)
            time.sleep(2)
            input(f"Did servo {servo_id} move to ~45 degrees? Press Enter to continue...")
            
            # Then to 135 degrees
            print(f"  Moving servo {servo_id} to 135 degrees...")
            arm.Arm_serial_servo_write(servo_id, 135, 1000)
            time.sleep(2)
            input(f"Did servo {servo_id} move to ~135 degrees? Press Enter to continue...")
            
            # Return to 90 degrees
            print(f"  Moving servo {servo_id} back to 90 degrees...")
            arm.Arm_serial_servo_write(servo_id, 90, 1000)
            time.sleep(2)
            
        except Exception as e:
            print(f"  - Error testing servo {servo_id}: {e}")
    
    print("\n3. Testing power supply with sequential movements...")
    try:
        # Test sequence
        print("Moving to home position...")
        arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
        time.sleep(2)
        
        print("Moving to extended position...")
        arm.Arm_serial_servo_write6(90, 45, 45, 90, 90, 90, 1000)
        time.sleep(2)
        
        print("Moving back to home position...")
        arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
        time.sleep(2)
        
    except Exception as e:
        print(f"  - Error during movement test: {e}")

if __name__ == "__main__":
    try:
        check_servos()
    finally:
        # Always re-enable torque when done
        arm.Arm_serial_set_torque(1)
        print("\nDiagnostic complete. Torque enabled.")
        print("\nIf no servos moved during this test:")
        print("1. Check if the power supply is properly connected and LED indicators are on")
        print("2. Verify all servo cables are properly connected")
        print("3. Try power cycling the robot")
        print("4. Check USB connection")
        print("\nIf some servos moved but others didn't:")
        print("1. Check the cables for the non-moving servos")
        print("2. Try recalibrating those specific servos") 