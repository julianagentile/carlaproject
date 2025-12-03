"""
Vehicle control logic
"""
import carla
import numpy as np
from utils.helpers import calculate_angle_to_waypoint

class VehicleController:
    def __init__(self, vehicle, config):
        self.vehicle = vehicle
        self.config = config
        
        # State
        self.paused = False
        self.resume_time = 0
        
        print("✓ Vehicle controller initialized")
    
    def calculate_steering(self, target_waypoint):
        """Calculate steering angle to reach waypoint"""
        angle = calculate_angle_to_waypoint(
            self.vehicle.get_transform(), 
            target_waypoint
        )
        
        # Clamp angle
        angle = np.clip(angle, -self.config.MAX_STEER_DEGREES, 
                       self.config.MAX_STEER_DEGREES)
        
        # Normalize to [-1, 1]
        steer = angle / self.config.MAX_STEER_DEGREES
        
        return steer
    
    def calculate_speed_control(self, current_speed, target_speed):
        """Calculate throttle and brake based on target speed"""
        speed_diff = target_speed - current_speed
        
        # PID-like control
        if abs(speed_diff) < 1.0:
            throttle = 0.5
            brake = 0.0
        elif speed_diff > 0:
            # Need to accelerate
            throttle = min(0.75, 0.5 + speed_diff * 0.05)
            brake = 0.0
        else:
            # Need to slow down
            if abs(speed_diff) > 10:
                throttle = 0.0
                brake = 0.5
            else:
                throttle = max(0.2, 0.5 + speed_diff * 0.05)
                brake = 0.0
        
        return throttle, brake
    
    def emergency_stop(self):
        """Apply emergency stop"""
        control = carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0
        )
        self.vehicle.apply_control(control)
    
    def apply_control(self, throttle, steer, brake):
        """Apply control to vehicle"""
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake)
        )
        self.vehicle.apply_control(control)