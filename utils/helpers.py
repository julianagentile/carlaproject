"""
Helper functions and utilities
"""
import numpy as np
import math

def calculate_angle_to_waypoint(vehicle_transform, waypoint):
    """Calculate angle between vehicle heading and waypoint"""
    vehicle_loc = vehicle_transform.location
    vehicle_fwd = vehicle_transform.get_forward_vector()
    wp_loc = waypoint.transform.location
    
    direction = np.array([wp_loc.x - vehicle_loc.x, wp_loc.y - vehicle_loc.y])
    if np.linalg.norm(direction) > 0:
        direction = direction / np.linalg.norm(direction)
    
    fwd = np.array([vehicle_fwd.x, vehicle_fwd.y])
    dot_product = np.dot(fwd, direction)
    angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
    
    cross_product = np.cross([vehicle_fwd.x, vehicle_fwd.y], direction)
    if cross_product < 0:
        angle = -angle
    
    return angle

def calculate_vehicle_speed(velocity):
    """Calculate vehicle speed in km/h from velocity vector"""
    return 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

def normalize_depth_map(depth_map, max_distance=50.0):
    """Normalize depth map to 0-1 range"""
    return np.clip(depth_map / max_distance, 0, 1)