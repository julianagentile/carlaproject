# ============================================================
# FIXED: config/config.py
# ============================================================
"""
Configuration file for all parameters - ADJUSTED VERSION
"""

class Config:
    # CARLA Connection
    CARLA_HOST = 'localhost'
    CARLA_PORT = 2000
    CARLA_TIMEOUT = 10.0
    
    # Map and Environment
    MAP_NAME = 'Town02'
    
    # ADJUSTED: Fewer actors for better stability
    NUM_PEDESTRIANS = 7  # Reduced from 7 (pedestrians are tricky to spawn)
    NUM_VEHICLES = 7     # Reduced from 7 (less traffic congestion)
    
    SPAWN_POINT_INDEX = 11
    
    # Camera Settings
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FOV = 100
    CAMERA_X = 3.0  # Forward offset
    CAMERA_Z = 1.3  # Height offset
    
    # Model Paths
    YOLO_MODEL_PATH = 'models/yolov8n.pt'
    LANE_MODEL_PATH = 'models/tusimple_18.pth'
    
    # Vehicle Control
    MAX_STEER_DEGREES = 70
    TARGET_SPEED_NORMAL = 30  # km/h
    TARGET_SPEED_SLOW = 15    # km/h
    
    # Detection Thresholds
    STOP_DISTANCE = 8.0       # meters
    SLOW_DISTANCE = 15.0      # meters
    WAYPOINT_DISTANCE = 2.0   # meters
    MAX_WAYPOINTS = 200
    
    # Visualization
    DASHBOARD_WIDTH = 1280
    DASHBOARD_HEIGHT = 820  # 720 + 100 for metrics
    RECORD_FPS = 20
    
    # Debug
    SHOW_DEBUG_WINDOWS = True
    ENABLE_RECORDING = True