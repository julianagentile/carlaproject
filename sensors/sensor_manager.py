# ============================================================
# FIXED: sensors/sensor_manager.py
# ============================================================
"""
Sensor setup and data management - FIXED DEPTH PROCESSING
"""
import carla
import numpy as np
import math
from threading import Lock

class SensorManager:
    def __init__(self, world, vehicle, config):
        self.world = world
        self.vehicle = vehicle
        self.config = config
        self.blueprint_library = world.get_blueprint_library()
        
        self.sensors = {}
        self.sensor_data = {
            'rgb_image': None,
            'depth_image': None
        }
        self.sensor_locks = {
            'rgb_image': Lock(),
            'depth_image': Lock()
        }
        
        self.setup_sensors()
    
    def setup_sensors(self):
        """Setup all sensors"""
        # RGB Camera
        cam_bp = self.blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(self.config.CAMERA_WIDTH))
        cam_bp.set_attribute('image_size_y', str(self.config.CAMERA_HEIGHT))
        cam_bp.set_attribute('fov', str(self.config.CAMERA_FOV))
        
        cam_transform = carla.Transform(
            carla.Location(x=self.config.CAMERA_X, z=self.config.CAMERA_Z)
        )
        
        self.sensors['rgb_camera'] = self.world.spawn_actor(
            cam_bp, cam_transform, attach_to=self.vehicle
        )
        self.sensors['rgb_camera'].listen(self.rgb_callback)
        
        # Depth Camera
        depth_bp = self.blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', str(self.config.CAMERA_WIDTH))
        depth_bp.set_attribute('image_size_y', str(self.config.CAMERA_HEIGHT))
        depth_bp.set_attribute('fov', str(self.config.CAMERA_FOV))
        
        self.sensors['depth_camera'] = self.world.spawn_actor(
            depth_bp, cam_transform, attach_to=self.vehicle
        )
        self.sensors['depth_camera'].listen(self.depth_callback)
        
        # Collision Sensor
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.sensors['collision_sensor'] = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.vehicle
        )
        self.sensors['collision_sensor'].listen(self.collision_callback)
        
        print("✓ Sensors initialized")
    
    def rgb_callback(self, image):
        """Process RGB camera data"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        with self.sensor_locks['rgb_image']:
            self.sensor_data['rgb_image'] = array
    
    def depth_callback(self, image):
        """
        Process depth camera data - FIXED VERSION
        Properly handles uint8 overflow by converting to larger dtype first
        """
        # Get raw data as uint8
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        
        # Convert to float32 BEFORE doing arithmetic to avoid overflow
        # CARLA depth encoding: (R + G * 256 + B * 256 * 256) / (256^3 - 1)
        r = array[:, :, 0].astype(np.float32)
        g = array[:, :, 1].astype(np.float32)
        b = array[:, :, 2].astype(np.float32)
        
        # Decode normalized depth [0, 1]
        normalized = (r + g * 256.0 + b * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1.0)
        
        # Convert to meters (CARLA depth is normalized to 1km max)
        depth_meters = normalized * 1000.0
        
        with self.sensor_locks['depth_image']:
            self.sensor_data['depth_image'] = depth_meters
    
    def collision_callback(self, event):
        """Handle collision events"""
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        print(f'⚠ Collision with {event.other_actor.type_id}, intensity: {intensity:.2f}')
        
        # Emergency stop
        control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
        self.vehicle.apply_control(control)
    
    def get_data(self, sensor_type):
        """Thread-safe data retrieval"""
        with self.sensor_locks[sensor_type]:
            return self.sensor_data[sensor_type]
    
    def cleanup(self):
        """Destroy all sensors"""
        for sensor in self.sensors.values():
            if sensor.is_alive:
                sensor.destroy()