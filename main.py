"""
Main entry point for autonomous driving system
"""
import sys
import time
import cv2
import carla

from config.config import Config
from environment.world_setup import WorldSetup
from sensors.sensor_manager import SensorManager
from perception.lane_detector import LaneDetector
from perception.object_detector import ObjectDetector
from perception.traffic_light_analyzer import TrafficLightAnalyzer
from planning.path_planner import PathPlanner
from control.vehicle_controller import VehicleController
from visualization.dashboard import Dashboard
from utils.helpers import calculate_vehicle_speed

class AutonomousDrivingSystem:
    def __init__(self):
        print("=" * 60)
        print("🚗 Enhanced Autonomous Driving System")
        print("=" * 60)
        
        self.config = Config()
        
        # Connect to CARLA
        print(f"\n📡 Connecting to CARLA at {self.config.CARLA_HOST}:{self.config.CARLA_PORT}...")
        self.client = carla.Client(self.config.CARLA_HOST, self.config.CARLA_PORT)
        self.client.set_timeout(self.config.CARLA_TIMEOUT)
        
        # Setup world and environment
        print("\n🌍 Setting up environment...")
        self.world_setup = WorldSetup(self.client, self.config)
        
        # Spawn actors
        print("\n🚙 Spawning ego vehicle...")
        self.ego_vehicle = self.world_setup.spawn_ego_vehicle()
        
        print("\n🚗 Spawning AI vehicles...")
        self.world_setup.spawn_vehicles()
        
        print("\n🚶 Spawning pedestrians...")
        self.world_setup.spawn_pedestrians()
        
        # Setup sensors
        print("\n📷 Setting up sensors...")
        self.sensor_manager = SensorManager(
            self.world_setup.world, 
            self.ego_vehicle, 
            self.config
        )
        
        # Initialize perception modules
        print("\n🧠 Loading ML models...")
        self.lane_detector = LaneDetector(self.config.LANE_MODEL_PATH)
        self.object_detector = ObjectDetector(self.config.YOLO_MODEL_PATH)
        self.traffic_light_analyzer = TrafficLightAnalyzer()
        
        # Initialize planning and control
        print("\n🗺️  Initializing path planner...")
        self.path_planner = PathPlanner(self.world_setup.world, self.config)
        
        print("\n🎮 Initializing vehicle controller...")
        self.vehicle_controller = VehicleController(self.ego_vehicle, self.config)
        
        # Initialize visualization
        print("\n📊 Initializing dashboard...")
        self.dashboard = Dashboard(self.config)
        
        print("\n✅ System initialization complete!")
        print("=" * 60)
    
    def run(self):
        """Main driving loop"""
        print("\n🏁 Starting autonomous navigation...\n")
        
        # Generate path
        spawn_point = self.ego_vehicle.get_transform()
        self.path_planner.generate_waypoints(spawn_point)
        
        # Create visualization window
        cv2.namedWindow('Autonomous Driving Dashboard', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Autonomous Driving Dashboard', 1280, 820)
        
        print("Controls:")
        print("  'r' - Toggle recording")
        print("  'q' - Quit")
        print("  'w' - Set weather to rain")
        print("  'f' - Set weather to fog")
        print("  'c' - Clear weather")
        print()
        
        frame_count = 0
        
        try:
            while not self.path_planner.is_route_complete():
                # Tick simulation
                self.world_setup.world.tick()
                frame_count += 1
                
                # Get sensor data
                rgb_data = self.sensor_manager.get_data('rgb_image')
                depth_data = self.sensor_manager.get_data('depth_image')
                
                if rgb_data is None or depth_data is None:
                    continue
                
                # ===== PERCEPTION =====
                # Detect lanes
                lane_viz, lanes_points = self.lane_detector.detect(rgb_data)
                
                # Detect objects
                detections = self.object_detector.detect(rgb_data)
                
                # Estimate distances and filter relevant objects
                objects_with_distance = []
                traffic_light_state = None
                
                for det in detections:
                    class_id = det['class_id']
                    bbox = det['bbox']
                    
                    if class_id in [0, 1, 2, 3, 5, 7]:  # Vehicles and people
                        distance = self.object_detector.estimate_distance(depth_data, bbox)
                        det['distance'] = distance
                        objects_with_distance.append(det)
                        
                        # Draw on visualization
                        x1, y1, x2, y2 = map(int, bbox)
                        color = (0, 255, 0) if distance > 15 else (0, 165, 255) if distance > 8 else (0, 0, 255)
                        cv2.rectangle(lane_viz, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(lane_viz, f"{distance:.1f}m", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    elif class_id == 9:  # Traffic light
                        x1, y1, x2, y2 = map(int, bbox)
                        roi = rgb_data[y1:y2, x1:x2]
                        traffic_light_state = self.traffic_light_analyzer.analyze(roi)
                        
                        cv2.rectangle(lane_viz, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        if traffic_light_state:
                            cv2.putText(lane_viz, traffic_light_state, (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # ===== DECISION MAKING =====
                should_stop = False
                should_slow = False
                target_speed = self.config.TARGET_SPEED_NORMAL
                
                # Check for obstacles
                for obj in objects_with_distance:
                    if obj['distance'] < self.config.STOP_DISTANCE:
                        should_stop = True
                    elif obj['distance'] < self.config.SLOW_DISTANCE:
                        should_slow = True
                
                # Check traffic light
                if traffic_light_state == 'red':
                    should_stop = True
                elif traffic_light_state == 'yellow':
                    should_slow = True
                
                # Determine target speed and status
                if should_stop:
                    target_speed = 0
                    status = "STOPPED"
                    self.vehicle_controller.paused = True
                    self.vehicle_controller.resume_time = time.time() + 2.0
                elif self.vehicle_controller.paused and time.time() < self.vehicle_controller.resume_time:
                    target_speed = 0
                    status = "PAUSED"
                elif should_slow:
                    target_speed = self.config.TARGET_SPEED_SLOW
                    status = "SLOWING"
                else:
                    self.vehicle_controller.paused = False
                    status = "DRIVING"
                
                # ===== CONTROL =====
                # Get vehicle state
                current_transform = self.ego_vehicle.get_transform()
                velocity = self.ego_vehicle.get_velocity()
                current_speed = calculate_vehicle_speed(velocity)
                
                # Update waypoint if reached
                self.path_planner.update_waypoint(current_transform.location)
                
                # Get current target waypoint
                target_waypoint = self.path_planner.get_current_waypoint()
                
                if target_waypoint:
                    # Calculate controls
                    steer = self.vehicle_controller.calculate_steering(target_waypoint)
                    throttle, brake = self.vehicle_controller.calculate_speed_control(
                        current_speed, target_speed
                    )
                    
                    # Apply emergency brake if needed
                    if should_stop:
                        throttle = 0.0
                        brake = 1.0
                    
                    # Apply control
                    self.vehicle_controller.apply_control(throttle, steer, brake)
                    
                    # Draw waypoints
                    self.path_planner.draw_waypoints()
                
                # ===== VISUALIZATION =====
                waypoint_info = f"{self.path_planner.current_waypoint_index}/{len(self.path_planner.waypoints)}"
                
                dashboard_frame = self.dashboard.create_dashboard(
                    rgb_data, depth_data, lane_viz,
                    objects_with_distance, traffic_light_state,
                    current_speed, steer if target_waypoint else 0.0,
                    status, waypoint_info
                )
                
                cv2.imshow('Autonomous Driving Dashboard', dashboard_frame)
                
                # Save frame if recording
                self.dashboard.save_frame(dashboard_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Quitting...")
                    break
                elif key == ord('r'):
                    self.dashboard.toggle_recording(dashboard_frame.shape)
                elif key == ord('w'):
                    self.set_weather('rain')
                elif key == ord('f'):
                    self.set_weather('fog')
                elif key == ord('c'):
                    self.set_weather('clear')
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        finally:
            print("\n🧹 Cleaning up...")
            self.cleanup()
            print("✅ Done!")
    
    def set_weather(self, weather_type):
        """Change weather conditions"""
        weather = self.world_setup.world.get_weather()
        
        if weather_type == 'rain':
            weather.precipitation = 80.0
            weather.precipitation_deposits = 50.0
            weather.wetness = 70.0
            self.dashboard.weather_mode = 'rain'
            print("🌧️  Weather set to RAIN")
        elif weather_type == 'fog':
            weather.fog_density = 50.0
            weather.fog_distance = 20.0
            self.dashboard.weather_mode = 'fog'
            print("🌫️  Weather set to FOG")
        else:  # clear
            weather.precipitation = 0.0
            weather.fog_density = 0.0
            weather.wetness = 0.0
            self.dashboard.weather_mode = 'clear'
            print("☀️  Weather set to CLEAR")
        
        self.world_setup.world.set_weather(weather)
    
    def cleanup(self):
        """Clean up all resources"""
        self.dashboard.cleanup()
        self.sensor_manager.cleanup()
        self.world_setup.cleanup()
        cv2.destroyAllWindows()

def main():
    try:
        system = AutonomousDrivingSystem()
        system.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())