"""
Path planning and waypoint generation
"""
import carla

class PathPlanner:
    def __init__(self, world, config):
        self.world = world
        self.config = config
        self.map = world.get_map()
        self.waypoints = []
        self.current_waypoint_index = 0
        
        print("✓ Path planner initialized")
    
    def generate_waypoints(self, spawn_point):
        """Generate waypoints from spawn point"""
        current_waypoint = self.map.get_waypoint(spawn_point.location)
        self.waypoints = [current_waypoint]
        start_location = current_waypoint.transform.location
        
        while len(self.waypoints) < self.config.MAX_WAYPOINTS:
            next_waypoints = current_waypoint.next(self.config.WAYPOINT_DISTANCE)
            if not next_waypoints:
                break
            
            current_waypoint = next_waypoints[0]
            self.waypoints.append(current_waypoint)
            
            # Check if we've looped back
            if (len(self.waypoints) > 10 and 
                current_waypoint.transform.location.distance(start_location) < 5.0):
                print("Route loops back to start")
                break
        
        print(f"✓ Generated {len(self.waypoints)} waypoints")
        return self.waypoints
    
    def get_current_waypoint(self):
        """Get current target waypoint"""
        if self.current_waypoint_index < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index]
        return None
    
    def update_waypoint(self, vehicle_location):
        """Update to next waypoint if current one is reached"""
        if self.current_waypoint_index >= len(self.waypoints):
            return False
        
        current_wp = self.waypoints[self.current_waypoint_index]
        distance = vehicle_location.distance(current_wp.transform.location)
        
        if distance < self.config.WAYPOINT_DISTANCE:
            self.current_waypoint_index += 1
            print(f"→ Waypoint {self.current_waypoint_index}/{len(self.waypoints)}")
            return True
        
        return False
    
    def is_route_complete(self):
        """Check if all waypoints have been reached"""
        return self.current_waypoint_index >= len(self.waypoints)
    
    def draw_waypoints(self):
        """Draw waypoints in the world for visualization"""
        for i, waypoint in enumerate(self.waypoints):
            if i < self.current_waypoint_index:
                color = carla.Color(r=255, g=0, b=0)  # Red for passed
            elif i == self.current_waypoint_index:
                color = carla.Color(r=0, g=255, b=0)  # Green for current
            else:
                color = carla.Color(r=0, g=0, b=255)  # Blue for upcoming
            
            self.world.debug.draw_point(
                waypoint.transform.location + carla.Location(z=0.5),
                size=0.1,
                color=color,
                life_time=0.1
            )