"""
CARLA world, vehicle, and pedestrian setup
FIXED VERSION - Handles Traffic Manager port conflicts
"""
import carla
import random

class WorldSetup:
    def __init__(self, client, config):
        self.client = client
        self.config = config
        self.world = client.load_world(config.MAP_NAME)
        self.blueprint_library = self.world.get_blueprint_library()
        
        self.ego_vehicle = None
        self.vehicles = []
        self.pedestrians = []
        self.controllers = []
        
        # Setup traffic manager with custom port to avoid conflicts
        self.traffic_manager = None
        self.tm_port = 8000  # Use different port than default (8002)
        
        self.setup_synchronous_mode()
        print(f"✓ World loaded: {config.MAP_NAME}")
    
    def setup_synchronous_mode(self):
        """Enable synchronous mode for deterministic simulation"""
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        
        # Setup traffic manager
        try:
            self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
            self.traffic_manager.set_synchronous_mode(True)
            print(f"✓ Traffic Manager initialized on port {self.tm_port}")
        except Exception as e:
            print(f"⚠️  Traffic Manager warning: {e}")
            print("⚠️  AI vehicles may not work, but ego vehicle will be fine")
        
        print("✓ Synchronous mode enabled")
    
    def spawn_ego_vehicle(self):
        """Spawn the ego vehicle"""
        ego_bp = self.blueprint_library.find('vehicle.tesla.cybertruck')
        spawn_points = self.world.get_map().get_spawn_points()
        
        spawn_point = (spawn_points[self.config.SPAWN_POINT_INDEX] 
                      if len(spawn_points) > self.config.SPAWN_POINT_INDEX 
                      else spawn_points[0])
        
        self.ego_vehicle = self.world.try_spawn_actor(ego_bp, spawn_point)
        
        if self.ego_vehicle:
            # DON'T set autopilot - we control it manually
            # self.ego_vehicle.set_autopilot(False)  # REMOVED - causes TM error
            print(f"✓ Ego vehicle spawned at index {self.config.SPAWN_POINT_INDEX}")
        else:
            raise Exception("Failed to spawn ego vehicle")
        
        return self.ego_vehicle
    
    def spawn_vehicles(self):
        """Spawn AI vehicles with traffic manager"""
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_attempts = 0
        
        while len(self.vehicles) < self.config.NUM_VEHICLES and spawn_attempts < 50:
            try:
                spawn_point = random.choice(spawn_points)
                vehicle_bp = random.choice(self.blueprint_library.filter('vehicle.*'))
                vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                
                if vehicle:
                    # Only set autopilot if traffic manager is available
                    if self.traffic_manager:
                        try:
                            vehicle.set_autopilot(True, self.tm_port)
                        except Exception as e:
                            print(f"⚠️  Could not enable autopilot for vehicle: {e}")
                            # Vehicle will just stay stationary
                    
                    self.vehicles.append(vehicle)
                    print(f"✓ Vehicle {len(self.vehicles)}/{self.config.NUM_VEHICLES} spawned")
            except Exception as e:
                pass
            spawn_attempts += 1
        
        print(f"✓ Spawned {len(self.vehicles)} vehicles")
        if not self.traffic_manager:
            print("⚠️  Vehicles spawned but won't move (Traffic Manager unavailable)")
    
    def spawn_pedestrians(self):
        """Spawn AI pedestrians (optimized for speed)"""
        spawn_points = self.world.get_map().get_spawn_points()
        spawned = 0
        max_total_attempts = self.config.NUM_PEDESTRIANS * 5  # 5 attempts per pedestrian
        attempts = 0
        
        while spawned < self.config.NUM_PEDESTRIANS and attempts < max_total_attempts:
            pedestrian, controller = self._spawn_single_pedestrian(spawn_points)
            if pedestrian and controller:
                self.pedestrians.append(pedestrian)
                self.controllers.append(controller)
                spawned += 1
                print(f"✓ Pedestrian {spawned}/{self.config.NUM_PEDESTRIANS} spawned")
            attempts += 1
        
        if spawned < self.config.NUM_PEDESTRIANS:
            print(f"⚠️  Only spawned {spawned}/{self.config.NUM_PEDESTRIANS} pedestrians")
        else:
            print(f"✓ Spawned {spawned} pedestrians")

    def _spawn_single_pedestrian(self, spawn_points):
        """Spawn a single pedestrian with controller (single attempt)"""
        try:
            # Pick random spawn location
            spawn_location = random.choice(spawn_points).location
            
            # Add small random offset to avoid exact spawn point collision
            spawn_location.x += random.uniform(-2, 2)
            spawn_location.y += random.uniform(-2, 2)
            
            pedestrian_bp = random.choice(self.blueprint_library.filter('walker.pedestrian.*'))
            
            if pedestrian_bp.has_attribute('speed'):
                pedestrian_bp.set_attribute('speed', '1.5')
            
            spawn_point = carla.Transform(spawn_location)
            pedestrian = self.world.try_spawn_actor(pedestrian_bp, spawn_point)
            
            if not pedestrian:
                return None, None
            
            # Spawn controller
            controller_bp = self.blueprint_library.find('controller.ai.walker')
            controller = self.world.try_spawn_actor(
                controller_bp, carla.Transform(), attach_to=pedestrian
            )
            
            if not controller:
                pedestrian.destroy()
                return None, None
            
            # Initialize controller
            self.world.tick()
            controller.start()
            destination = random.choice(spawn_points).location
            controller.go_to_location(destination)
            controller.set_max_speed(1.5)
            
            return pedestrian, controller
            
        except Exception as e:
            # Clean up on failure
            if 'pedestrian' in locals() and pedestrian:
                try:
                    pedestrian.destroy()
                except:
                    pass
            if 'controller' in locals() and controller:
                try:
                    controller.destroy()
                except:
                    pass
            return None, None
    
    def cleanup(self):
        """Clean up all spawned actors"""
        print("\n🧹 Cleaning up world...")
        
        # Stop and destroy controllers first
        for controller in self.controllers:
            try:
                if controller.is_alive:
                    controller.stop()
                    controller.destroy()
            except Exception as e:
                pass
        
        # Destroy pedestrians
        for pedestrian in self.pedestrians:
            try:
                if pedestrian.is_alive:
                    pedestrian.destroy()
            except Exception as e:
                pass
        
        # Destroy vehicles
        for vehicle in self.vehicles:
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
            except Exception as e:
                pass
        
        # Destroy ego vehicle
        if self.ego_vehicle:
            try:
                if self.ego_vehicle.is_alive:
                    self.ego_vehicle.destroy()
            except Exception as e:
                pass
        
        print("✓ Environment cleanup complete")