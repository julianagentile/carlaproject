"""
Multi-panel visualization dashboard
"""
import cv2
import numpy as np
import time
from collections import deque

class Dashboard:
    def __init__(self, config):
        self.config = config
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Recording
        self.recording = False
        self.video_writer = None
        
        # Weather mode
        self.weather_mode = 'clear'
        
        print("✓ Dashboard initialized")
    
    def create_depth_visualization(self, depth_map):
        """Create color-coded depth visualization"""
        normalized_depth = np.clip(depth_map / 50.0, 0, 1)
        colored_depth = cv2.applyColorMap(
            (normalized_depth * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        return colored_depth
    
    def create_bird_eye_view(self, rgb_image, lane_points):
        """Create bird's eye view transformation"""
        h, w = rgb_image.shape[:2]
        
        # Define source points (trapezoid in front view)
        src_points = np.float32([
            [w * 0.2, h * 0.7],  # Bottom left
            [w * 0.8, h * 0.7],  # Bottom right
            [w * 0.6, h * 0.4],  # Top right
            [w * 0.4, h * 0.4]   # Top left
        ])
        
        # Define destination points (rectangle in bird's eye view)
        dst_points = np.float32([
            [w * 0.2, h],
            [w * 0.8, h],
            [w * 0.8, 0],
            [w * 0.2, 0]
        ])
        
        # Get perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply transformation
        bird_eye = cv2.warpPerspective(rgb_image, matrix, (w, h))
        
        return bird_eye
    
    def create_dashboard(self, rgb_image, depth_image, lane_viz, 
                        objects_detected, traffic_light_state, 
                        speed, steer, status, waypoint_info):
        """Create comprehensive 4-panel dashboard"""
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time) if (current_time - self.last_frame_time) > 0 else 0
        self.last_frame_time = current_time
        self.fps_history.append(fps)
        avg_fps = np.mean(self.fps_history)
        
        # Panel dimensions
        panel_h, panel_w = 360, 640
        
        # ===== PANEL 1: RGB Camera View =====
        rgb_panel = cv2.resize(rgb_image, (panel_w, panel_h))
        cv2.putText(rgb_panel, "Camera View", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # ===== PANEL 2: Lane Detection =====
        lane_panel = cv2.resize(lane_viz, (panel_w, panel_h))
        cv2.putText(lane_panel, "Lane Detection (ML)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # ===== PANEL 3: Depth Perception =====
        depth_viz = self.create_depth_visualization(depth_image)
        depth_panel = cv2.resize(depth_viz, (panel_w, panel_h))
        cv2.putText(depth_panel, "Depth Perception (ML)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Add depth scale
        scale_h = 30
        scale = np.linspace(0, 255, panel_w).astype(np.uint8)
        scale = np.tile(scale, (scale_h, 1))
        scale_colored = cv2.applyColorMap(scale, cv2.COLORMAP_JET)
        depth_panel[panel_h-scale_h:panel_h, :] = scale_colored
        cv2.putText(depth_panel, "0m", (5, panel_h-35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(depth_panel, "50m", (panel_w-40, panel_h-35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # ===== PANEL 4: Object Detection Summary =====
        obj_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        cv2.putText(obj_panel, "Object Detection (YOLO)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw object list
        y_offset = 70
        sorted_objects = sorted(objects_detected, key=lambda x: x.get('distance', float('inf')))
        
        for i, obj in enumerate(sorted_objects[:8]):  # Show top 8
            class_name = obj.get('class_name', 'unknown')
            distance = obj.get('distance', float('inf'))
            confidence = obj.get('confidence', 0.0)
            
            # Color code by distance
            if distance < 8:
                color = (0, 0, 255)  # Red - danger
                status_text = "STOP"
            elif distance < 15:
                color = (0, 165, 255)  # Orange - caution
                status_text = "SLOW"
            else:
                color = (0, 255, 0)  # Green - safe
                status_text = "OK"
            
            text = f"{i+1}. {class_name}: {distance:.1f}m [{status_text}]"
            cv2.putText(obj_panel, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw mini confidence bar
            bar_width = int(confidence * 100)
            cv2.rectangle(obj_panel, (20, y_offset+5), 
                         (20+bar_width, y_offset+10), color, -1)
            
            y_offset += 35
        
        if len(objects_detected) == 0:
            cv2.putText(obj_panel, "No objects detected", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
        
        # ===== Combine into 2x2 grid =====
        top_row = np.hstack([rgb_panel, lane_panel])
        bottom_row = np.hstack([depth_panel, obj_panel])
        dashboard = np.vstack([top_row, bottom_row])
        
        # ===== Add metrics panel at bottom =====
        metrics_h = 100
        metrics_panel = np.zeros((metrics_h, dashboard.shape[1], 3), dtype=np.uint8)
        
        # Create metrics layout
        metrics = [
            f"FPS: {avg_fps:.1f}",
            f"Speed: {speed:.1f} km/h",
            f"Steering: {steer:.3f}",
            f"Status: {status}",
            f"Objects: {len(objects_detected)}",
            f"Traffic: {traffic_light_state if traffic_light_state else 'None'}",
            f"Waypoint: {waypoint_info}",
            f"Weather: {self.weather_mode.upper()}"
        ]
        
        # Status color coding
        status_colors = {
            'DRIVING': (0, 255, 0),
            'SLOWING': (0, 165, 255),
            'STOPPED': (0, 0, 255),
            'PAUSED': (255, 255, 0)
        }
        
        x_pos = 20
        row = 0
        for i, metric in enumerate(metrics):
            y_pos = 25 + row * 30
            
            # Special color for status
            if 'Status:' in metric:
                color = status_colors.get(status, (255, 255, 255))
            else:
                color = (255, 255, 255)
            
            cv2.putText(metrics_panel, metric, (x_pos, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            x_pos += 300
            if (i + 1) % 4 == 0:
                x_pos = 20
                row += 1
        
        # Recording indicator
        if self.recording:
            cv2.circle(metrics_panel, (dashboard.shape[1] - 30, 25), 
                      10, (0, 0, 255), -1)
            cv2.putText(metrics_panel, "REC", (dashboard.shape[1] - 60, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Combine everything
        final_dashboard = np.vstack([dashboard, metrics_panel])
        
        return final_dashboard
    
    def toggle_recording(self, frame_shape):
        """Toggle video recording"""
        if not self.recording:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f'recordings/carla_run_{timestamp}.mp4'
            
            self.video_writer = cv2.VideoWriter(
                filename, fourcc, self.config.RECORD_FPS, 
                (frame_shape[1], frame_shape[0])
            )
            self.recording = True
            print(f"🔴 Recording started: {filename}")
        else:
            if self.video_writer:
                self.video_writer.release()
            self.recording = False
            print("⏹ Recording stopped")
    
    def save_frame(self, frame):
        """Save frame to video if recording"""
        if self.recording and self.video_writer is not None:
            self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def cleanup(self):
        """Clean up resources"""
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
