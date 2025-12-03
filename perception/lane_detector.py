# ============================================================
# perception/lane_detector.py - COMPLETE WORKING VERSION
# ============================================================
"""
Production-ready lane detection using OpenCV
Reliable, fast, and requires no external models
"""
import cv2
import numpy as np

class LaneDetector:
    def __init__(self, model_path=None):
        """
        Simple lane detector using traditional computer vision
        No model file needed - works immediately!
        """
        print("✓ Lane detector initialized (OpenCV mode - fast and reliable)")
        
        # Region of interest parameters
        self.roi_top = 0.55  # Start ROI at 55% from top
        self.roi_bottom = 1.0  # End at bottom
        
        # Canny edge detection parameters
        self.canny_low = 50
        self.canny_high = 150
        
        # Hough transform parameters
        self.hough_threshold = 30
        self.hough_min_line_length = 50
        self.hough_max_line_gap = 100
        
        # Lane width estimation (pixels at bottom of image)
        self.lane_width = 300
    
    def detect(self, rgb_image):
        """
        Detect lanes using OpenCV
        Returns: (visualization_image, lanes_points)
        """
        height, width = rgb_image.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        
        # Define and apply region of interest
        roi_mask = self._create_roi_mask(height, width)
        roi_edges = cv2.bitwise_and(edges, roi_mask)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            roi_edges,
            rho=2,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )
        
        # Create visualization
        lane_viz = rgb_image.copy()
        
        # Initialize lanes_points with proper structure
        # Format: [left_outer, left_inner, right_inner, right_outer]
        lanes_points = [[], [], [], []]
        
        if lines is not None:
            # Separate left and right lanes
            left_lines = []
            right_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate slope
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter by slope (ignore nearly horizontal lines)
                if abs(slope) < 0.3:
                    continue
                
                # Separate left (negative slope) and right (positive slope)
                if slope < 0:
                    left_lines.append(line[0])
                else:
                    right_lines.append(line[0])
            
            # Process left lane
            if left_lines:
                left_lane = self._extrapolate_lane(left_lines, height, width, 'left')
                if left_lane is not None:
                    x1, y1, x2, y2 = left_lane
                    # Draw the lane
                    cv2.line(lane_viz, (x1, y1), (x2, y2), (0, 255, 0), 5)
                    
                    # Create smooth points along the lane
                    lane_points = self._create_smooth_points(x1, y1, x2, y2, 10)
                    lanes_points[1] = lane_points  # Left inner lane
            
            # Process right lane
            if right_lines:
                right_lane = self._extrapolate_lane(right_lines, height, width, 'right')
                if right_lane is not None:
                    x1, y1, x2, y2 = right_lane
                    # Draw the lane
                    cv2.line(lane_viz, (x1, y1), (x2, y2), (0, 255, 0), 5)
                    
                    # Create smooth points along the lane
                    lane_points = self._create_smooth_points(x1, y1, x2, y2, 10)
                    lanes_points[2] = lane_points  # Right inner lane
            
            # Draw center line if both lanes detected
            if lanes_points[1] and lanes_points[2]:
                left_points = np.array(lanes_points[1])
                right_points = np.array(lanes_points[2])
                
                if len(left_points) > 0 and len(right_points) > 0:
                    # Calculate center points
                    min_len = min(len(left_points), len(right_points))
                    for i in range(min_len - 1):
                        # Center point between left and right
                        cx1 = int((left_points[i][0] + right_points[i][0]) / 2)
                        cy1 = int((left_points[i][1] + right_points[i][1]) / 2)
                        cx2 = int((left_points[i+1][0] + right_points[i+1][0]) / 2)
                        cy2 = int((left_points[i+1][1] + right_points[i+1][1]) / 2)
                        
                        # Draw center line
                        cv2.line(lane_viz, (cx1, cy1), (cx2, cy2), (255, 255, 0), 2)
            
            # Draw ROI boundary for debugging
            roi_y = int(height * self.roi_top)
            cv2.line(lane_viz, (0, roi_y), (width, roi_y), (255, 0, 0), 1)
            
            # Add lane detection info
            cv2.putText(lane_viz, f"Left: {'OK' if lanes_points[1] else 'NONE'}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(lane_viz, f"Right: {'OK' if lanes_points[2] else 'NONE'}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # No lanes detected
            cv2.putText(lane_viz, "No lanes detected", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return lane_viz, lanes_points
    
    def _create_roi_mask(self, height, width):
        """Create region of interest mask"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Define trapezoid region
        roi_top_y = int(height * self.roi_top)
        roi_bottom_y = int(height * self.roi_bottom)
        
        # Create trapezoid
        polygon = np.array([[
            (0, roi_bottom_y),                    # Bottom left
            (int(width * 0.4), roi_top_y),        # Top left
            (int(width * 0.6), roi_top_y),        # Top right
            (width, roi_bottom_y)                 # Bottom right
        ]], dtype=np.int32)
        
        cv2.fillPoly(mask, polygon, 255)
        return mask
    
    def _extrapolate_lane(self, lines, height, width, side):
        """
        Extrapolate a lane line from multiple segments
        """
        if not lines:
            return None
        
        lines = np.array(lines)
        
        # Fit a line to all points
        x_coords = np.concatenate([lines[:, 0], lines[:, 2]])
        y_coords = np.concatenate([lines[:, 1], lines[:, 3]])
        
        # Fit polynomial (degree 1 = line)
        try:
            poly = np.polyfit(y_coords, x_coords, 1)
        except:
            return None
        
        # Calculate x coordinates for top and bottom of ROI
        y1 = int(height * self.roi_top)   # Top of ROI
        y2 = height - 1                    # Bottom of image
        
        x1 = int(poly[0] * y1 + poly[1])
        x2 = int(poly[0] * y2 + poly[1])
        
        # Clamp to image bounds
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        
        return [x1, y1, x2, y2]
    
    def _create_smooth_points(self, x1, y1, x2, y2, num_points=10):
        """
        Create smooth points along a line for better visualization
        """
        points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            points.append([x, y])
        return points
    
    def draw_lanes(self, image, lanes_points):
        """Draw lane boundaries on image (for compatibility)"""
        debug_image = image.copy()
        
        if lanes_points:
            for i, lane in enumerate(lanes_points):
                if lane and len(lane) > 0:
                    points = np.array(lane)
                    if points.size > 0:
                        # Draw lane lines
                        for j in range(len(points) - 1):
                            cv2.line(
                                debug_image,
                                (int(points[j][0]), int(points[j][1])),
                                (int(points[j+1][0]), int(points[j+1][1])),
                                (0, 255, 0), 2
                            )
        
        return debug_image