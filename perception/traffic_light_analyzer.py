"""
Traffic light state detection
"""
import cv2
import numpy as np

class TrafficLightAnalyzer:
    def __init__(self):
        self.min_pixel_threshold = 0.05  # 5% of ROI
        print("✓ Traffic light analyzer initialized")
    
    def analyze(self, roi):
        """Analyze traffic light ROI to determine state"""
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return None
        
        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges
        red_mask1 = cv2.inRange(hsv_roi, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv_roi, np.array([160, 100, 100]), np.array([180, 255, 255]))
        yellow_mask = cv2.inRange(hsv_roi, np.array([20, 100, 100]), np.array([40, 255, 255]))
        green_mask = cv2.inRange(hsv_roi, np.array([40, 100, 100]), np.array([90, 255, 255]))
        
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # Count pixels
        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)
        
        total_pixels = roi.shape[0] * roi.shape[1]
        min_pixels = total_pixels * self.min_pixel_threshold
        
        # Determine state
        if red_pixels > min_pixels and red_pixels > max(yellow_pixels, green_pixels):
            return 'red'
        elif yellow_pixels > min_pixels and yellow_pixels > max(red_pixels, green_pixels):
            return 'yellow'
        elif green_pixels > min_pixels and green_pixels > max(red_pixels, yellow_pixels):
            return 'green'
        
        return None
