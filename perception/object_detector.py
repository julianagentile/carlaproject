"""
Object detection module using YOLO
"""
import cv2
import numpy as np
import torch
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = YOLO(model_path)
        self.detector.to(self.device)
        
        self.class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            5: 'bus', 7: 'truck', 9: 'traffic_light'
        }
        
        print(f"✓ Object detector initialized on {self.device}")
    
    def preprocess_image(self, rgb_image):
        """Preprocess image for YOLO"""
        orig_h, orig_w = rgb_image.shape[:2]
        scale_factor = 640 / max(orig_h, orig_w)
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        
        resized = cv2.resize(rgb_image, (new_w, new_h))
        square_img = np.zeros((640, 640, 3), dtype=np.uint8)
        dx = (640 - new_w) // 2
        dy = (640 - new_h) // 2
        square_img[dy:dy+new_h, dx:dx+new_w] = resized
        
        return square_img, scale_factor, dx, dy, orig_w, orig_h
    
    def detect(self, rgb_image):
        """Detect objects in image"""
        square_img, scale_factor, dx, dy, orig_w, orig_h = self.preprocess_image(rgb_image)
        
        # Convert to tensor
        rgb_tensor = torch.from_numpy(square_img.transpose(2, 0, 1)).float() / 255.0
        rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)
        
        # Run detection
        results = self.detector(rgb_tensor)
        
        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                conf = boxes.conf[i].item()
                bbox = boxes.xyxy[i].cpu().numpy()
                
                # Scale back to original coordinates
                bbox[0] = (bbox[0] - dx) / scale_factor
                bbox[1] = (bbox[1] - dy) / scale_factor
                bbox[2] = (bbox[2] - dx) / scale_factor
                bbox[3] = (bbox[3] - dy) / scale_factor
                bbox = np.clip(bbox, [0, 0, 0, 0], [orig_w-1, orig_h-1, orig_w-1, orig_h-1])
                
                detections.append({
                    'class_id': class_id,
                    'class_name': self.class_names.get(class_id, 'unknown'),
                    'confidence': conf,
                    'bbox': bbox
                })
        
        return detections
    
    def estimate_distance(self, depth_map, bbox):
        """Estimate distance using depth map"""
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, depth_map.shape[1]-1))
        x2 = max(0, min(x2, depth_map.shape[1]-1))
        y1 = max(0, min(y1, depth_map.shape[0]-1))
        y2 = max(0, min(y2, depth_map.shape[0]-1))
        
        depth_roi = depth_map[y1:y2, x1:x2]
        valid_depths = depth_roi[(depth_roi > 0) & (depth_roi < 1000)]
        
        if len(valid_depths) > 0:
            return np.mean(valid_depths)
        return float('inf')
