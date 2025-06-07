import cv2
import numpy as np
from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, weights_path='best.pt', conf_threshold=0.25):
        # Load the model using the ultralytics YOLO class
        print(f"Loading YOLO model from {weights_path}...")
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        
    def detect(self, frame):
        """
        Detect players in a frame using YOLO
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of detections, each containing [x1, y1, x2, y2, confidence]
        """
        # Run inference with the ultralytics YOLO model
        results = self.model(frame)
        
        # Process detections
        detections = []
        
        # Extract bounding boxes from results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates in (x1, y1, x2, y2) format
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                if conf >= self.conf_threshold:
                    detections.append([x1, y1, x2, y2, float(conf)])
                
        return np.array(detections)
    
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes on frame
        
        Args:
            frame: BGR image
            detections: Array of [x1, y1, x2, y2, confidence]
            
        Returns:
            Frame with drawn detections
        """
        frame_copy = frame.copy()
        for x1, y1, x2, y2, conf in detections:
            cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame_copy, f'{conf:.2f}', (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame_copy
