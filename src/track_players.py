import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from detect import PlayerDetector
from feature_extractor import FeatureExtractor
from matcher import PlayerMatcher
from visualization_utils import save_results

class PlayerTracker:
    def __init__(self, detector, feature_extractor, matcher):
        self.detector = detector
        self.feature_extractor = feature_extractor
        self.matcher = matcher
        
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        tracks = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect players
            detections = self.detector.detect(frame)
            
            # Extract features
            features = self.feature_extractor.extract(frame, detections)
            
            # Update tracks
            tracks.append((detections, features))
            
        cap.release()
        return tracks

def main():
    parser = argparse.ArgumentParser(description='Cross-Camera Player Tracking')
    parser.add_argument('--broadcast', type=str, required=True, help='Path to broadcast video')
    parser.add_argument('--tacticam', type=str, required=True, help='Path to tactical camera video')
    parser.add_argument('--weights', type=str, default='best.pt', help='Path to YOLOv11 weights')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    args = parser.parse_args()
    
    # Initialize components
    detector = PlayerDetector(weights_path=args.weights)
    feature_extractor = FeatureExtractor()
    matcher = PlayerMatcher()
    
    # Create tracker
    tracker = PlayerTracker(detector, feature_extractor, matcher)
    
    # Process both videos
    print("Processing broadcast video...")
    broadcast_tracks = tracker.process_video(args.broadcast)
    
    print("Processing tactical camera video...")
    tacticam_tracks = tracker.process_video(args.tacticam)
    
    # Match players across cameras
    print("Matching players across cameras...")
    matches = matcher.match_across_cameras(broadcast_tracks, tacticam_tracks)
    
    # Save results
    print("Saving results...")
    save_results(matches, args.broadcast, args.tacticam, args.output_dir)
    
    print("Done!")

if __name__ == "__main__":
    main()
