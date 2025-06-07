import cv2
import numpy as np
from pathlib import Path
import json

def save_results(matches, broadcast_path, tacticam_path, output_dir):
    """
    Save matching results and create visualization
    
    Args:
        matches: Dictionary mapping broadcast IDs to tacticam IDs
        broadcast_path: Path to broadcast video
        tacticam_path: Path to tactical camera video
        output_dir: Directory to save results
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save matches to JSON
    with open(output_dir / 'matches.json', 'w') as f:
        json.dump(matches, f, indent=4)
    
    # Create visualizations
    create_visualization(broadcast_path, matches, output_dir / 'broadcast_annotated.mp4',
                       source_type='broadcast')
    create_visualization(tacticam_path, matches, output_dir / 'tacticam_annotated.mp4',
                       source_type='tacticam')

def create_visualization(video_path, matches, output_path, source_type='broadcast'):
    """Create annotated video with player IDs"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Add annotations here based on matches
        # This is a placeholder - you'll need to implement the actual visualization
        # based on your tracking results
        
        out.write(frame)
    
    cap.release()
    out.release()

def draw_player_id(frame, x, y, player_id, color=(0, 255, 0)):
    """Draw player ID on frame"""
    cv2.putText(frame, str(player_id), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2)
