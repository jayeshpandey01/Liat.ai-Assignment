# How to Use the Advanced Functions in the Player Tracking System

This guide explains how to use the new functions added to the cross-camera player mapping project.

## 1. Using the Re-ID Feature Extractor

The Re-ID feature extractor provides more robust player matching across different camera views by using specialized person re-identification models.

```python
# First, install the required libraries
!pip install torch torchvision torchreid

# Initialize the Re-ID feature extractor
from src.cross_camera_reid import ReIDFeatureExtractor

# Create the extractor
reid_extractor = ReIDFeatureExtractor()

# Extract features from detected players
reid_features = reid_extractor.extract(frame, player_detections)

# Use these features for player matching
# They can be used with the existing matcher or the temporal matcher
```

## 2. Using the Temporal Player Matcher

The temporal matcher maintains player identities across frames by considering both appearance and movement patterns.

```python
from src.fixed_matcher import TemporalPlayerMatcher

# Initialize the temporal matcher
temporal_matcher = TemporalPlayerMatcher(max_cost=0.7, temporal_weight=0.3)

# Process a sequence of frames
for frame_idx in range(num_frames):
    # Get frames from both cameras
    ret1, broadcast_frame = broadcast_cap.read()
    ret2, tacticam_frame = tacticam_cap.read()
    
    # Detect players
    broadcast_detections = detector.detect(broadcast_frame)
    tacticam_detections = detector.detect(tacticam_frame)
    
    # Extract features
    broadcast_features = feature_extractor.extract(broadcast_frame, broadcast_detections)
    tacticam_features = feature_extractor.extract(tacticam_frame, tacticam_detections)
    
    # Format features properly
    broadcast_features = normalize_features(broadcast_features)
    tacticam_features = normalize_features(tacticam_features)
    
    # Match players with temporal consistency
    matches = temporal_matcher.match_with_temporal_consistency(
        broadcast_detections, broadcast_features,
        tacticam_detections, tacticam_features
    )
    
    # Visualize the results
    visualize_matches(broadcast_frame, tacticam_frame, broadcast_detections, tacticam_detections, matches)
```

## 3. Using the Complete Pipeline

The `PlayerTrackingPipeline` class provides an integrated workflow for processing video streams.

```python
from src.utils import PlayerTrackingPipeline, normalize_features

# Initialize components
detector = PlayerDetector(weights_path='best.pt')
feature_extractor = ReIDFeatureExtractor()  # Use Re-ID features for better matching
matcher = TemporalPlayerMatcher()           # Use temporal matching for consistency

# Create the pipeline
pipeline = PlayerTrackingPipeline(detector, feature_extractor, matcher)

# Process a sequence of frames from both videos
results = pipeline.process_videos(
    broadcast_path='broadcast.mp4',
    tacticam_path='tacticam.mp4',
    max_frames=100,  # Limit to 100 frames for faster processing
    visualization=True  # Show visualizations during processing
)
```

## 4. Player Trajectory Visualization and Analysis

```python
from src.utils import plot_player_trajectories, generate_player_heatmap, analyze_player_movement

# Extract trajectory data from processing results
matches_over_time = [match for match, _, _, _, _ in results]
broadcast_detections_over_time = [broadcast_det for _, _, _, broadcast_det, _ in results]
tacticam_detections_over_time = [tacticam_det for _, _, _, _, tacticam_det in results]

# Visualize player trajectories
plot_player_trajectories(
    matches_over_time,
    broadcast_detections_over_time,
    tacticam_detections_over_time
)

# Extract full player data for analysis
broadcast_trajectories, tacticam_trajectories = extract_and_analyze_player_data(results)

# Generate heatmap of player positions
generate_player_heatmap(broadcast_trajectories)

# Analyze player movement patterns
analyze_player_movement(broadcast_trajectories, n_top_players=5)
```

## 5. Creating the Soccer Field Visualization

For a more intuitive visualization of player positions and movements on a soccer field:

```python
import matplotlib.pyplot as plt
from src.utils import draw_soccer_field

# Create a figure with a soccer field background
fig, ax = plt.subplots(figsize=(12, 8))
draw_soccer_field(ax)

# Plot player positions on the field (example)
for player_id, trajectory in broadcast_trajectories.items():
    if len(trajectory) > 0:
        # Skip frame_idx and just get x, y coordinates
        _, xs, ys = zip(*trajectory)
        ax.plot(xs, ys, '-o', label=player_id, alpha=0.7, markersize=3)

plt.title('Player Positions on Field')
plt.xlabel('Field Length (meters)')
plt.ylabel('Field Width (meters)')
plt.tight_layout()
plt.show()
```

## Example: Complete Analysis Workflow

Here's a workflow combining all the advanced features:

```python
# 1. Initialize components with advanced features
detector = PlayerDetector(weights_path='best.pt')
feature_extractor = ReIDFeatureExtractor()
matcher = TemporalPlayerMatcher(temporal_weight=0.3)

# 2. Create and run the pipeline
pipeline = PlayerTrackingPipeline(detector, feature_extractor, matcher)
results = pipeline.process_videos('broadcast.mp4', 'tacticam.mp4', max_frames=150)

# 3. Extract player trajectories and perform analysis
broadcast_trajectories, tacticam_trajectories = extract_and_analyze_player_data(results)

# 4. Generate specific visualizations
# Player heatmap
generate_player_heatmap(broadcast_trajectories)

# Movement statistics
analyze_player_movement(broadcast_trajectories, n_top_players=5)

# Save results for further analysis
import json
import pickle
import os

# Create results directory
os.makedirs('analysis_results', exist_ok=True)

# Save trajectories for later use
with open('analysis_results/player_trajectories.pkl', 'wb') as f:
    pickle.dump({
        'broadcast': broadcast_trajectories,
        'tacticam': tacticam_trajectories
    }, f)

# Export match statistics
with open('analysis_results/match_summary.json', 'w') as f:
    json.dump({
        'total_frames': len(results),
        'player_counts': {
            'broadcast': len(broadcast_trajectories),
            'tacticam': len(tacticam_trajectories)
        },
        'match_counts': [len(match) for match, _, _, _, _ in results]
    }, f, indent=2)

print("Analysis complete! Results saved to 'analysis_results' folder.")
```

By following these examples, you can leverage the advanced functions to improve player detection, tracking, and analysis in your cross-camera player mapping system.
