# Cross-Camera Player Mapping

This project implements a system for matching players across different camera views in sports videos using computer vision and deep learning techniques.

## Features

- Player detection using YOLOv11
- Feature extraction using ResNet50
- Cross-camera player matching using appearance and spatial consistency
- Visualization of matched players

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
pip install ultralytics  # Required for YOLO model
```

2. Make sure you have the following files:
- `best.pt` (YOLO weights)
- Your broadcast and tactical camera videos

## Interactive Jupyter Notebook

For a step-by-step analysis, use the provided Jupyter notebook:

```bash
jupyter notebook player_tracking_complete.ipynb
```

The notebook provides an interactive walkthrough of:
- Player detection in each camera view
- Feature extraction and visualization
- Cross-camera player matching
- Processing sequences of frames

## Usage

Run the main script:

```bash
python src/track_players.py --broadcast broadcast.mp4 --tacticam tacticam.mp4 --output_dir results
```

Arguments:
- `--broadcast`: Path to broadcast video file
- `--tacticam`: Path to tactical camera video file
- `--weights`: Path to YOLOv11 weights (default: best.pt)
- `--output_dir`: Directory to save results (default: results)

## Output

The script will generate:
1. `matches.json`: Player ID mappings across cameras
2. `broadcast_annotated.mp4`: Annotated broadcast video
3. `tacticam_annotated.mp4`: Annotated tactical camera video

## Project Structure

```
├── src/
│   ├── track_players.py    # Main script
│   ├── detect.py          # Player detection module
│   ├── feature_extractor.py # Feature extraction
│   ├── matcher.py         # Player matching
│   └── utils.py           # Utility functions
├── requirements.txt
└── README.md
```

## Methodology

1. **Detection**: YOLOv11 for player detection in each frame
2. **Feature Extraction**: ResNet50 for appearance embedding
3. **Tracking**: Frame-by-frame tracking with feature matching
4. **Cross-Camera Matching**: Hungarian algorithm with appearance and spatial consistency

## Future Improvements

- Implement temporal consistency checking
- Add real-time processing capabilities
- Improve matching accuracy with pose estimation
- Add support for more than two camera views
"# Liat.ai-Assignment" 
"# Liat.ai-Assignment" 
