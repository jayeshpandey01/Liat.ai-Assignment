# Cross-Camera Player Mapping - Project Report

## Task Overview

This project implements a system for matching and tracking players across multiple camera views in sports videos. The system uses computer vision and deep learning techniques to detect players, extract visual features, and match corresponding players between different camera perspectives.

## Key Methodology

### 1. Player Detection

- **YOLOv5/YOLOv11**: We implemented player detection using the YOLOv5 model, which accurately locates players in each frame with bounding boxes.
- **Detection Confidence**: A confidence threshold filters out low-confidence detections, ensuring only reliable player detections are processed.

### 2. Feature Extraction

- **Deep Features**: We extract appearance features using both ResNet50 and a dedicated Re-ID model (OSNet) to create distinguishing visual signatures for each player.
- **Color Histograms**: As a simpler alternative, we implemented color histogram extraction for player matching.
- **Feature Normalization**: Features are normalized to ensure consistent comparison across different player detections.

### 3. Cross-Camera Player Matching

- **Appearance Similarity**: Players are matched based on the similarity of their visual features using cosine distance.
- **Temporal Consistency**: We implemented temporal tracking to maintain player identities across frames, enhancing the matching accuracy.
- **Hungarian Algorithm**: The optimal assignment problem is solved using the Hungarian algorithm to find the best matches between players.

### 4. Tracking and Visualization

- **Short-term Tracking**: Players are tracked within each camera view using frame-to-frame feature matching.
- **Trajectory Visualization**: Player movements are visualized as trajectories on the field.
- **Analysis Tools**: We implemented heatmaps and movement statistics to analyze player positioning and activity.

## Tools and Libraries Used

- **OpenCV**: For video processing, image manipulation, and drawing visualizations
- **PyTorch**: For deep learning model inference
- **Ultralytics YOLOv5**: For player detection
- **TorchReID**: For person re-identification features
- **NumPy/SciPy**: For mathematical operations and the Hungarian algorithm
- **Matplotlib/Seaborn**: For visualization of results

## Challenges and Solutions

### Challenges

1. **Feature Consistency**: Initially, feature vectors had inconsistent shapes, causing errors in similarity calculations.
   - **Solution**: Implemented proper feature formatting and normalization to ensure consistent feature dimensions.

2. **Player Identity Preservation**: Players would sometimes switch identities between frames.
   - **Solution**: Added temporal consistency tracking to maintain player identities across frames.

3. **Different Camera Perspectives**: The broadcast and tactical cameras have very different perspectives and scales.
   - **Solution**: Focused on appearance features rather than spatial positions for cross-camera matching.

4. **Model Compatibility**: Initial issues with YOLOv5/YOLOv11 model loading.
   - **Solution**: Updated the model loading code to handle different PyTorch versions and weight formats.

## Future Improvements

1. **Camera Calibration**: Implementing proper camera calibration to map player positions to actual field coordinates.
2. **Team Identification**: Adding team classification to separate players by team.
3. **Player Identification**: Implementing jersey number recognition for identifying specific players.
4. **Tracking Optimization**: Using more sophisticated tracking algorithms like Deep SORT or ByteTrack.
5. **Performance Optimization**: Optimizing the processing pipeline for real-time applications.
6. **3D Reconstruction**: Implementing 3D position estimation using multiple camera views.
7. **Game Event Detection**: Detecting key events such as passes, shots, and goal attempts.

## Results and Conclusion

The implemented system successfully detects players in both broadcast and tactical camera views and matches them across different perspectives. The Jupyter notebook provides an interactive environment to visualize and analyze the results.

Key achievements:
- Accurate player detection in various poses and scales
- Successful cross-camera matching based on visual appearance
- Temporal consistency in player tracking
- Visualization and analysis tools for player movements

The system serves as a solid foundation for sports video analysis and can be extended for various applications in sports analytics, coaching, and broadcasting.
