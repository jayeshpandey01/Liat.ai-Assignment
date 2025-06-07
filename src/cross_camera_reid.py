from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm
from utils import extract_player_crops, get_color_histogram, compute_similarity_matrix

def detect_players(video_path, model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    features = []

    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        bboxes = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                if cls_id == 0:  # Player class
                    bboxes.append(box.xyxy.cpu().numpy()[0])
        
        crops = extract_player_crops(frame, bboxes)
        for crop in crops:
            feat = get_color_histogram(crop)
            features.append(feat)
    
    cap.release()
    return features

def main():
    model_path = "yolov11.pt"
    features_broadcast = detect_players("broadcast.mp4", model_path)
    features_tacticam = detect_players("tacticam.mp4", model_path)

    print("[INFO] Computing similarity matrix...")
    sim_matrix = compute_similarity_matrix(features_tacticam, features_broadcast)

    print("[INFO] Mapping players...")
    mapping = np.argmax(sim_matrix, axis=1)
    for i, m in enumerate(mapping):
        print(f"Tacticam Player {i} → Broadcast Player {m}")

if __name__ == "__main__":
    main()
