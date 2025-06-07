import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np

class FeatureExtractor:
    def __init__(self, use_resnet=True):
        if use_resnet:
            # Load pre-trained ResNet50
            self.model = models.resnet50(pretrained=True)
            # Remove the final classification layer
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.use_resnet = use_resnet
        
    def extract(self, frame, detections):
        """
        Extract features from detected player regions
        
        Args:
            frame: BGR image
            detections: Array of [x1, y1, x2, y2, conf]
            
        Returns:
            List of feature vectors
        """
        features = []
        
        for x1, y1, x2, y2, _ in detections:
            # Extract player region
            player_roi = frame[int(y1):int(y2), int(x1):int(x2)]
            
            if self.use_resnet:
                # Extract deep features using ResNet
                features.append(self._extract_deep_features(player_roi))
            else:
                # Extract color histogram features
                features.append(self._extract_color_histogram(player_roi))
                
        return features
    
    def _extract_deep_features(self, player_roi):
        """Extract deep features using ResNet50"""
        if player_roi.size == 0:
            return np.zeros(2048)  # ResNet50 feature dimension
            
        # Prepare image
        img_tensor = self.transform(player_roi)
        img_tensor = img_tensor.unsqueeze(0)
        
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
            
        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)
            
        return features.cpu().numpy().flatten()
    
    def _extract_color_histogram(self, player_roi):
        """Extract color histogram features"""
        if player_roi.size == 0:
            return np.zeros(256 * 3)  # 3 channels * 256 bins
            
        hist_features = []
        for channel in cv2.split(player_roi):
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)
            
        return np.array(hist_features)
