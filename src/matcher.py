import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

class PlayerMatcher:
    def __init__(self, max_cost=0.7):
        self.max_cost = max_cost
        
    def match_across_cameras(self, broadcast_tracks, tacticam_tracks):
        """
        Match players across two camera views
        
        Args:
            broadcast_tracks: List of (detections, features) from broadcast view
            tacticam_tracks: List of (detections, features) from tactical camera view
            
        Returns:
            Dictionary mapping broadcast player IDs to tactical camera IDs
        """
        # Calculate similarity matrix between all players
        similarity_matrix = self._calculate_similarity_matrix(
            broadcast_tracks, tacticam_tracks)
        
        # Use Hungarian algorithm for optimal assignment
        broadcast_indices, tacticam_indices = linear_sum_assignment(similarity_matrix)
        
        # Create mapping dictionary
        matches = {}
        for b_idx, t_idx in zip(broadcast_indices, tacticam_indices):
            if similarity_matrix[b_idx, t_idx] <= self.max_cost:
                matches[f'broadcast_{b_idx}'] = f'tacticam_{t_idx}'
                
        return matches
    def _calculate_similarity_matrix(self, broadcast_tracks, tacticam_tracks):
    
        broadcast_features = [track[1] for track in broadcast_tracks]
        tacticam_features = [track[1] for track in tacticam_tracks]

        # Flatten and validate
        def prepare_features(features):
            if isinstance(features, list):
                features = [np.asarray(f).flatten() for f in features]
            features = np.array(features)
            if features.ndim == 1:
                features = features.reshape(1, -1)
            return features

        broadcast_features = prepare_features(broadcast_features)
        tacticam_features = prepare_features(tacticam_features)

        # Match dimensions by padding or truncating
        b_dim = broadcast_features.shape[1]
        t_dim = tacticam_features.shape[1]
        target_dim = max(b_dim, t_dim)

        def resize_features(arr, target_dim):
            current_dim = arr.shape[1]
            if current_dim == target_dim:
                return arr
            elif current_dim < target_dim:
                pad_width = target_dim - current_dim
                return np.pad(arr, ((0, 0), (0, pad_width)), mode='constant')
            else:
                return arr[:, :target_dim]  # truncate

        broadcast_features = resize_features(broadcast_features, target_dim)
        tacticam_features = resize_features(tacticam_features, target_dim)

        # Compute cosine distances
        distance_matrix = cdist(broadcast_features, tacticam_features, metric='cosine')
        return distance_matrix
