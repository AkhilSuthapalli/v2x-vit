# File path: v2xvit/utils/spatial_aligner.py

import numpy as np

class CentroidConsensusAligner:
    """
    Approach 1: Bounding Box Centroid Consensus Alignment for V2X-ViT.
    Computes an SE(2) rigid transformation matrix correction (BEV translation 
    and yaw) between Ego and Cooperative Sender CAVs using SVD / Kabsch 
    on co-observed object centers before feature fusion.
    """
    def __init__(self, method="svd", max_match_dist=2.5, min_match_dist=0.05):
        """
        Parameters:
            method: 'svd' for joint translation+rotation, 'mean' for pure translation shift.
            max_match_dist: Maximum distance in meters to match centroids between vehicles.
            min_match_dist: Threshold to ignore zero-distance self-matches.
        """
        self.method = method
        self.max_match_dist = max_match_dist
        self.min_match_dist = min_match_dist

    def _extract_bev_centers(self, boxes):
        """
        Extracts 2D BEV center coordinates (X, Y) from V2X-ViT box formats:
        - (N, 7) array: [x, y, z, dx, dy, dz, yaw]
        - (N, 8, 3) array: 8-corner bounding box vertices
        """
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 2))

        boxes = np.asarray(boxes)

        if boxes.ndim == 3 and boxes.shape[1] == 8:
            # Shape (N, 8, 3) -> 8 corner representation (average corners)
            centers = np.mean(boxes, axis=1)[:, :2]
        elif boxes.ndim == 2 and boxes.shape[1] >= 2:
            # Shape (N, 7) or (N, 3) -> center positions
            centers = boxes[:, :2]
        else:
            return np.empty((0, 2))

        return centers

    def match_centroids(self, ego_centers, sender_centers_in_ego):
        """Pairs co-observed object centers in Ego space using distance gating."""
        if len(ego_centers) == 0 or len(sender_centers_in_ego) == 0:
            return np.empty((0, 2)), np.empty((0, 2))

        # Compute pairwise Euclidean distance matrix on BEV plane (X, Y)
        diff = ego_centers[:, None, :] - sender_centers_in_ego[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)

        matched_ego = []
        matched_snd = []
        
        # Greedy distance-gated matching
        for i in range(len(ego_centers)):
            min_idx = np.argmin(dist_matrix[i])
            min_dist = dist_matrix[i, min_idx]

            if self.min_match_dist <= min_dist <= self.max_match_dist:
                matched_ego.append(ego_centers[i])
                matched_snd.append(sender_centers_in_ego[min_idx])

        return np.array(matched_ego), np.array(matched_snd)

    def compute_rigid_delta(self, matched_ego, matched_snd):
        """Computes 4x4 homogeneous transformation delta matrix."""
        T_delta = np.identity(4, dtype=np.float64)

        if len(matched_ego) == 0:
            return T_delta

        if self.method == "mean" or len(matched_ego) < 2:
            # Variant 1A: Direct centroid translation shift mean
            shift = np.mean(matched_ego - matched_snd, axis=0)
            T_delta[0, 3] = shift[0]
            T_delta[1, 3] = shift[1]

        elif self.method == "svd":
            # Variant 1B: Closed-form SVD (Kabsch) over SE(2)
            c_ego = np.mean(matched_ego, axis=0)
            c_snd = np.mean(matched_snd, axis=0)

            p_ego = matched_ego - c_ego
            p_snd = matched_snd - c_snd

            # Cross-covariance matrix
            H = p_snd.T @ p_ego

            # SVD decomposition
            U, S, Vt = np.linalg.svd(H)
            R_2d = Vt.T @ U.T

            # Reflection correction
            if np.linalg.det(R_2d) < 0:
                Vt[1, :] *= -1
                R_2d = Vt.T @ U.T

            # 2D translation vector
            t_2d = c_ego - (R_2d @ c_snd)

            # Assign to 4x4 matrix
            T_delta[0:2, 0:2] = R_2d
            T_delta[0, 3] = t_2d[0]
            T_delta[1, 3] = t_2d[1]

        return T_delta

    def correct_pose_matrix(self, T_noisy, ego_boxes, sender_boxes):
        """
        Main Hook: Accepts noisy 4x4 transformation matrix and bounding boxes,
        returns corrected 4x4 transformation matrix.
        """
        if ego_boxes is None or sender_boxes is None:
            return T_noisy

        ego_pts = self._extract_bev_centers(ego_boxes)
        snd_pts = self._extract_bev_centers(sender_boxes)

        if len(ego_pts) == 0 or len(snd_pts) == 0:
            return T_noisy

        # Transform sender centers into Ego frame using current noisy T_matrix
        snd_homo = np.hstack((snd_pts, np.zeros((len(snd_pts), 1)), np.ones((len(snd_pts), 1))))
        snd_in_ego = (T_noisy @ snd_homo.T).T[:, :2]

        # Match co-observed pairs
        m_ego, m_snd = self.match_centroids(ego_pts, snd_in_ego)

        # Compute correction delta
        T_delta = self.compute_rigid_delta(m_ego, m_snd)

        # Apply correction: T_corrected = T_delta @ T_noisy
        return T_delta @ T_noisy