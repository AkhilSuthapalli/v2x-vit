import numpy as np

class CentroidConsensusAligner:
    """
    Approach 1: Single-Centroid SVD Aligner with Mutual 1-to-1 Matching,
    Median Displacement Consensus, and Deadband Filtering (<0.35m).
    """
    def __init__(self, method="svd", max_match_dist=3.0, min_match_dist=0.05, deadband_thresh=0.35):
        self.method = method
        self.max_match_dist = max_match_dist
        self.min_match_dist = min_match_dist
        self.deadband_thresh = deadband_thresh  # Threshold in meters (0.35m)

    def match_centroids(self, ego_boxes, snd_in_ego_boxes):
        """
        Performs mutual (1-to-1) nearest-neighbor matching to prevent
        cross-vehicle duplicate pairing.
        """
        ego_centers = ego_boxes[:, :2]
        snd_centers = snd_in_ego_boxes[:, :2]

        if len(ego_centers) == 0 or len(snd_centers) == 0:
            return np.empty((0, 2)), np.empty((0, 2))

        diff = ego_centers[:, None, :] - snd_centers[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)

        # Mutual nearest neighbor matching
        ego_to_snd = np.argmin(dist_matrix, axis=1)  # Best sender for each ego
        snd_to_ego = np.argmin(dist_matrix, axis=0)  # Best ego for each sender

        matched_ego = []
        matched_snd = []

        for ego_idx, snd_idx in enumerate(ego_to_snd):
            if snd_to_ego[snd_idx] == ego_idx:
                dist = dist_matrix[ego_idx, snd_idx]
                if self.min_match_dist <= dist <= self.max_match_dist:
                    matched_ego.append(ego_centers[ego_idx])
                    matched_snd.append(snd_centers[snd_idx])

        return np.array(matched_ego), np.array(matched_snd)

    def compute_rigid_delta(self, matched_ego, matched_snd):
        T_delta = np.identity(4, dtype=np.float64)

        if len(matched_ego) == 0:
            return T_delta, 0.0

        # Pairwise displacement vectors
        displacements = matched_ego - matched_snd  # (K, 2)
        
        # Calculate robust consensus displacement using MEDIAN (ignores 3.5m cross-lane outliers)
        median_shift = np.median(displacements, axis=0)

        # Filter out outlier matched pairs that deviate from median shift (> 1.5m deviation)
        pair_deviations = np.linalg.norm(displacements - median_shift, axis=1)
        inliers = pair_deviations < 1.5

        valid_ego = matched_ego[inliers]
        valid_snd = matched_snd[inliers]

        if len(valid_ego) == 0:
            return T_delta, 0.0

        # Compute shift magnitude on clean inliers
        inlier_displacements = valid_ego - valid_snd
        consensus_shift = np.mean(inlier_displacements, axis=0)
        consensus_shift_mag = np.linalg.norm(consensus_shift)

        if consensus_shift_mag < self.deadband_thresh:
            return T_delta, consensus_shift_mag

        # Solve closed-form SVD on consensus inliers
        c_ego = np.mean(valid_ego, axis=0)
        c_snd = np.mean(valid_snd, axis=0)

        p_ego = valid_ego - c_ego
        p_snd = valid_snd - c_snd

        H = p_snd.T @ p_ego
        U, S, Vt = np.linalg.svd(H)
        R_2d = Vt.T @ U.T

        if np.linalg.det(R_2d) < 0:
            Vt[1, :] *= -1
            R_2d = Vt.T @ U.T

        t_2d = c_ego - (R_2d @ c_snd)

        T_delta[0:2, 0:2] = R_2d
        T_delta[0, 3] = t_2d[0]
        T_delta[1, 3] = t_2d[1]

        return T_delta, consensus_shift_mag

    def correct_pose_matrix(self, T_noisy, ego_boxes, sender_boxes):
        if ego_boxes is None or sender_boxes is None or len(ego_boxes) == 0 or len(sender_boxes) == 0:
            return T_noisy

        ego_boxes = np.asarray(ego_boxes)
        sender_boxes = np.asarray(sender_boxes)

        snd_3d = sender_boxes[:, :3] if sender_boxes.shape[1] >= 3 else np.hstack((sender_boxes[:, :2], np.zeros((len(sender_boxes), 1))))
        snd_homo = np.hstack((snd_3d, np.ones((len(snd_3d), 1))))
        snd_in_ego_centers = (T_noisy @ snd_homo.T).T[:, :3]

        snd_in_ego_boxes = sender_boxes.copy()
        snd_in_ego_boxes[:, :3] = snd_in_ego_centers

        m_ego, m_snd = self.match_centroids(ego_boxes, snd_in_ego_boxes)

        if len(m_ego) == 0:
            return T_noisy

        T_delta, shift_mag = self.compute_rigid_delta(m_ego, m_snd)

        if shift_mag < self.deadband_thresh:
            print(f"[FILTER BYPASS] Shift: {shift_mag:.3f}m < {self.deadband_thresh}m -> Keeping T_noisy")
            return T_noisy
        else:
            print(f"[FILTER TRIGGERED] Shift: {shift_mag:.3f}m >= {self.deadband_thresh}m -> Applying SVD correction")
            return T_delta @ T_noisy