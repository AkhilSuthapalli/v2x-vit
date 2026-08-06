import numpy as np

class CentroidConsensusAligner:
    """
    Approach 1: Single-Centroid SVD Aligner with a Hard Deadband Filter.
    
    - For noise >= 0.3m: Applies full closed-form SVD registration.
    - For noise < 0.3m:  Bypasses correction to preserve V2X-ViT's native high-precision baseline.
    """
    def __init__(self, method="svd", max_match_dist=4.0, min_match_dist=0.05, deadband_thresh=0.30):
        self.method = method
        self.max_match_dist = max_match_dist
        self.min_match_dist = min_match_dist
        self.deadband_thresh = deadband_thresh  # Threshold in meters (0.30m)

    def match_centroids(self, ego_boxes, snd_in_ego_boxes):
        """
        Matches vehicle centroids between Ego and Sender using 2D BEV distance gating.
        """
        ego_centers = ego_boxes[:, :2]
        snd_centers = snd_in_ego_boxes[:, :2]

        diff = ego_centers[:, None, :] - snd_centers[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)

        matched_ego = []
        matched_snd = []

        for i in range(len(ego_centers)):
            min_idx = np.argmin(dist_matrix[i])
            min_dist = dist_matrix[i, min_idx]

            if self.min_match_dist <= min_dist <= self.max_match_dist:
                matched_ego.append(ego_centers[i])
                matched_snd.append(snd_centers[min_idx])

        return np.array(matched_ego), np.array(matched_snd)

    def compute_rigid_delta(self, matched_ego, matched_snd):
        """
        Solves closed-form SVD (Kabsch algorithm) for 2D rotation and translation.
        """
        T_delta = np.identity(4, dtype=np.float64)

        if len(matched_ego) == 0:
            return T_delta

        c_ego = np.mean(matched_ego, axis=0)
        c_snd = np.mean(matched_snd, axis=0)

        p_ego = matched_ego - c_ego
        p_snd = matched_snd - c_snd

        H = p_snd.T @ p_ego
        U, S, Vt = np.linalg.svd(H)
        R_2d = Vt.T @ U.T

        # Reflection handling
        if np.linalg.det(R_2d) < 0:
            Vt[1, :] *= -1
            R_2d = Vt.T @ U.T

        t_2d = c_ego - (R_2d @ c_snd)

        T_delta[0:2, 0:2] = R_2d
        T_delta[0, 3] = t_2d[0]
        T_delta[1, 3] = t_2d[1]

        return T_delta

    def correct_pose_matrix(self, T_noisy, ego_boxes, sender_boxes):
        """
        Computes spatial alignment correction with < 0.3m deadband check.
        """
        if ego_boxes is None or sender_boxes is None or len(ego_boxes) == 0 or len(sender_boxes) == 0:
            return T_noisy

        ego_boxes = np.asarray(ego_boxes)
        sender_boxes = np.asarray(sender_boxes)

        # 1. Project sender box centroids into Ego local coordinate space using T_noisy
        snd_3d = sender_boxes[:, :3] if sender_boxes.shape[1] >= 3 else np.hstack((sender_boxes[:, :2], np.zeros((len(sender_boxes), 1))))
        snd_homo = np.hstack((snd_3d, np.ones((len(snd_3d), 1))))
        snd_in_ego_centers = (T_noisy @ snd_homo.T).T[:, :3]

        snd_in_ego_boxes = sender_boxes.copy()
        snd_in_ego_boxes[:, :3] = snd_in_ego_centers

        # 2. Match co-observed vehicle centroids
        m_ego, m_snd = self.match_centroids(ego_boxes, snd_in_ego_boxes)

        if len(m_ego) == 0:
            return T_noisy

        # 3. Compute raw SVD rigid delta
        T_delta = self.compute_rigid_delta(m_ego, m_snd)

        # 4. --- HARD DEADBAND FILTER (< 0.3m) ---
        shift_magnitude = np.linalg.norm(T_delta[0:2, 3])

        if shift_magnitude < self.deadband_thresh:
            # Shift is below noise floor threshold; bypass alignment
            return T_noisy

        # 5. Apply correction matrix
        return T_delta @ T_noisy