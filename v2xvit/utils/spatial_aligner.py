import numpy as np

class CentroidConsensusAligner:
    """
    Approach 1: Single-Centroid SVD Aligner with Raw Translation Gating
    and Rotation Lock for Low-Noise Stability.
    """
    def __init__(self, method="svd", max_match_dist=4.0, min_match_dist=0.05, deadband_thresh=0.35):
        self.method = method
        self.max_match_dist = max_match_dist
        self.min_match_dist = min_match_dist
        self.deadband_thresh = deadband_thresh  # Threshold in meters (0.35m)

    def match_centroids(self, ego_boxes, snd_in_ego_boxes):
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
        T_delta = np.identity(4, dtype=np.float64)

        if len(matched_ego) == 0:
            return T_delta, 0.0

        c_ego = np.mean(matched_ego, axis=0)
        c_snd = np.mean(matched_snd, axis=0)

        # 1. Raw spatial shift magnitude before fitting rotation
        raw_shift_magnitude = np.linalg.norm(c_ego - c_snd)

        p_ego = matched_ego - c_ego
        p_snd = matched_snd - c_snd

        # If shift is below deadband threshold, do not fit rotation or translation
        if raw_shift_magnitude < self.deadband_thresh:
            return T_delta, raw_shift_magnitude

        # 2. Fit SVD for larger shifts
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

        return T_delta, raw_shift_magnitude

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

        T_delta, raw_shift = self.compute_rigid_delta(m_ego, m_snd)

        # Print diagnostic info to verify filter behavior in terminal
        if raw_shift < self.deadband_thresh:
            print(f"[FILTER BYPASS] Raw shift: {raw_shift:.3f}m < {self.deadband_thresh}m -> Keeping T_noisy")
            return T_noisy
        else:
            print(f"[FILTER TRIGGERED] Raw shift: {raw_shift:.3f}m >= {self.deadband_thresh}m -> Applying SVD correction")
            return T_delta @ T_noisy