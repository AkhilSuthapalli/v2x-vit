# File path: v2xvit/utils/spatial_aligner.py

import numpy as np

class CentroidConsensusAligner:
    """
    Approach 1: Bounding Box Centroid Consensus Alignment for V2X-ViT.
    """
    def __init__(self, method="svd", max_match_dist=4.0, min_match_dist=0.05):
        self.method = method
        self.max_match_dist = max_match_dist
        self.min_match_dist = min_match_dist

    def match_centroids(self, ego_centers, sender_centers_in_ego):
        if len(ego_centers) == 0 or len(sender_centers_in_ego) == 0:
            return np.empty((0, 2)), np.empty((0, 2))

        diff = ego_centers[:, None, :2] - sender_centers_in_ego[None, :, :2]
        dist_matrix = np.linalg.norm(diff, axis=-1)

        matched_ego = []
        matched_snd = []
        
        for i in range(len(ego_centers)):
            min_idx = np.argmin(dist_matrix[i])
            min_dist = dist_matrix[i, min_idx]

            if self.min_match_dist <= min_dist <= self.max_match_dist:
                matched_ego.append(ego_centers[i, :2])
                matched_snd.append(sender_centers_in_ego[min_idx, :2])

        return np.array(matched_ego), np.array(matched_snd)

    def compute_rigid_delta(self, matched_ego, matched_snd):
        T_delta = np.identity(4, dtype=np.float64)

        if len(matched_ego) == 0:
            return T_delta

        if self.method == "mean" or len(matched_ego) < 2:
            shift = np.mean(matched_ego - matched_snd, axis=0)
            T_delta[0, 3] = shift[0]
            T_delta[1, 3] = shift[1]

        elif self.method == "svd":
            c_ego = np.mean(matched_ego, axis=0)
            c_snd = np.mean(matched_snd, axis=0)

            p_ego = matched_ego - c_ego
            p_snd = matched_snd - c_snd

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

        return T_delta

    def correct_pose_matrix(self, T_noisy, ego_boxes, sender_boxes):
        if ego_boxes is None or sender_boxes is None:
            return T_noisy

        ego_pts = np.asarray(ego_boxes)
        snd_pts = np.asarray(sender_boxes)

        if len(ego_pts) == 0 or len(snd_pts) == 0:
            return T_noisy

        # Project sender centroids using noisy T matrix
        snd_3d = snd_pts[:, :3] if snd_pts.shape[1] >= 3 else np.hstack((snd_pts[:, :2], np.zeros((len(snd_pts), 1))))
        snd_homo = np.hstack((snd_3d, np.ones((len(snd_3d), 1))))
        snd_in_ego = (T_noisy @ snd_homo.T).T[:, :3]

        m_ego, m_snd = self.match_centroids(ego_pts, snd_in_ego)

        T_delta = self.compute_rigid_delta(m_ego, m_snd)

        return T_delta @ T_noisy