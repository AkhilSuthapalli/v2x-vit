import numpy as np
from scipy.optimize import linear_sum_assignment

class RobustCornerICPAligner:
    """
    Approach 3: Quality-Gated Bounding Box Corner ICP (SVD / Kabsch)
    Computes closed-form SE(2) relative pose correction with centroid matching,
    post-alignment residual gating, and deadband filtering.
    """
    def __init__(self, max_match_dist=2.5, min_boxes_required=2, max_residual_err=1.0):
        self.max_match_dist = max_match_dist
        self.min_boxes_required = min_boxes_required
        self.max_residual_err = max_residual_err  # Max allowed mean corner residual (meters)

    @staticmethod
    def _get_2d_corners(box):
        """Converts box [x, y, z, dx, dy, dz, yaw] to 4 2D BEV corner vertices (4, 2)."""
        x, y = box[0], box[1]
        length, width = box[3], box[4]
        yaw = box[6] if len(box) > 6 else box[4]

        l2, w2 = length / 2.0, width / 2.0
        base_corners = np.array([
            [-l2, -w2],
            [ l2, -w2],
            [ l2,  w2],
            [-l2,  w2]
        ], dtype=np.float64)

        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        return (base_corners @ R.T) + np.array([x, y], dtype=np.float64)

    def _match_box_pairs(self, ego_boxes, sender_boxes):
        """Hungarian bipartite assignment on box centroids within search radius."""
        s_centers = sender_boxes[:, :2]
        e_centers = ego_boxes[:, :2]
        dist_matrix = np.linalg.norm(s_centers[:, None, :] - e_centers[None, :, :], axis=-1)

        s_ind, e_ind = linear_sum_assignment(dist_matrix)

        valid_s_corners = []
        valid_e_corners = []

        for s_i, e_i in zip(s_ind, e_ind):
            if dist_matrix[s_i, e_i] <= self.max_match_dist:
                s_c = self._get_2d_corners(sender_boxes[s_i])
                e_c = self._get_2d_corners(ego_boxes[e_i])
                valid_s_corners.append(s_c)
                valid_e_corners.append(e_c)

        if len(valid_s_corners) < self.min_boxes_required:
            return None, None

        P_s = np.vstack(valid_s_corners)  # (4K, 2)
        P_e = np.vstack(valid_e_corners)  # (4K, 2)
        return P_s, P_e

    def align(self, ego_boxes, sender_boxes):
        if len(ego_boxes) < self.min_boxes_required or len(sender_boxes) < self.min_boxes_required:
            return np.eye(4, dtype=np.float64), (0.0, 0.0, 0.0)

        # 1. Hungarian Corner Pair Matching
        P_s, P_e = self._match_box_pairs(ego_boxes, sender_boxes)
        if P_s is None or P_e is None:
            return np.eye(4, dtype=np.float64), (0.0, 0.0, 0.0)

        # Pre-alignment mean corner error
        pre_err = np.mean(np.linalg.norm(P_s - P_e, axis=-1))

        # 2. Compute Centroids
        centroid_s = np.mean(P_s, axis=0)
        centroid_e = np.mean(P_e, axis=0)

        Q_s = P_s - centroid_s
        Q_e = P_e - centroid_e

        # 3. Cross-Covariance Matrix & SVD (Kabsch Algorithm)
        H = Q_s.T @ Q_e
        U, S, Vt = np.linalg.svd(H)
        R_2d = Vt.T @ U.T

        # Reflection check (ensures det(R) = +1)
        if np.linalg.det(R_2d) < 0:
            Vt[1, :] *= -1
            R_2d = Vt.T @ U.T

        opt_dtheta = np.arctan2(R_2d[1, 0], R_2d[0, 0])
        dt = centroid_e - (centroid_s @ R_2d.T)
        opt_dx, opt_dy = dt[0], dt[1]

        # 4. Post-Alignment Quality Gate Check
        P_s_aligned = (P_s @ R_2d.T) + dt
        post_err = np.mean(np.linalg.norm(P_s_aligned - P_e, axis=-1))

        # Reject transformation if residual error increases or exceeds safety threshold
        if post_err > pre_err or post_err > self.max_residual_err:
            return np.eye(4, dtype=np.float64), (0.0, 0.0, 0.0)

        # 5. Deadband Filter (Skip micro-adjustments)
        if abs(opt_dx) < 0.05 and abs(opt_dy) < 0.05 and abs(opt_dtheta) < np.radians(0.3):
            return np.eye(4, dtype=np.float64), (0.0, 0.0, 0.0)

        # 6. Construct SE(3) Homogeneous Matrix
        T_corr = np.eye(4, dtype=np.float64)
        T_corr[0, 0], T_corr[0, 1] = R_2d[0, 0], R_2d[0, 1]
        T_corr[1, 0], T_corr[1, 1] = R_2d[1, 0], R_2d[1, 1]
        T_corr[0, 3] = opt_dx
        T_corr[1, 3] = opt_dy

        return T_corr, (opt_dx, opt_dy, opt_dtheta)