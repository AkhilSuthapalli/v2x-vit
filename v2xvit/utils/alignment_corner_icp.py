import numpy as np
from scipy.optimize import linear_sum_assignment

class BoxCornerSVDAligner:
    """
    Approach 3: Bounding Box Corner SVD Aligner (Debug Instrumented)
    Extracts 2D corner keypoints from matching co-observed boxes.
    Solves for (R, T) in a single closed-form SVD step.
    """
    def __init__(self, max_match_dist=6.0, min_boxes_required=2, debug=True):
        self.max_match_dist = max_match_dist
        self.min_boxes_required = min_boxes_required
        self.debug = debug

    @staticmethod
    def _get_canonical_corners(box):
        """Converts box [x, y, z, dx, dy, dz, yaw] to 4 2D BEV canonical corners."""
        x, y = box[0], box[1]
        yaw = box[6] if len(box) > 6 else box[4]
        
        # Canonical dimensions
        l2, w2 = 4.5 / 2.0, 2.0 / 2.0
        
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
        s_centers = sender_boxes[:, :2]
        e_centers = ego_boxes[:, :2]
        
        dist_matrix = np.linalg.norm(s_centers[:, None, :] - e_centers[None, :, :], axis=-1)
        s_ind, e_ind = linear_sum_assignment(dist_matrix)

        valid_s_pts, valid_e_pts = [], []

        for s_i, e_i in zip(s_ind, e_ind):
            if dist_matrix[s_i, e_i] <= self.max_match_dist:
                valid_s_pts.append(self._get_canonical_corners(sender_boxes[s_i]))
                valid_e_pts.append(self._get_canonical_corners(ego_boxes[e_i]))

        if len(valid_s_pts) < self.min_boxes_required:
            return None, None

        P_s = np.vstack(valid_s_pts)
        P_e = np.vstack(valid_e_pts)
        return P_s, P_e

    def align(self, ego_boxes, sender_boxes, cav_id="Unknown"):
        if self.debug:
            print(f"\n[DEBUG SVD ALIGNER] === Processing CAV {cav_id} ===")
            print(f"[DEBUG SVD ALIGNER] Input Ego Boxes: {len(ego_boxes)} | Input Sender Boxes: {len(sender_boxes)}")

        if len(ego_boxes) < self.min_boxes_required or len(sender_boxes) < self.min_boxes_required:
            if self.debug:
                print("[DEBUG SVD ALIGNER] REJECTED: Not enough boxes to process. Returning Identity.")
            return np.eye(4, dtype=np.float64), (0.0, 0.0, 0.0)

        P_s, P_e = self._match_box_pairs(ego_boxes, sender_boxes)
        if P_s is None:
            if self.debug:
                print(f"[DEBUG SVD ALIGNER] REJECTED: Could not find {self.min_boxes_required} valid matches within {self.max_match_dist}m.")
            return np.eye(4, dtype=np.float64), (0.0, 0.0, 0.0)

        if self.debug:
            print(f"[DEBUG SVD ALIGNER] Successfully matched {len(P_s)//4} boxes.")

        # Compute Centroids
        centroid_s = np.mean(P_s, axis=0)
        centroid_e = np.mean(P_e, axis=0)

        # Cross-Covariance Matrix and SVD (Kabsch)
        H = (P_s - centroid_s).T @ (P_e - centroid_e)
        U, _, Vt = np.linalg.svd(H)
        R_2d = Vt.T @ U.T

        if np.linalg.det(R_2d) < 0:
            Vt[1, :] *= -1
            R_2d = Vt.T @ U.T

        opt_dtheta = np.arctan2(R_2d[1, 0], R_2d[0, 0])
        dt = centroid_e - (centroid_s @ R_2d.T)
        opt_dx, opt_dy = dt[0], dt[1]

        if self.debug:
            print(f"[DEBUG SVD ALIGNER] Solved Offsets -> dx: {opt_dx:.3f}m, dy: {opt_dy:.3f}m, yaw: {np.degrees(opt_dtheta):.2f}°")

        T_corr = np.eye(4, dtype=np.float64)
        T_corr[:2, :2] = R_2d
        T_corr[0, 3] = opt_dx
        T_corr[1, 3] = opt_dy

        return T_corr, (opt_dx, opt_dy, opt_dtheta)