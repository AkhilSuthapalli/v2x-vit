import numpy as np
from scipy.optimize import linear_sum_assignment

class CornerICPAligner:
    """
    Approach 3: Bounding Box Corner ICP (SVD / Kabsch) with Diagnostic Logging.
    Extracts 2D BEV corners from matching bounding box proposals, solves for SE(2)
    transformation via closed-form SVD, and logs quality metrics to terminal.
    """
    def __init__(self, max_match_dist=3.0, min_boxes_required=2, max_residual_err=1.2, debug=True):
        self.max_match_dist = max_match_dist
        self.min_boxes_required = min_boxes_required
        self.max_residual_err = max_residual_err
        self.debug = debug

    @staticmethod
    def _get_2d_corners(box):
        """Converts box [x, y, z, dx, dy, dz, yaw] into 4 2D BEV corner vertices shape (4, 2)."""
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
        """Hungarian assignment on box centroids within max search distance."""
        s_centers = sender_boxes[:, :2]
        e_centers = ego_boxes[:, :2]
        dist_matrix = np.linalg.norm(s_centers[:, None, :] - e_centers[None, :, :], axis=-1)

        s_ind, e_ind = linear_sum_assignment(dist_matrix)

        valid_s_pts = []
        valid_e_pts = []

        for s_i, e_i in zip(s_ind, e_ind):
            if dist_matrix[s_i, e_i] <= self.max_match_dist:
                s_corners = self._get_2d_corners(sender_boxes[s_i])
                e_corners = self._get_2d_corners(ego_boxes[e_i])
                valid_s_pts.append(s_corners)
                valid_e_pts.append(e_corners)

        if len(valid_s_pts) < self.min_boxes_required:
            return None, None

        P_s = np.vstack(valid_s_pts)
        P_e = np.vstack(valid_e_pts)
        return P_s, P_e

    def align(self, ego_boxes, sender_boxes, T_gt_noise=None):
        if self.debug:
            print(f"\n[DEBUG ALIGNER] Input Box Counts -> Ego: {len(ego_boxes)}, Sender: {len(sender_boxes)}")

        # 1. Box Count Check
        if len(ego_boxes) < self.min_boxes_required or len(sender_boxes) < self.min_boxes_required:
            if self.debug:
                print(f"[DEBUG ALIGNER] -> REJECTED: Fewer than {self.min_boxes_required} boxes available. Returning Identity.")
            return np.eye(4, dtype=np.float64), (0.0, 0.0, 0.0)

        # 2. Match Bounding Box Corners
        P_s, P_e = self._match_box_pairs(ego_boxes, sender_boxes)
        if P_s is None or P_e is None:
            if self.debug:
                print(f"[DEBUG ALIGNER] -> REJECTED: Hungarian matching yielded fewer than {self.min_boxes_required} pairs within {self.max_match_dist}m radius. Returning Identity.")
            return np.eye(4, dtype=np.float64), (0.0, 0.0, 0.0)

        # Pre-alignment mean corner residual error
        pre_err = np.mean(np.linalg.norm(P_s - P_e, axis=-1))

        # 3. Compute Centroids
        centroid_s = np.mean(P_s, axis=0)
        centroid_e = np.mean(P_e, axis=0)

        Q_s = P_s - centroid_s
        Q_e = P_e - centroid_e

        # 4. Cross-Covariance & SVD (Kabsch)
        H = Q_s.T @ Q_e
        U, S, Vt = np.linalg.svd(H)
        R_2d = Vt.T @ U.T

        if np.linalg.det(R_2d) < 0:
            Vt[1, :] *= -1
            R_2d = Vt.T @ U.T

        opt_dtheta = np.arctan2(R_2d[1, 0], R_2d[0, 0])
        dt = centroid_e - (centroid_s @ R_2d.T)
        opt_dx, opt_dy = dt[0], dt[1]

        # Post-alignment mean corner residual error
        P_s_aligned = (P_s @ R_2d.T) + dt
        post_err = np.mean(np.linalg.norm(P_s_aligned - P_e, axis=-1))

        if self.debug:
            print(f"[DEBUG ALIGNER] Corner Residual Error -> Pre-align: {pre_err:.4f}m | Post-align: {post_err:.4f}m")
            print(f"[DEBUG ALIGNER] Solved Delta -> dx: {opt_dx:.3f}m, dy: {opt_dy:.3f}m, yaw: {np.degrees(opt_dtheta):.2f}°")

        # 5. Quality Gate Checks
        if post_err > pre_err:
            if self.debug:
                print(f"[DEBUG ALIGNER] -> REJECTED BY GATE: Post-align residual ({post_err:.4f}m) > Pre-align ({pre_err:.4f}m). Returning Identity.")
            return np.eye(4, dtype=np.float64), (0.0, 0.0, 0.0)

        if post_err > self.max_residual_err:
            if self.debug:
                print(f"[DEBUG ALIGNER] -> REJECTED BY GATE: Post-align residual ({post_err:.4f}m) > Max limit ({self.max_residual_err:.4f}m). Returning Identity.")
            return np.eye(4, dtype=np.float64), (0.0, 0.0, 0.0)

        # 6. Deadband Threshold Filter
        if abs(opt_dx) < 0.05 and abs(opt_dy) < 0.05 and abs(opt_dtheta) < np.radians(0.3):
            if self.debug:
                print(f"[DEBUG ALIGNER] -> REJECTED BY DEADBAND: Sub-5cm / Sub-0.3deg adjustment. Returning Identity.")
            return np.eye(4, dtype=np.float64), (0.0, 0.0, 0.0)

        if self.debug:
            print(f"[DEBUG ALIGNER] -> ACCEPTED: Applying T_corr [dx={opt_dx:.3f}m, dy={opt_dy:.3f}m, yaw={np.degrees(opt_dtheta):.2f}°]")

        # 7. Construct Homogeneous SE(3) Matrix
        T_corr = np.eye(4, dtype=np.float64)
        T_corr[0, 0], T_corr[0, 1] = R_2d[0, 0], R_2d[0, 1]
        T_corr[1, 0], T_corr[1, 1] = R_2d[1, 0], R_2d[1, 1]
        T_corr[0, 3] = opt_dx
        T_corr[1, 3] = opt_dy

        # 8. Compare against True Ground-Truth Noise if available
        if T_gt_noise is not None and self.debug:
            gt_dx, gt_dy = T_gt_noise[0, 3], T_gt_noise[1, 3]
            gt_yaw = np.arctan2(T_gt_noise[1, 0], T_gt_noise[0, 0])

            res_x = abs(gt_dx + opt_dx)
            res_y = abs(gt_dy + opt_dy)
            res_yaw = abs(np.degrees(gt_yaw + opt_dtheta))

            print(f"[DIAGNOSTIC] True GT Noise -> dx: {-gt_dx:.3f}m, dy: {-gt_dy:.3f}m, yaw: {-np.degrees(gt_yaw):.2f}°")
            print(f"[DIAGNOSTIC] Residual Error -> X: {res_x:.3f}m, Y: {res_y:.3f}m, Yaw: {res_yaw:.2f}°")

        return T_corr, (opt_dx, opt_dy, opt_dtheta)