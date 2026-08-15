import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

class DiagnosticDIoUAligner:
    """
    Approach 2: High-Precision Hybrid Hungarian DIoU + Nelder-Mead Aligner
    Combines global Hungarian bipartite assignment, continuous Distance-IoU,
    and continuous Nelder-Mead simplex polishing to eliminate grid quantization.
    """
    def __init__(self, max_trans_bound=2.0, max_yaw_bound=np.radians(6.0), reg_lambda=0.01, debug=False):
        self.max_trans_bound = max_trans_bound
        self.max_yaw_bound = max_yaw_bound
        self.reg_lambda = reg_lambda
        self.debug = debug

    @staticmethod
    def _get_canonical_corners_batch(boxes):
        """Vectorized conversion of (N, 7) boxes to 4 2D BEV canonical corner coordinates."""
        N = len(boxes)
        x, y = boxes[:, 0], boxes[:, 1]
        yaw = boxes[:, 6] if boxes.shape[1] > 6 else boxes[:, 4]

        # Standard canonical vehicle footprint (4.5m x 2.0m)
        l2, w2 = 4.5 / 2.0, 2.0 / 2.0
        base_corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)

        scaled = base_corners[None, :, :] * np.array([l2, w2], dtype=np.float64)[None, None, :]
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.stack([c, -s, s, c], axis=-1).reshape(N, 2, 2)

        rotated = np.matmul(scaled, R.transpose(0, 2, 1))
        centers = np.stack([x, y], axis=-1)[:, None, :]
        return rotated + centers

    def _transform_boxes(self, boxes, dx, dy, dtheta):
        """Applies continuous (dx, dy, dtheta) offset to sender bounding boxes."""
        transformed = boxes.copy()
        c, s = np.cos(dtheta), np.sin(dtheta)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)

        transformed[:, :2] = transformed[:, :2] @ R.T + np.array([dx, dy], dtype=np.float64)
        yaw_idx = 6 if transformed.shape[1] > 6 else 4
        transformed[:, yaw_idx] += dtheta
        return transformed

    def _compute_cost(self, delta, ego_boxes, sender_boxes):
        """
        Continuous Hungarian Distance-IoU Objective Function.
        Evaluates 1-to-1 optimal assignment between ego and shifted sender boxes.
        """
        dx, dy, dtheta = delta

        # Hard boundary barrier
        if abs(dx) > self.max_trans_bound or abs(dy) > self.max_trans_bound or abs(dtheta) > self.max_yaw_bound:
            return 1e5

        shifted_sender = self._transform_boxes(sender_boxes, dx, dy, dtheta)

        s_centers = shifted_sender[:, :2]
        e_centers = ego_boxes[:, :2]

        # Centroid distance matrix (N, M)
        centroid_dists = np.linalg.norm(s_centers[:, None, :] - e_centers[None, :, :], axis=-1)

        # Corner geometry distance matrix (N, M)
        s_corners = self._get_canonical_corners_batch(shifted_sender)
        e_corners = self._get_canonical_corners_batch(ego_boxes)
        corner_dists = np.mean(
            np.linalg.norm(s_corners[:, None, :, :] - e_corners[None, :, :, :], axis=-1),
            axis=-1
        )

        # Combined Chamfer-DIoU Cost Matrix
        cost_matrix = centroid_dists + 0.6 * corner_dists

        # Optimal 1-to-1 Bipartite Matching (Prevents Greedy Collapse)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_costs = cost_matrix[row_ind, col_ind]

        # Robust Inlier Gating: downweight outliers > 4.0m
        robust_weights = 1.0 / (1.0 + (matched_costs / 2.0)**2)
        total_loss = np.sum(matched_costs * robust_weights)

        # L2 Regularizer to prevent extreme drifts
        reg_penalty = self.reg_lambda * (dx**2 + dy**2 + (np.degrees(dtheta))**2)
        return total_loss + reg_penalty

    def align(self, ego_boxes, sender_boxes_ego_frame):
        """
        Executes Two-Stage Approach 2 Alignment:
        1. Coarse Global Grid Search (finds global basin).
        2. Continuous Bounded Nelder-Mead Simplex (eliminates quantization errors).
        """
        if len(ego_boxes) == 0 or len(sender_boxes_ego_frame) == 0:
            return np.eye(4, dtype=np.float64), (0.0, 0.0, 0.0)

        # Neighborhood pre-filter (6.0m search window)
        candidate_sender = []
        for s_box in sender_boxes_ego_frame:
            dists = np.hypot(ego_boxes[:, 0] - s_box[0], ego_boxes[:, 1] - s_box[1])
            if np.min(dists) <= 6.0:
                candidate_sender.append(s_box)

        if len(candidate_sender) == 0:
            return np.eye(4, dtype=np.float64), (0.0, 0.0, 0.0)

        candidate_sender = np.array(candidate_sender)

        # -------------------------------------------------------------
        # STAGE 1: Coarse Grid Warm-Start (Locate Global Basin)
        # -------------------------------------------------------------
        coarse_x = np.linspace(-1.2, 1.2, 5)
        coarse_y = np.linspace(-1.2, 1.2, 5)
        coarse_yaw = np.linspace(-np.radians(3.0), np.radians(3.0), 5)

        best_cost = float('inf')
        best_seed = (0.0, 0.0, 0.0)

        for dx in coarse_x:
            for dy in coarse_y:
                for dtheta in coarse_yaw:
                    cost = self._compute_cost((dx, dy, dtheta), ego_boxes, candidate_sender)
                    if cost < best_cost:
                        best_cost = cost
                        best_seed = (dx, dy, dtheta)

        # -------------------------------------------------------------
        # STAGE 2: Continuous Nelder-Mead Simplex Polish
        # -------------------------------------------------------------
        x0 = np.array(best_seed, dtype=np.float64)
        step_x, step_y, step_theta = 0.15, 0.15, np.radians(0.75)
        
        custom_simplex = np.array([
            x0,
            x0 + [step_x, 0.0, 0.0],
            x0 + [0.0, step_y, 0.0],
            x0 + [0.0, 0.0, step_theta]
        ])

        res = minimize(
            self._compute_cost,
            x0=x0,
            args=(ego_boxes, candidate_sender),
            method='Nelder-Mead',
            options={
                'initial_simplex': custom_simplex,
                'maxiter': 40,
                'xatol': 1e-3,  # 1mm spatial resolution
                'fatol': 1e-3
            }
        )

        opt_dx, opt_dy, opt_dtheta = res.x

        # Deadband filter to protect clean baseline frames
        if abs(opt_dx) < 0.03 and abs(opt_dy) < 0.03 and abs(opt_dtheta) < np.radians(0.15):
            return np.eye(4, dtype=np.float64), (0.0, 0.0, 0.0)

        # Construct 4x4 Homogeneous SE(3) Transformation Matrix
        c, s = np.cos(opt_dtheta), np.sin(opt_dtheta)
        T_corr = np.eye(4, dtype=np.float64)
        T_corr[0, 0], T_corr[0, 1] = c, -s
        T_corr[1, 0], T_corr[1, 1] = s, c
        T_corr[0, 3] = opt_dx
        T_corr[1, 3] = opt_dy

        return T_corr, (opt_dx, opt_dy, opt_dtheta)