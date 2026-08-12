import numpy as np

class FastVectorizedDIoUAligner:
    """
    Optimized Approach 2: Vectorized Coarse-to-Fine Distance-IoU (DIoU) Aligner
    Maximizes 2D bounding box overlap in pure NumPy without SciPy or Shapely overhead.
    """
    def __init__(self, max_trans_bound=2.0, max_yaw_bound=np.radians(10.0)):
        self.max_trans_bound = max_trans_bound  # Strict search bound for 1.0m noise
        self.max_yaw_bound = max_yaw_bound      # Strict search bound for 8.0 deg noise

    @staticmethod
    def _compute_vectorized_diou(ego_boxes, shifted_sender_boxes):
        """
        Computes continuous Distance-IoU (DIoU) cost matrix between ego and sender boxes.
        DIoU = IoU - (rho^2 / c^2)
        """
        # Centers
        e_centers = ego_boxes[:, :2]          # (M, 2)
        s_centers = shifted_sender_boxes[:, :2] # (N, 2)

        # Pairwise centroid distance squared (rho^2) -> (N, M)
        rho_sq = np.sum((s_centers[:, None, :] - e_centers[None, :, :]) ** 2, axis=-1)

        # Dimensions
        e_dims = ego_boxes[:, 3:5]            # (M, 2) [length, width]
        s_dims = shifted_sender_boxes[:, 3:5]   # (N, 2) [length, width]

        # Enclosing box diagonal squared (c^2 approximation)
        dx_max = np.abs(s_centers[:, None, 0] - e_centers[None, :, 0]) + (s_dims[:, None, 0] + e_dims[None, :, 0]) / 2.0
        dy_max = np.abs(s_centers[:, None, 1] - e_centers[None, :, 1]) + (s_dims[:, None, 1] + e_dims[None, :, 1]) / 2.0
        c_sq = dx_max**2 + dy_max**2 + 1e-6

        # Continuous distance penalty (-rho^2 / c^2)
        dist_penalty = rho_sq / c_sq

        # Fast BEV Intersection Box Approximation
        inter_dx = np.maximum(0.0, (s_dims[:, None, 0] + e_dims[None, :, 0]) / 2.0 - np.abs(s_centers[:, None, 0] - e_centers[None, :, 0]))
        inter_dy = np.maximum(0.0, (s_dims[:, None, 1] + e_dims[None, :, 1]) / 2.0 - np.abs(s_centers[:, None, 1] - e_centers[None, :, 1]))
        inter_area = inter_dx * inter_dy

        s_area = (s_dims[:, 0] * s_dims[:, 1])[:, None]
        e_area = (e_dims[:, 0] * e_dims[:, 1])[None, :]
        union_area = s_area + e_area - inter_area + 1e-6

        iou = inter_area / union_area
        diou_score = iou - dist_penalty

        # Return negative average DIoU score of best matched box pairs
        best_matches = np.max(diou_score, axis=1)
        return -np.sum(best_matches)

    def _transform_boxes(self, boxes, dx, dy, dtheta):
        """Applies spatial offset (dx, dy, dtheta) to sender bounding boxes."""
        transformed = boxes.copy()
        c, s = np.cos(dtheta), np.sin(dtheta)
        R = np.array([[c, -s], [s, c]])

        transformed[:, :2] = transformed[:, :2] @ R.T + np.array([dx, dy])
        yaw_idx = 6 if transformed.shape[1] > 6 else 4
        transformed[:, yaw_idx] += dtheta
        return transformed

    def _evaluate_grid(self, ego_boxes, sender_boxes, x_range, y_range, yaw_range):
        """Evaluates a grid of candidate (dx, dy, dtheta) offsets in vectorized NumPy."""
        best_cost = float('inf')
        best_delta = (0.0, 0.0, 0.0)

        for dx in x_range:
            for dy in y_range:
                for dtheta in yaw_range:
                    shifted_sender = self._transform_boxes(sender_boxes, dx, dy, dtheta)
                    cost = self._compute_vectorized_diou(ego_boxes, shifted_sender)

                    if cost < best_cost:
                        best_cost = cost
                        best_delta = (dx, dy, dtheta)

        return best_delta, best_cost

    def align(self, ego_boxes, sender_boxes_ego_frame):
        """
        Solves relative pose T_corr using 2-Pass Coarse-to-Fine Vectorized DIoU Search.
        """
        if len(ego_boxes) == 0 or len(sender_boxes_ego_frame) == 0:
            return np.eye(4), (0.0, 0.0, 0.0)

        # Filter candidate sender boxes within 5m radius
        candidate_sender = []
        for s_box in sender_boxes_ego_frame:
            dists = np.hypot(ego_boxes[:, 0] - s_box[0], ego_boxes[:, 1] - s_box[1])
            if np.min(dists) <= 5.0:
                candidate_sender.append(s_box)

        if len(candidate_sender) == 0:
            return np.eye(4), (0.0, 0.0, 0.0)

        candidate_sender = np.array(candidate_sender)

        # Pass 1: Coarse Search Grid (Covers +/- 1.5m and +/- 8 deg)
        coarse_x = np.linspace(-1.5, 1.5, 5)
        coarse_y = np.linspace(-1.5, 1.5, 5)
        coarse_yaw = np.linspace(-np.radians(8.0), np.radians(8.0), 5)

        (c_dx, c_dy, c_yaw), _ = self._evaluate_grid(
            ego_boxes, candidate_sender, coarse_x, coarse_y, coarse_yaw
        )

        # Pass 2: Fine Search Grid around Coarse Seed (Fine resolution: +/- 0.3m, +/- 2 deg)
        fine_x = np.linspace(c_dx - 0.3, c_dx + 0.3, 5)
        fine_y = np.linspace(c_dy - 0.3, c_dy + 0.3, 5)
        fine_yaw = np.linspace(c_yaw - np.radians(2.0), c_yaw + np.radians(2.0), 5)

        (opt_dx, opt_dy, opt_dtheta), _ = self._evaluate_grid(
            ego_boxes, candidate_sender, fine_x, fine_y, fine_yaw
        )

        # Deadband safety filter for clean frames
        if abs(opt_dx) < 0.05 and abs(opt_dy) < 0.05 and abs(opt_dtheta) < np.radians(0.3):
            return np.eye(4), (0.0, 0.0, 0.0)

        # Construct SE(3) Homogeneous Matrix
        c, s = np.cos(opt_dtheta), np.sin(opt_dtheta)
        T_corr = np.eye(4)
        T_corr[0, 0], T_corr[0, 1] = c, -s
        T_corr[1, 0], T_corr[1, 1] = s, c
        T_corr[0, 3] = opt_dx
        T_corr[1, 3] = opt_dy

        return T_corr, (opt_dx, opt_dy, opt_dtheta)