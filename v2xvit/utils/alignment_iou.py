import numpy as np
from scipy.optimize import linear_sum_assignment

class RobustVectorizedDIoUAligner:
    """
    Approach 2: Regularized Canonical-Box DIoU Aligner
    Uses canonical box sizing, Hungarian matching, and L2 shift regularization
    to eliminate proposal dimension noise and prevent volatile delta jumps.
    """
    def __init__(self, max_trans_bound=2.0, max_yaw_bound=np.radians(8.0), 
                 min_quality_gain=0.04, reg_lambda=0.08, debug=False, **kwargs):
        self.max_trans_bound = max_trans_bound
        self.max_yaw_bound = max_yaw_bound
        self.min_quality_gain = min_quality_gain
        self.reg_lambda = reg_lambda  # L2 regularization weight
        self.debug = debug

    @staticmethod
    def _get_canonical_corners_batch(boxes):
        """
        Converts (N, 7) boxes into 2D corner vertices using standard CANONICAL 
        vehicle dimensions (length=4.5m, width=2.0m) to remove local detection noise.
        """
        N = len(boxes)
        x, y = boxes[:, 0], boxes[:, 1]
        yaw = boxes[:, 6] if boxes.shape[1] > 6 else boxes[:, 4]

        # Standard canonical car footprint
        l2, w2 = 4.5 / 2.0, 2.0 / 2.0

        base_corners = np.array([
            [-1, -1],
            [ 1, -1],
            [ 1,  1],
            [-1,  1]
        ], dtype=np.float32)

        scaled_corners = base_corners[None, :, :] * np.array([l2, w2], dtype=np.float32)[None, None, :]
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.stack([c, -s, s, c], axis=-1).reshape(N, 2, 2)

        rotated = np.matmul(scaled_corners, R.transpose(0, 2, 1))
        centers = np.stack([x, y], axis=-1)[:, None, :]

        return rotated + centers

    def _compute_hungarian_diou_cost(self, ego_boxes, shifted_sender_boxes, dx, dy, dtheta):
        """
        Calculates Hungarian 1-to-1 DIoU assignment score with L2 shift regularization.
        """
        s_centers = shifted_sender_boxes[:, :2]
        e_centers = ego_boxes[:, :2]
        centroid_dists = np.linalg.norm(s_centers[:, None, :] - e_centers[None, :, :], axis=-1)

        s_corners = self._get_canonical_corners_batch(shifted_sender_boxes)
        e_corners = self._get_canonical_corners_batch(ego_boxes)

        corner_dists = np.mean(
            np.linalg.norm(s_corners[:, None, :, :] - e_corners[None, :, :, :], axis=-1),
            axis=-1
        )

        obb_dists = centroid_dists + 0.5 * corner_dists  # Pairwise distance matrix (N, M)

        # Hungarian Bipartite Matching for unique 1-to-1 pairing
        row_ind, col_ind = linear_sum_assignment(obb_dists)
        matched_dists = obb_dists[row_ind, col_ind]

        # Base similarity score
        similarity = np.sum(1.0 / (1.0 + matched_dists / 2.0))
        mean_quality = np.mean(1.0 / (1.0 + matched_dists / 2.0))

        # L2 Regularization penalty: penalizes large shift deltas
        shift_penalty = self.reg_lambda * (dx**2 + dy**2 + (np.degrees(dtheta) / 4.0)**2)
        
        total_cost = -similarity + shift_penalty
        return total_cost, mean_quality

    def _transform_boxes(self, boxes, dx, dy, dtheta):
        """Applies spatial delta offset (dx, dy, dtheta) to sender boxes."""
        transformed = boxes.copy()
        c, s = np.cos(dtheta), np.sin(dtheta)
        R = np.array([[c, -s], [s, c]])

        transformed[:, :2] = transformed[:, :2] @ R.T + np.array([dx, dy])
        yaw_idx = 6 if transformed.shape[1] > 6 else 4
        transformed[:, yaw_idx] += dtheta
        return transformed

    def _evaluate_grid(self, ego_boxes, sender_boxes, x_range, y_range, yaw_range):
        """Evaluates 2D OBB search grid using Hungarian assignment."""
        best_cost = float('inf')
        best_delta = (0.0, 0.0, 0.0)
        best_quality = 0.0

        for dx in x_range:
            for dy in y_range:
                for dtheta in yaw_range:
                    shifted_sender = self._transform_boxes(sender_boxes, dx, dy, dtheta)
                    cost, quality = self._compute_hungarian_diou_cost(
                        ego_boxes, shifted_sender, dx, dy, dtheta
                    )

                    if cost < best_cost:
                        best_cost = cost
                        best_delta = (dx, dy, dtheta)
                        best_quality = quality

        return best_delta, best_quality

    def align(self, ego_boxes, sender_boxes_ego_frame):
        """Performs Regularized Coarse-to-Fine DIoU Grid Search."""
        if len(ego_boxes) < 1 or len(sender_boxes_ego_frame) < 1:
            return np.eye(4), (0.0, 0.0, 0.0)

        # Distance Pre-filtering (6.0m neighborhood radius)
        candidate_sender = []
        for s_box in sender_boxes_ego_frame:
            dists = np.hypot(ego_boxes[:, 0] - s_box[0], ego_boxes[:, 1] - s_box[1])
            if np.min(dists) <= 6.0:
                candidate_sender.append(s_box)

        if len(candidate_sender) < 1:
            return np.eye(4), (0.0, 0.0, 0.0)

        candidate_sender = np.array(candidate_sender)

        # Initial baseline quality at (0, 0, 0)
        _, init_quality = self._compute_hungarian_diou_cost(
            ego_boxes, candidate_sender, 0.0, 0.0, 0.0
        )

        # Pass 1: Coarse Grid Search
        coarse_x = np.linspace(-1.2, 1.2, 5)
        coarse_y = np.linspace(-1.2, 1.2, 5)
        coarse_yaw = np.linspace(-np.radians(6.0), np.radians(6.0), 5)

        (c_dx, c_dy, c_yaw), coarse_quality = self._evaluate_grid(
            ego_boxes, candidate_sender, coarse_x, coarse_y, coarse_yaw
        )

        # Pass 2: Fine Grid Search around Coarse Seed
        fine_x = np.linspace(c_dx - 0.25, c_dx + 0.25, 5)
        fine_y = np.linspace(c_dy - 0.25, c_dy + 0.25, 5)
        fine_yaw = np.linspace(c_yaw - np.radians(1.5), c_yaw + np.radians(1.5), 5)

        (opt_dx, opt_dy, opt_dtheta), final_quality = self._evaluate_grid(
            ego_boxes, candidate_sender, fine_x, fine_y, fine_yaw
        )

        quality_gain = final_quality - init_quality

        # Quality gain gate check
        if quality_gain < self.min_quality_gain:
            return np.eye(4), (0.0, 0.0, 0.0)

        # Deadband threshold
        if abs(opt_dx) < 0.05 and abs(opt_dy) < 0.05 and abs(opt_dtheta) < np.radians(0.3):
            return np.eye(4), (0.0, 0.0, 0.0)

        c, s = np.cos(opt_dtheta), np.sin(opt_dtheta)
        T_corr = np.eye(4)
        T_corr[0, 0], T_corr[0, 1] = c, -s
        T_corr[1, 0], T_corr[1, 1] = s, c
        T_corr[0, 3] = opt_dx
        T_corr[1, 3] = opt_dy

        return T_corr, (opt_dx, opt_dy, opt_dtheta)