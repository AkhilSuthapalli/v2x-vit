import numpy as np

class RobustVectorizedDIoUAligner:
    """
    Upgraded Approach 2: Oriented Bounding Box (OBB) DIoU Grid Aligner
    Uses exact 2D rotated corner distance matching, proposal quality filtering,
    and confidence gating to prevent accuracy degradation.
    """
    def __init__(self, max_trans_bound=2.0, max_yaw_bound=np.radians(10.0), min_match_score=0.35):
        self.max_trans_bound = max_trans_bound  # +/- 2.0m translation limit
        self.max_yaw_bound = max_yaw_bound      # +/- 10.0 deg heading limit
        self.min_match_score = min_match_score  # Gate threshold to apply correction

    @staticmethod
    def _get_2d_corners_batch(boxes):
        """
        Vectorized conversion of (N, 7) boxes [x, y, z, dx, dy, dz, yaw] 
        into 4 oriented 2D corner points (N, 4, 2).
        """
        N = len(boxes)
        x, y = boxes[:, 0], boxes[:, 1]
        length, width = boxes[:, 3], boxes[:, 4]
        yaw = boxes[:, 6] if boxes.shape[1] > 6 else boxes[:, 4]

        l2, w2 = length / 2.0, width / 2.0

        # Unrotated corners centered at origin (4, 2)
        base_corners = np.array([
            [-1, -1],
            [ 1, -1],
            [ 1,  1],
            [-1,  1]
        ], dtype=np.float32)

        # Scale corners by length and width -> (N, 4, 2)
        scaled_corners = base_corners[None, :, :] * np.stack([l2, w2], axis=-1)[:, None, :]

        # Rotate and translate
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.stack([c, -s, s, c], axis=-1).reshape(N, 2, 2)  # (N, 2, 2)

        rotated = np.matmul(scaled_corners, R.transpose(0, 2, 1))  # (N, 4, 2)
        centers = np.stack([x, y], axis=-1)[:, None, :]           # (N, 1, 2)

        return rotated + centers  # (N, 4, 2)

    def _compute_obb_diou_cost(self, ego_boxes, shifted_sender_boxes):
        """
        Calculates exact Oriented Bounding Box (OBB) Chamfer-DIoU distance.
        """
        num_s = len(shifted_sender_boxes)
        num_e = len(ego_boxes)

        # 1. Centroid Distances (N, M)
        s_centers = shifted_sender_boxes[:, :2]
        e_centers = ego_boxes[:, :2]
        centroid_dists = np.linalg.norm(s_centers[:, None, :] - e_centers[None, :, :], axis=-1)

        # 2. Rotated Corner Distance Matching (N, M)
        s_corners = self._get_2d_corners_batch(shifted_sender_boxes)  # (N, 4, 2)
        e_corners = self._get_2d_corners_batch(ego_boxes)             # (M, 4, 2)

        # Mean Euclidean distance between 4 box corners (N, M)
        corner_dists = np.mean(
            np.linalg.norm(s_corners[:, None, :, :] - e_corners[None, :, :, :], axis=-1),
            axis=-1
        )

        # Total combined OBB distance score
        obb_dists = centroid_dists + 0.5 * corner_dists

        # For each sender box, select closest ego box
        best_matches = np.min(obb_dists, axis=1)

        # Convert distance to similarity score (higher is better)
        similarity_score = np.sum(1.0 / (1.0 + best_matches))
        return -similarity_score, np.mean(1.0 / (1.0 + best_matches))

    def _transform_boxes(self, boxes, dx, dy, dtheta):
        """Applies (dx, dy, dtheta) offset to sender bounding boxes."""
        transformed = boxes.copy()
        c, s = np.cos(dtheta), np.sin(dtheta)
        R = np.array([[c, -s], [s, c]])

        transformed[:, :2] = transformed[:, :2] @ R.T + np.array([dx, dy])
        yaw_idx = 6 if transformed.shape[1] > 6 else 4
        transformed[:, yaw_idx] += dtheta
        return transformed

    def _evaluate_grid(self, ego_boxes, sender_boxes, x_range, y_range, yaw_range):
        """Evaluates 2D OBB search grid in parallel NumPy operations."""
        best_cost = float('inf')
        best_delta = (0.0, 0.0, 0.0)
        best_quality = 0.0

        for dx in x_range:
            for dy in y_range:
                for dtheta in yaw_range:
                    shifted_sender = self._transform_boxes(sender_boxes, dx, dy, dtheta)
                    cost, quality = self._compute_obb_diou_cost(ego_boxes, shifted_sender)

                    if cost < best_cost:
                        best_cost = cost
                        best_delta = (dx, dy, dtheta)
                        best_quality = quality

        return best_delta, best_quality

    def align(self, ego_boxes, sender_boxes_ego_frame):
        """
        Performs 2-Pass Coarse-to-Fine OBB DIoU Grid Search with Quality Gate.
        """
        # REQUIREMENT 1: Must have at least 2 boxes in both views for reliable geometric matching
        if len(ego_boxes) < 2 or len(sender_boxes_ego_frame) < 2:
            return np.eye(4), (0.0, 0.0, 0.0)

        # Pre-filter candidate sender boxes within 6.0m search neighborhood
        candidate_sender = []
        for s_box in sender_boxes_ego_frame:
            dists = np.hypot(ego_boxes[:, 0] - s_box[0], ego_boxes[:, 1] - s_box[1])
            if np.min(dists) <= 6.0:
                candidate_sender.append(s_box)

        if len(candidate_sender) < 2:
            return np.eye(4), (0.0, 0.0, 0.0)

        candidate_sender = np.array(candidate_sender)

        # Pass 1: Coarse Grid Search (+/- 1.5m translation, +/- 8 deg heading)
        coarse_x = np.linspace(-1.5, 1.5, 5)
        coarse_y = np.linspace(-1.5, 1.5, 5)
        coarse_yaw = np.linspace(-np.radians(8.0), np.radians(8.0), 5)

        (c_dx, c_dy, c_yaw), coarse_quality = self._evaluate_grid(
            ego_boxes, candidate_sender, coarse_x, coarse_y, coarse_yaw
        )

        # Pass 2: Fine Grid Search around Coarse Seed (+/- 0.3m, +/- 2 deg)
        fine_x = np.linspace(c_dx - 0.3, c_dx + 0.3, 5)
        fine_y = np.linspace(c_dy - 0.3, c_dy + 0.3, 5)
        fine_yaw = np.linspace(c_yaw - np.radians(2.0), c_yaw + np.radians(2.0), 5)

        (opt_dx, opt_dy, opt_dtheta), final_quality = self._evaluate_grid(
            ego_boxes, candidate_sender, fine_x, fine_y, fine_yaw
        )

        # REQUIREMENT 2: Quality Match Gate (Reject correction if similarity is low)
        if final_quality < self.min_match_score:
            return np.eye(4), (0.0, 0.0, 0.0)

        # REQUIREMENT 3: Deadband Thresholding (Ignore sub-5cm micro-adjustments)
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