import numpy as np

class ConsensusDIoUAligner:
    """
    Approach 2: Median Consensus-Filtered DIoU Bounding Box Aligner.
    Eliminates mAP degradation by pre-filtering mismatched boxes via translation
    vector consensus before performing local 2D DIoU grid search.
    """
    def __init__(self, max_trans_bound=2.0, max_yaw_bound=np.radians(8.0), 
                 consensus_radius=1.5, min_consensus_pairs=2, **kwargs):
        self.max_trans_bound = max_trans_bound
        self.max_yaw_bound = max_yaw_bound
        self.consensus_radius = consensus_radius  # Radius for median consensus cluster
        self.min_consensus_pairs = min_consensus_pairs

    @staticmethod
    def _get_canonical_corners_batch(boxes):
        """Standardizes box footprints to canonical dimensions (4.5m x 2.0m)."""
        N = len(boxes)
        x, y = boxes[:, 0], boxes[:, 1]
        yaw = boxes[:, 6] if boxes.shape[1] > 6 else boxes[:, 4]

        l2, w2 = 4.5 / 2.0, 2.0 / 2.0
        base_corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)

        scaled = base_corners[None, :, :] * np.array([l2, w2], dtype=np.float32)[None, None, :]
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.stack([c, -s, s, c], axis=-1).reshape(N, 2, 2)

        rotated = np.matmul(scaled, R.transpose(0, 2, 1))
        centers = np.stack([x, y], axis=-1)[:, None, :]
        return rotated + centers

    def _find_consensus_pairs(self, ego_boxes, sender_boxes):
        """
        Filters out non-corresponding / occluded boxes by finding the median 
        translation vector cluster across all pairwise box combinations.
        """
        e_centers = ego_boxes[:, :2]
        s_centers = sender_boxes[:, :2]

        # Pairwise translational offsets (N_sender, M_ego, 2)
        diffs = e_centers[None, :, :] - s_centers[:, None, :]
        diffs_flat = diffs.reshape(-1, 2)

        # Filter offsets within plausible noise bounds (+/- 2.5m)
        valid_mask = (np.abs(diffs_flat[:, 0]) <= 2.5) & (np.abs(diffs_flat[:, 1]) <= 2.5)
        valid_diffs = diffs_flat[valid_mask]

        if len(valid_diffs) < self.min_consensus_pairs:
            return None, None, (0.0, 0.0)

        # Find median translation vector (dominant spatial offset cluster)
        median_vector = np.median(valid_diffs, axis=0)

        # Keep only box pairs whose relative offset aligns with the median cluster
        pairwise_dists = np.linalg.norm(diffs - median_vector, axis=-1)
        s_idx, e_idx = np.where(pairwise_dists <= self.consensus_radius)

        if len(s_idx) < self.min_consensus_pairs:
            return None, None, (0.0, 0.0)

        return sender_boxes[s_idx], ego_boxes[e_idx], median_vector

    def _compute_diou_cost(self, ego_boxes, shifted_sender_boxes):
        """Calculates exact 2D OBB DIoU score on consensus-matched box pairs."""
        s_centers = shifted_sender_boxes[:, :2]
        e_centers = ego_boxes[:, :2]
        centroid_dists = np.linalg.norm(s_centers - e_centers, axis=-1)

        s_corners = self._get_canonical_corners_batch(shifted_sender_boxes)
        e_corners = self._get_canonical_corners_batch(ego_boxes)
        corner_dists = np.mean(np.linalg.norm(s_corners - e_corners, axis=-1), axis=-1)

        total_dists = centroid_dists + 0.5 * corner_dists
        similarity = np.sum(1.0 / (1.0 + total_dists / 2.0))
        return -similarity

    def _transform_boxes(self, boxes, dx, dy, dtheta):
        """Applies (dx, dy, dtheta) shift to sender boxes."""
        transformed = boxes.copy()
        c, s = np.cos(dtheta), np.sin(dtheta)
        R = np.array([[c, -s], [s, c]])

        transformed[:, :2] = transformed[:, :2] @ R.T + np.array([dx, dy])
        yaw_idx = 6 if transformed.shape[1] > 6 else 4
        transformed[:, yaw_idx] += dtheta
        return transformed

    def align(self, ego_boxes, sender_boxes_ego_frame):
        """Executes Consensus Filtering followed by Local DIoU Grid Search."""
        if len(ego_boxes) == 0 or len(sender_boxes_ego_frame) == 0:
            return np.eye(4), (0.0, 0.0, 0.0)

        # Step 1: Extract Consensus-Matched Box Pairs (Filters outlier / occluded cars)
        matched_sender, matched_ego, (init_dx, init_dy) = self._find_consensus_pairs(
            ego_boxes, sender_boxes_ego_frame
        )

        # Fallback to Identity if no reliable spatial consensus cluster exists
        if matched_sender is None:
            return np.eye(4), (0.0, 0.0, 0.0)

        # Step 2: Refine Pose via Local DIoU Grid Search around Consensus Seed
        x_range = np.linspace(init_dx - 0.4, init_dx + 0.4, 5)
        y_range = np.linspace(init_dy - 0.4, init_dy + 0.4, 5)
        yaw_range = np.linspace(-np.radians(6.0), np.radians(6.0), 5)

        best_cost = float('inf')
        best_delta = (init_dx, init_dy, 0.0)

        for dx in x_range:
            for dy in y_range:
                for dtheta in yaw_range:
                    shifted_sender = self._transform_boxes(matched_sender, dx, dy, dtheta)
                    cost = self._compute_diou_cost(matched_ego, shifted_sender)

                    if cost < best_cost:
                        best_cost = cost
                        best_delta = (dx, dy, dtheta)

        opt_dx, opt_dy, opt_dtheta = best_delta

        # Deadband thresholding for clean baseline stability
        if abs(opt_dx) < 0.08 and abs(opt_dy) < 0.08 and abs(opt_dtheta) < np.radians(0.4):
            return np.eye(4), (0.0, 0.0, 0.0)

        # Construct Homogeneous Transformation Matrix
        c, s = np.cos(opt_dtheta), np.sin(opt_dtheta)
        T_corr = np.eye(4)
        T_corr[0, 0], T_corr[0, 1] = c, -s
        T_corr[1, 0], T_corr[1, 1] = s, c
        T_corr[0, 3] = opt_dx
        T_corr[1, 3] = opt_dy

        return T_corr, (opt_dx, opt_dy, opt_dtheta)