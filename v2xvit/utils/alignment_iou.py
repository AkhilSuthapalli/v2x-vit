import numpy as np

class DiagnosticDIoUAligner:
    """
    Approach 2: High-Precision Sub-Voxel DIoU Aligner
    Uses a 2-pass coarse-to-fine search grid with 5cm translation resolution
    and 0.25 deg yaw resolution to eliminate discretization overshooting.
    """
    def __init__(self, max_trans_bound=2.0, max_yaw_bound=np.radians(8.0), debug=True):
        self.max_trans_bound = max_trans_bound
        self.max_yaw_bound = max_yaw_bound
        self.debug = debug

    @staticmethod
    def _get_canonical_corners_batch(boxes):
        N = len(boxes)
        x, y = boxes[:, 0], boxes[:, 1]
        yaw = boxes[:, 6] if boxes.shape[1] > 6 else boxes[:, 4]

        # Standard canonical vehicle footprint (4.5m x 2.0m)
        l2, w2 = 4.5 / 2.0, 2.0 / 2.0
        base_corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)

        scaled = base_corners[None, :, :] * np.array([l2, w2], dtype=np.float32)[None, None, :]
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.stack([c, -s, s, c], axis=-1).reshape(N, 2, 2)

        rotated = np.matmul(scaled, R.transpose(0, 2, 1))
        centers = np.stack([x, y], axis=-1)[:, None, :]
        return rotated + centers

    def _compute_diou_cost(self, ego_boxes, shifted_sender_boxes):
        s_centers = shifted_sender_boxes[:, :2]
        e_centers = ego_boxes[:, :2]
        centroid_dists = np.linalg.norm(s_centers[:, None, :] - e_centers[None, :, :], axis=-1)

        s_corners = self._get_canonical_corners_batch(shifted_sender_boxes)
        e_corners = self._get_canonical_corners_batch(ego_boxes)
        corner_dists = np.mean(np.linalg.norm(s_corners[:, None, :, :] - e_corners[None, :, :, :], axis=-1), axis=-1)

        total_dists = centroid_dists + 0.5 * corner_dists
        best_matches = np.min(total_dists, axis=1)

        similarity = np.sum(1.0 / (1.0 + best_matches / 2.0))
        quality = np.mean(1.0 / (1.0 + best_matches / 2.0))
        return -similarity, quality

    def _transform_boxes(self, boxes, dx, dy, dtheta):
        transformed = boxes.copy()
        c, s = np.cos(dtheta), np.sin(dtheta)
        R = np.array([[c, -s], [s, c]])

        transformed[:, :2] = transformed[:, :2] @ R.T + np.array([dx, dy])
        yaw_idx = 6 if transformed.shape[1] > 6 else 4
        transformed[:, yaw_idx] += dtheta
        return transformed

    def _evaluate_grid(self, ego_boxes, sender_boxes, x_range, y_range, yaw_range):
        best_cost = float('inf')
        best_delta = (0.0, 0.0, 0.0)
        best_quality = 0.0

        for dx in x_range:
            for dy in y_range:
                for dtheta in yaw_range:
                    shifted = self._transform_boxes(sender_boxes, dx, dy, dtheta)
                    cost, quality = self._compute_diou_cost(ego_boxes, shifted)
                    if cost < best_cost:
                        best_cost = cost
                        best_delta = (dx, dy, dtheta)
                        best_quality = quality

        return best_delta, best_quality

    def align(self, ego_boxes, sender_boxes_ego_frame, T_gt_noise=None):
        if self.debug:
            print(f"\n[DEBUG ALIGNER] Input Box Counts -> Ego: {len(ego_boxes)}, Sender (Ego Frame): {len(sender_boxes_ego_frame)}")

        if len(ego_boxes) == 0 or len(sender_boxes_ego_frame) == 0:
            return np.eye(4), (0.0, 0.0, 0.0)

        # Distance Pre-filtering (6.0m search radius)
        candidate_sender = []
        for s_box in sender_boxes_ego_frame:
            dists = np.hypot(ego_boxes[:, 0] - s_box[0], ego_boxes[:, 1] - s_box[1])
            if np.min(dists) <= 6.0:
                candidate_sender.append(s_box)

        if len(candidate_sender) == 0:
            return np.eye(4), (0.0, 0.0, 0.0)

        candidate_sender = np.array(candidate_sender)
        _, init_quality = self._compute_diou_cost(ego_boxes, candidate_sender)

        # PASS 1: Coarse Search Grid (0.25m translation resolution, 1.5 deg yaw resolution)
        coarse_x = np.linspace(-1.5, 1.5, 13)
        coarse_y = np.linspace(-1.5, 1.5, 13)
        coarse_yaw = np.linspace(-np.radians(6.0), np.radians(6.0), 9)

        (c_dx, c_dy, c_yaw), coarse_quality = self._evaluate_grid(
            ego_boxes, candidate_sender, coarse_x, coarse_y, coarse_yaw
        )

        # PASS 2: High-Precision Fine Grid (0.05m / 5cm translation, 0.25 deg yaw resolution)
        fine_x = np.linspace(c_dx - 0.20, c_dx + 0.20, 9)
        fine_y = np.linspace(c_dy - 0.20, c_dy + 0.20, 9)
        fine_yaw = np.linspace(c_yaw - np.radians(1.0), c_yaw + np.radians(1.0), 9)

        (opt_dx, opt_dy, opt_dtheta), final_quality = self._evaluate_grid(
            ego_boxes, candidate_sender, fine_x, fine_y, fine_yaw
        )

        quality_gain = final_quality - init_quality

        if self.debug:
            print(f"[DEBUG ALIGNER] Quality -> Initial: {init_quality:.4f} | Final: {final_quality:.4f} | Gain: +{quality_gain:.4f}")
            print(f"[DEBUG ALIGNER] Solved Delta -> dx: {opt_dx:.3f}m, dy: {opt_dy:.3f}m, yaw: {np.degrees(opt_dtheta):.2f}°")

        # Deadband Safety Filter
        if abs(opt_dx) < 0.05 and abs(opt_dy) < 0.05 and abs(opt_dtheta) < np.radians(0.25):
            return np.eye(4), (0.0, 0.0, 0.0)

        c, s = np.cos(opt_dtheta), np.sin(opt_dtheta)
        T_corr = np.eye(4)
        T_corr[0, 0], T_corr[0, 1] = c, -s
        T_corr[1, 0], T_corr[1, 1] = s, c
        T_corr[0, 3] = opt_dx
        T_corr[1, 3] = opt_dy

        if T_gt_noise is not None and self.debug:
            gt_dx, gt_dy = T_gt_noise[0, 3], T_gt_noise[1, 3]
            gt_yaw = np.arctan2(T_gt_noise[1, 0], T_gt_noise[0, 0])

            res_x = abs(gt_dx + opt_dx)
            res_y = abs(gt_dy + opt_dy)
            res_yaw = abs(np.degrees(gt_yaw + opt_dtheta))

            print(f"[DIAGNOSTIC] True Noise Offset -> dx: {-gt_dx:.3f}m, dy: {-gt_dy:.3f}m, yaw: {-np.degrees(gt_yaw):.2f}°")
            print(f"[DIAGNOSTIC] Residual Error   -> X: {res_x:.3f}m, Y: {res_y:.3f}m, Yaw: {res_yaw:.2f}°")

        return T_corr, (opt_dx, opt_dy, opt_dtheta)