import numpy as np
from scipy.optimize import linear_sum_assignment

class FastVectorizedIoUAligner:
    """
    Approach 2 (High-Speed): Vectorized Pseudo-IoU Grid Search.
    Eliminates Shapely polygon bottlenecks by using pure NumPy broadcasting
    to evaluate bounded overlap approximations.
    """
    def __init__(self, max_trans=1.2, max_yaw=np.radians(5.0), debug=False):
        self.max_trans = max_trans
        self.max_yaw = max_yaw
        self.debug = debug

    @staticmethod
    def _get_canonical_corners_batch(boxes):
        """Vectorized corner generation."""
        N = len(boxes)
        x, y = boxes[:, 0], boxes[:, 1]
        yaw = boxes[:, 6] if boxes.shape[1] > 6 else boxes[:, 4]

        l2, w2 = 4.5 / 2.0, 2.0 / 2.0
        base_corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64)

        scaled = base_corners[None, :, :] * np.array([l2, w2], dtype=np.float64)[None, None, :]
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.stack([c, -s, s, c], axis=-1).reshape(N, 2, 2)

        rotated = np.matmul(scaled, R.transpose(0, 2, 1))
        centers = np.stack([x, y], axis=-1)[:, None, :]
        return rotated + centers

    def _evaluate_grid_vectorized(self, ego_boxes, sender_boxes):
        """
        Evaluates a 5x5x5 grid (125 states) using pure NumPy.
        Returns the (dx, dy, dtheta) that minimizes Corner DIoU distance.
        """
        # Condensed 5x5x5 grid for speed
        dx_grid = np.linspace(-self.max_trans, self.max_trans, 5)
        dy_grid = np.linspace(-self.max_trans, self.max_trans, 5)
        dyaw_grid = np.linspace(-self.max_yaw, self.max_yaw, 5)

        e_corners = self._get_canonical_corners_batch(ego_boxes)
        
        best_cost = float('inf')
        best_delta = (0.0, 0.0, 0.0)

        # We still iterate the grid, but the inner math is heavily vectorized
        for dx in dx_grid:
            for dy in dy_grid:
                for dyaw in dyaw_grid:
                    # Shift sender boxes
                    shifted_sender = sender_boxes.copy()
                    c, s = np.cos(dyaw), np.sin(dyaw)
                    R = np.array([[c, -s], [s, c]])
                    
                    shifted_sender[:, :2] = shifted_sender[:, :2] @ R.T + np.array([dx, dy])
                    yaw_idx = 6 if shifted_sender.shape[1] > 6 else 4
                    shifted_sender[:, yaw_idx] += dyaw
                    
                    s_corners = self._get_canonical_corners_batch(shifted_sender)

                    # Compute pairwise corner distances (N, M)
                    corner_dists = np.mean(
                        np.linalg.norm(s_corners[:, None, :, :] - e_corners[None, :, :, :], axis=-1),
                        axis=-1
                    )

                    # Hungarian matching to prevent greedy collapse
                    row_ind, col_ind = linear_sum_assignment(corner_dists)
                    matched_costs = corner_dists[row_ind, col_ind]
                    
                    # Cost is the sum of overlapping box distances (pseudo-IoU)
                    cost = np.sum(matched_costs[matched_costs < 3.0]) 

                    if cost < best_cost and len(matched_costs[matched_costs < 3.0]) > 0:
                        best_cost = cost
                        best_delta = (dx, dy, dyaw)

        return best_delta, best_cost

    def align(self, ego_boxes, sender_boxes_ego_frame):
        if len(ego_boxes) == 0 or len(sender_boxes_ego_frame) == 0:
            return np.eye(4), (0.0, 0.0, 0.0)

        # Run fast vectorized grid search
        opt_dx, opt_dy, opt_dtheta = self._evaluate_grid_vectorized(ego_boxes, sender_boxes_ego_frame)[0]

        # Deadband Filter to prevent micro-jitter on clean data
        if abs(opt_dx) < 0.1 and abs(opt_dy) < 0.1 and abs(opt_dtheta) < np.radians(0.5):
            return np.eye(4), (0.0, 0.0, 0.0)

        # Build SE(3) Matrix
        c, s = np.cos(opt_dtheta), np.sin(opt_dtheta)
        T_corr = np.eye(4)
        T_corr[0, 0], T_corr[0, 1] = c, -s
        T_corr[1, 0], T_corr[1, 1] = s, c
        T_corr[0, 3] = opt_dx
        T_corr[1, 3] = opt_dy

        return T_corr, (opt_dx, opt_dy, opt_dtheta)