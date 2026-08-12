import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from shapely.geometry import Polygon

class AdvancedDIoUAligner:
    """
    Upgraded Approach 2: Coarse-to-Fine Distance-IoU (DIoU) Aligner
    Combines Hungarian Bipartite Matching, DIoU continuous loss, and 
    Coarse-to-Fine Grid Seeding to outperform Centroid SVD (Approach 1).
    """
    def __init__(self, max_trans_bound=2.5, max_yaw_bound=np.radians(10.0)):
        self.max_trans_bound = max_trans_bound  # Covers 1.0m noise + margin
        self.max_yaw_bound = max_yaw_bound      # Covers 8.0 deg noise + margin

    @staticmethod
    def _get_2d_corners(box):
        """Converts box [x, y, z, dx, dy, dz, yaw] into 2D corner vertices."""
        x, y = box[0], box[1]
        length, width = box[3], box[4]
        yaw = box[6] if len(box) > 6 else box[4]

        l2, w2 = length / 2.0, width / 2.0
        corners = np.array([
            [-l2, -w2],
            [ l2, -w2],
            [ l2,  w2],
            [-l2,  w2]
        ])

        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        return corners @ R.T + np.array([x, y])

    def _compute_diou(self, s_box, e_box):
        """
        Calculates Distance-IoU (DIoU) between two 2D boxes.
        DIoU = IoU - (rho^2 / c^2), where rho is centroid distance and c is diagonal of enclosing box.
        """
        p1 = Polygon(self._get_2d_corners(s_box))
        p2 = Polygon(self._get_2d_corners(e_box))

        # Centroid distance squared (rho^2)
        rho_sq = (s_box[0] - e_box[0])**2 + (s_box[1] - e_box[1])**2

        # Diagonal distance squared of enclosing convex hull (c^2)
        min_x = min(np.min(p1.exterior.coords.xy[0]), np.min(p2.exterior.coords.xy[0]))
        max_x = max(np.max(p1.exterior.coords.xy[0]), np.max(p2.exterior.coords.xy[0]))
        min_y = min(np.min(p1.exterior.coords.xy[1]), np.min(p2.exterior.coords.xy[1]))
        max_y = max(np.max(p1.exterior.coords.xy[1]), np.max(p2.exterior.coords.xy[1]))
        c_sq = (max_x - min_x)**2 + (max_y - min_y)**2 + 1e-6

        if not p1.intersects(p2):
            return -(rho_sq / c_sq)

        inter = p1.intersection(p2).area
        union = p1.area + p2.area - inter
        iou = inter / union if union > 0 else 0.0

        return iou - (rho_sq / c_sq)

    def _transform_boxes(self, boxes, dx, dy, dtheta):
        """Applies spatial offset (dx, dy, dtheta) to sender bounding boxes."""
        transformed = boxes.copy()
        c, s = np.cos(dtheta), np.sin(dtheta)
        R = np.array([[c, -s], [s, c]])

        transformed[:, :2] = transformed[:, :2] @ R.T + np.array([dx, dy])
        yaw_idx = 6 if transformed.shape[1] > 6 else 4
        transformed[:, yaw_idx] += dtheta
        return transformed

    def _objective_function(self, delta, ego_boxes, sender_boxes):
        """
        Objective function using Hungarian Algorithm (Bipartite Matching) on DIoU Cost Matrix.
        """
        dx, dy, dtheta = delta

        # Hard Boundary Barrier
        if abs(dx) > self.max_trans_bound or abs(dy) > self.max_trans_bound or abs(dtheta) > self.max_yaw_bound:
            return 1000.0

        shifted_sender = self._transform_boxes(sender_boxes, dx, dy, dtheta)
        num_s, num_e = len(shifted_sender), len(ego_boxes)

        # Build pairwise DIoU cost matrix
        cost_matrix = np.zeros((num_s, num_e))
        for i, s_box in enumerate(shifted_sender):
            for j, e_box in enumerate(ego_boxes):
                # Negative DIoU for minimization
                cost_matrix[i, j] = -self._compute_diou(s_box, e_box)

        # Hungarian Bipartite Matching for optimal 1-to-1 box assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_cost = cost_matrix[row_ind, col_ind].sum()

        return total_cost

    def _coarse_grid_search(self, ego_boxes, sender_boxes):
        """
        Coarse-to-fine 3x3x3 grid warm-start to locate global basin of attraction.
        """
        x_grid = np.linspace(-1.5, 1.5, 3)
        y_grid = np.linspace(-1.5, 1.5, 3)
        theta_grid = np.linspace(-np.radians(8.0), np.radians(8.0), 3)

        best_seed = (0.0, 0.0, 0.0)
        best_cost = float('inf')

        for dx in x_grid:
            for dy in y_grid:
                for dtheta in theta_grid:
                    cost = self._objective_function((dx, dy, dtheta), ego_boxes, sender_boxes)
                    if cost < best_cost:
                        best_cost = cost
                        best_seed = (dx, dy, dtheta)

        return best_seed

    def align(self, ego_boxes, sender_boxes_ego_frame):
        """
        Solves relative pose using Coarse Seeding + Bounded Hungarian DIoU Optimization.
        """
        if len(ego_boxes) == 0 or len(sender_boxes_ego_frame) == 0:
            return np.eye(4), (0.0, 0.0, 0.0)

        # 1. Coarse-to-fine Warm Start (Find global basin)
        best_seed = self._coarse_grid_search(ego_boxes, sender_boxes_ego_frame)

        # 2. Local Fine-Tuning via Bounded Simplex
        step_x, step_y, step_theta = 0.3, 0.3, np.radians(2.0)
        x0 = np.array(best_seed, dtype=np.float64)
        custom_simplex = np.array([
            x0,
            x0 + [step_x, 0.0, 0.0],
            x0 + [0.0, step_y, 0.0],
            x0 + [0.0, 0.0, step_theta]
        ])

        res = minimize(
            self._objective_function,
            x0=x0,
            args=(ego_boxes, sender_boxes_ego_frame),
            method='Nelder-Mead',
            options={
                'initial_simplex': custom_simplex,
                'maxiter': 40,
                'xatol': 1e-2,
                'fatol': 1e-2
            }
        )

        opt_dx, opt_dy, opt_dtheta = res.x

        # Deadband thresholding for clean baseline stability
        if abs(opt_dx) < 0.05 and abs(opt_dy) < 0.05 and abs(opt_dtheta) < np.radians(0.3):
            return np.eye(4), (0.0, 0.0, 0.0)

        # Construct SE(3) Homogeneous Transformation Matrix
        c, s = np.cos(opt_dtheta), np.sin(opt_dtheta)
        T_corr = np.eye(4)
        T_corr[0, 0], T_corr[0, 1] = c, -s
        T_corr[1, 0], T_corr[1, 1] = s, c
        T_corr[0, 3] = opt_dx
        T_corr[1, 3] = opt_dy

        return T_corr, (opt_dx, opt_dy, opt_dtheta)