import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon

class OptimizedNelderMeadAligner:
    """
    Approach 2: Bounding Box IoU Optimization Loop
    Uses constrained Nelder-Mead simplex search with distance-penalized continuous cost.
    """
    def __init__(self, max_search_dist=5.0):
        self.max_search_dist = max_search_dist

    def _get_2d_corners(self, box):
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

    def _calculate_pair_cost(self, s_box, e_box):
        """
        Calculates match score between two boxes.
        Returns continuous cost: -IoU if overlapping, or normalized distance penalty if not.
        """
        dist = np.hypot(s_box[0] - e_box[0], s_box[1] - e_box[1])
        if dist > self.max_search_dist:
            return 0.0

        p1 = Polygon(self._get_2d_corners(s_box))
        p2 = Polygon(self._get_2d_corners(e_box))

        if not p1.intersects(p2):
            return -1.0 / (1.0 + dist)

        inter = p1.intersection(p2).area
        union = p1.area + p2.area - inter
        iou = inter / union if union > 0 else 0.0
        return -2.0 - iou

    def _transform_boxes(self, boxes, dx, dy, dtheta):
        """Applies (dx, dy, dtheta) offset to sender bounding boxes."""
        transformed = boxes.copy()
        c, s = np.cos(dtheta), np.sin(dtheta)
        R = np.array([[c, -s], [s, c]])

        transformed[:, :2] = transformed[:, :2] @ R.T + np.array([dx, dy])
        yaw_idx = 6 if transformed.shape[1] > 6 else 4
        transformed[:, yaw_idx] += dtheta
        return transformed

    def _objective_function(self, delta, ego_boxes, sender_boxes):
        """Objective function to MINIMIZE."""
        dx, dy, dtheta = delta
        shifted_sender = self._transform_boxes(sender_boxes, dx, dy, dtheta)

        total_cost = 0.0
        for s_box in shifted_sender:
            best_cost = 0.0
            for e_box in ego_boxes:
                cost = self._calculate_pair_cost(s_box, e_box)
                if cost < best_cost:
                    best_cost = cost
            total_cost += best_cost

        return total_cost

    def align(self, ego_boxes, sender_boxes, initial_guess=(0.0, 0.0, 0.0)):
        """
        Solves for spatial delta T = (dx, dy, dtheta) using Nelder-Mead optimization.
        Returns corrected transformation matrix T_corr and estimated delta.
        """
        if len(ego_boxes) == 0 or len(sender_boxes) == 0:
            return np.eye(4), (0.0, 0.0, 0.0)

        x0 = np.array(initial_guess, dtype=np.float64)

        step_x, step_y, step_theta = 0.5, 0.5, np.radians(3.0)
        custom_simplex = np.array([
            x0,
            x0 + [step_x, 0.0, 0.0],
            x0 + [0.0, step_y, 0.0],
            x0 + [0.0, 0.0, step_theta]
        ])

        res = minimize(
            self._objective_function,
            x0=x0,
            args=(ego_boxes, sender_boxes),
            method='Nelder-Mead',
            options={
                'initial_simplex': custom_simplex,
                'maxiter': 50,
                'xatol': 1e-2,
                'fatol': 1e-2
            }
        )

        opt_dx, opt_dy, opt_dtheta = res.x

        c, s = np.cos(opt_dtheta), np.sin(opt_dtheta)
        T_corr = np.eye(4)
        T_corr[0, 0], T_corr[0, 1] = c, -s
        T_corr[1, 0], T_corr[1, 1] = s, c
        T_corr[0, 3] = opt_dx
        T_corr[1, 3] = opt_dy

        return T_corr, (opt_dx, opt_dy, opt_dtheta)