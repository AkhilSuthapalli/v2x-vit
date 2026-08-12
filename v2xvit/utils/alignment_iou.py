import numpy as np
from scipy.optimize import minimize
from shapely.geometry import Polygon

class BoundedNelderMeadAligner:
    """
    Approach 2: Bounded Bounding Box IoU Optimization Loop
    Prevents box aliasing/jumping by enforcing strict search bounds,
    pre-filtering candidate pairs, and applying deadband thresholding.
    """
    def __init__(self, max_trans_bound=2.0, max_yaw_bound=np.radians(5.0), max_pair_dist=3.5):
        self.max_trans_bound = max_trans_bound  # Max +/- 2.0m translation
        self.max_yaw_bound = max_yaw_bound      # Max +/- 5.0 deg heading shift
        self.max_pair_dist = max_pair_dist      # Candidate pairing radius

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

    def _calculate_pair_cost(self, s_box, e_box):
        """Calculates match cost between candidate boxes."""
        dist = np.hypot(s_box[0] - e_box[0], s_box[1] - e_box[1])
        if dist > self.max_pair_dist:
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

    def _objective_function(self, delta, ego_boxes, candidate_sender_boxes):
        """Objective function with penalty barrier for out-of-bound search steps."""
        dx, dy, dtheta = delta

        # ENFORCE STRICT SEARCH BOUNDS
        if abs(dx) > self.max_trans_bound or abs(dy) > self.max_trans_bound or abs(dtheta) > self.max_yaw_bound:
            return 1000.0  # High penalty barrier

        shifted_sender = self._transform_boxes(candidate_sender_boxes, dx, dy, dtheta)

        total_cost = 0.0
        for s_box in shifted_sender:
            best_cost = 0.0
            for e_box in ego_boxes:
                cost = self._calculate_pair_cost(s_box, e_box)
                if cost < best_cost:
                    best_cost = cost
            total_cost += best_cost

        return total_cost

    def align(self, ego_boxes, sender_boxes_ego_frame, initial_guess=(0.0, 0.0, 0.0)):
        """
        Runs bounded Nelder-Mead alignment on candidate co-observed pairs.
        """
        if len(ego_boxes) == 0 or len(sender_boxes_ego_frame) == 0:
            return np.eye(4), (0.0, 0.0, 0.0)

        # PRE-FILTER: Keep only sender boxes that are close to at least one ego box
        candidate_sender_boxes = []
        for s_box in sender_boxes_ego_frame:
            dists = np.hypot(ego_boxes[:, 0] - s_box[0], ego_boxes[:, 1] - s_box[1])
            if np.min(dists) <= self.max_pair_dist:
                candidate_sender_boxes.append(s_box)

        if len(candidate_sender_boxes) == 0:
            return np.eye(4), (0.0, 0.0, 0.0)

        candidate_sender_boxes = np.array(candidate_sender_boxes)
        x0 = np.array(initial_guess, dtype=np.float64)

        # Restricted initial simplex steps (+0.3m X, +0.3m Y, +1.5 deg Yaw)
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
            args=(ego_boxes, candidate_sender_boxes),
            method='Nelder-Mead',
            options={
                'initial_simplex': custom_simplex,
                'maxiter': 40,
                'xatol': 1e-2,
                'fatol': 1e-2
            }
        )

        opt_dx, opt_dy, opt_dtheta = res.x

        # DEADBAND FILTER: Ignore micro-corrections
        if abs(opt_dx) < 0.10 and abs(opt_dy) < 0.10 and abs(opt_dtheta) < np.radians(0.5):
            return np.eye(4), (0.0, 0.0, 0.0)

        # Construct SE(3) Matrix
        c, s = np.cos(opt_dtheta), np.sin(opt_dtheta)
        T_corr = np.eye(4)
        T_corr[0, 0], T_corr[0, 1] = c, -s
        T_corr[1, 0], T_corr[1, 1] = s, c
        T_corr[0, 3] = opt_dx
        T_corr[1, 3] = opt_dy

        return T_corr, (opt_dx, opt_dy, opt_dtheta)