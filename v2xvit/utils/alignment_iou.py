import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment

class PureGridIoUAligner:
    """
    Complete rewrite of Approach 2: Pure Global Polygon IoU Grid Search.
    Abandons continuous optimizers to prevent volatile spatial jumps.
    Directly maximizes exact geometric Intersection-over-Union (IoU) using Shapely.
    """
    def __init__(self, max_trans=1.2, max_yaw=np.radians(5.0), min_iou_gate=0.10):
        self.max_trans = max_trans
        self.max_yaw = max_yaw
        self.min_iou_gate = min_iou_gate # Strict gate to prevent false alignments

    @staticmethod
    def get_polygon(box):
        """Converts a bounding box array into an exact Shapely 2D Polygon."""
        x, y = box[0], box[1]
        l, w = 4.5, 2.0 # Force standard canonical vehicle dimensions
        yaw = box[6] if len(box) > 6 else box[4]
        
        c, s = np.cos(yaw), np.sin(yaw)
        corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
        R = np.array([[c, -s], [s, c]])
        
        # Rotate and translate
        transformed_corners = (corners @ R.T) + np.array([x, y])
        return Polygon(transformed_corners)

    def compute_total_iou(self, ego_polys, sender_boxes, dx, dy, dtheta):
        """Calculates exact Shapely IoU across transformed sender boxes."""
        shifted_sender = sender_boxes.copy()
        c, s = np.cos(dtheta), np.sin(dtheta)
        R = np.array([[c, -s], [s, c]])
        
        # Apply trial shift
        shifted_sender[:, :2] = shifted_sender[:, :2] @ R.T + np.array([dx, dy])
        yaw_idx = 6 if shifted_sender.shape[1] > 6 else 4
        shifted_sender[:, yaw_idx] += dtheta

        sender_polys = [self.get_polygon(b) for b in shifted_sender]

        # Build Exact IoU Matrix
        iou_matrix = np.zeros((len(sender_polys), len(ego_polys)))
        for i, s_poly in enumerate(sender_polys):
            for j, e_poly in enumerate(ego_polys):
                if s_poly.intersects(e_poly):
                    inter = s_poly.intersection(e_poly).area
                    union = s_poly.area + e_poly.area - inter
                    iou_matrix[i, j] = inter / union if union > 0 else 0.0

        # Maximize global IoU (maximize=True is critical here!)
        row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)
        
        # Only sum highly confident pairs
        valid_ious = iou_matrix[row_ind, col_ind]
        total_iou = valid_ious[valid_ious > 0.05].sum()
        
        return total_iou

    def align(self, ego_boxes, sender_boxes_ego_frame):
        """Brute-force grid search to find absolute maximum IoU overlap."""
        if len(ego_boxes) == 0 or len(sender_boxes_ego_frame) == 0:
            return np.eye(4), (0.0, 0.0, 0.0)
            
        ego_polys = [self.get_polygon(b) for b in ego_boxes]
        
        # 1. Define Strict Search Grid
        x_grid = np.linspace(-self.max_trans, self.max_trans, 9)
        y_grid = np.linspace(-self.max_trans, self.max_trans, 9)
        yaw_grid = np.linspace(-self.max_yaw, self.max_yaw, 5)

        best_iou = 0.0
        best_delta = (0.0, 0.0, 0.0)

        # 2. Evaluate Exact Geometric Overlap
        for dx in x_grid:
            for dy in y_grid:
                for dyaw in yaw_grid:
                    iou = self.compute_total_iou(ego_polys, sender_boxes_ego_frame, dx, dy, dyaw)
                    if iou > best_iou:
                        best_iou = iou
                        best_delta = (dx, dy, dyaw)
                        
        opt_dx, opt_dy, opt_dtheta = best_delta
        
        # 3. Strict Gate: Ignore if highly confident pairs are not found
        if best_iou < self.min_iou_gate:
            return np.eye(4), (0.0, 0.0, 0.0)
            
        # 4. Deadband Filter: Ignore microscopic shifts
        if abs(opt_dx) < 0.1 and abs(opt_dy) < 0.1 and abs(opt_dtheta) < np.radians(0.5):
            return np.eye(4), (0.0, 0.0, 0.0)

        # 5. Build SE(3) Matrix
        c, s = np.cos(opt_dtheta), np.sin(opt_dtheta)
        T_corr = np.eye(4)
        T_corr[0, 0], T_corr[0, 1] = c, -s
        T_corr[1, 0], T_corr[1, 1] = s, c
        T_corr[0, 3] = opt_dx
        T_corr[1, 3] = opt_dy

        return T_corr, (opt_dx, opt_dy, opt_dtheta)