import numpy as np
from scipy.optimize import linear_sum_assignment

class CornerICPAligner:
    """
    Approach 3: Bounding Box Corner ICP (Iterative Closest Point via SVD / Kabsch)
    Extracts 2D BEV corner keypoints from matching co-observed bounding boxes
    and solves for optimal relative rotation (dtheta) and translation (dx, dy)
    in a single closed-form SVD step.
    """
    def __init__(self, max_match_dist=5.0, min_boxes_required=2):
        self.max_match_dist = max_match_dist
        self.min_boxes_required = min_boxes_required

    @staticmethod
    def _get_2d_corners(box):
        """
        Converts box [x, y, z, dx, dy, dz, yaw] or [x, y, dx, dy, yaw]
        into 4 2D BEV corner vertices shape (4, 2).
        """
        x, y = box[0], box[1]
        length, width = box[3], box[4]
        yaw = box[6] if len(box) > 6 else box[4]

        l2, w2 = length / 2.0, width / 2.0
        
        # Canonical unrotated corner offsets
        base_corners = np.array([
            [-l2, -w2],
            [ l2, -w2],
            [ l2,  w2],
            [-l2,  w2]
        ], dtype=np.float64)

        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        
        # Rotate and translate to box centroid position
        return (base_corners @ R.T) + np.array([x, y], dtype=np.float64)

    def _match_box_pairs(self, ego_boxes, sender_boxes):
        """
        Uses Hungarian matching (linear sum assignment) on box centroids
        to pair sender boxes with ego boxes within maximum search distance.
        """
        num_e = len(ego_boxes)
        num_s = len(sender_boxes)
        
        # Distance matrix between centroids (num_s, num_e)
        s_centers = sender_boxes[:, :2]
        e_centers = ego_boxes[:, :2]
        dist_matrix = np.linalg.norm(s_centers[:, None, :] - e_centers[None, :, :], axis=-1)

        s_ind, e_ind = linear_sum_assignment(dist_matrix)

        valid_s_pts = []
        valid_e_pts = []

        for s_i, e_i in zip(s_ind, e_ind):
            if dist_matrix[s_i, e_i] <= self.max_match_dist:
                s_corners = self._get_2d_corners(sender_boxes[s_i]) # (4, 2)
                e_corners = self._get_2d_corners(ego_boxes[e_i])   # (4, 2)
                
                valid_s_pts.append(s_corners)
                valid_e_pts.append(e_corners)

        if len(valid_s_pts) < self.min_boxes_required:
            return None, None

        # Flatten list of (4, 2) corners into point cloud arrays of shape (4K, 2)
        P_s = np.vstack(valid_s_pts)
        P_e = np.vstack(valid_e_pts)
        
        return P_s, P_e

    def align(self, ego_boxes, sender_boxes):
        """
        Computes spatial correction delta T = (dx, dy, dtheta) using closed-form 2D SVD.
        Returns 4x4 Homogeneous SE(3) transformation matrix T_corr and estimated offsets.
        """
        if len(ego_boxes) < self.min_boxes_required or len(sender_boxes) < self.min_boxes_required:
            return np.eye(4, dtype=np.float64), (0.0, 0.0, 0.0)

        # 1. Match bounding box corners
        P_s, P_e = self._match_box_pairs(ego_boxes, sender_boxes)
        if P_s is None or P_e is None:
            return np.eye(4, dtype=np.float64), (0.0, 0.0, 0.0)

        # 2. Compute Centroids
        centroid_s = np.mean(P_s, axis=0) # (2,)
        centroid_e = np.mean(P_e, axis=0) # (2,)

        # Center point sets
        Q_s = P_s - centroid_s
        Q_e = P_e - centroid_e

        # 3. Compute Cross-Covariance Matrix H (2x2)
        H = Q_s.T @ Q_e

        # 4. Singular Value Decomposition
        U, S, Vt = np.linalg.svd(H)
        R_2d = Vt.T @ U.T

        # Reflection check (ensures orthonormal matrix with det(R) = +1)
        if np.linalg.det(R_2d) < 0:
            Vt[1, :] *= -1
            R_2d = Vt.T @ U.T

        # Extract yaw angle and translation vector
        opt_dtheta = np.arctan2(R_2d[1, 0], R_2d[0, 0])
        dt = centroid_e - (centroid_s @ R_2d.T)
        opt_dx, opt_dy = dt[0], dt[1]

        # 5. Deadband Thresholding (Skip microscopic adjustments)
        if abs(opt_dx) < 0.05 and abs(opt_dy) < 0.05 and abs(opt_dtheta) < np.radians(0.3):
            return np.eye(4, dtype=np.float64), (0.0, 0.0, 0.0)

        # 6. Construct 4x4 Homogeneous SE(3) Matrix
        T_corr = np.eye(4, dtype=np.float64)
        T_corr[0, 0] = R_2d[0, 0]
        T_corr[0, 1] = R_2d[0, 1]
        T_corr[1, 0] = R_2d[1, 0]
        T_corr[1, 1] = R_2d[1, 1]
        T_corr[0, 3] = opt_dx
        T_corr[1, 3] = opt_dy

        return T_corr, (opt_dx, opt_dy, opt_dtheta)