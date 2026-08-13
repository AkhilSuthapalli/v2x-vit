import numpy as np
from scipy.optimize import linear_sum_assignment

class BoxCornerSVDAligner:
    """
    Approach 4 (Final): Canonical RANSAC ICP
    Uses true dimensions for matching, but Canonical Unit Corners for SVD 
    to completely decouple spatial alignment from PointPillar dimension jitter.
    """
    def __init__(self, max_match_dist=4.0, min_boxes_required=2, ransac_iters=30, debug=True):
        self.max_match_dist = max_match_dist
        self.min_boxes_required = min_boxes_required
        self.ransac_iters = ransac_iters
        self.debug = debug

    @staticmethod
    def _get_canonical_corners(box):
        """Converts box to 4 corners using a FIXED unit shape to prevent dimension jitter."""
        x, y = box[0], box[1]
        yaw = box[6] if len(box) > 6 else box[4]
        
        # FIXED CANONICAL SHAPE: Strips away PointPillar length/width estimation noise
        l2, w2 = 1.0, 0.5 
        
        base_corners = np.array([
            [-l2, -w2],
            [ l2, -w2],
            [ l2,  w2],
            [-l2,  w2]
        ], dtype=np.float64)
        
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        return (base_corners @ R.T) + np.array([x, y], dtype=np.float64)

    def _solve_kabsch(self, P_s, P_e):
        """Standard Kabsch SVD Algorithm."""
        centroid_s = np.mean(P_s, axis=0)
        centroid_e = np.mean(P_e, axis=0)
        H = (P_s - centroid_s).T @ (P_e - centroid_e)
        U, _, Vt = np.linalg.svd(H)
        R_2d = Vt.T @ U.T
        
        # Reflection correction
        if np.linalg.det(R_2d) < 0:
            Vt[1, :] *= -1
            R_2d = Vt.T @ U.T
            
        dtheta = np.arctan2(R_2d[1, 0], R_2d[0, 0])
        t = centroid_e - (centroid_s @ R_2d.T)
        return t[0], t[1], dtheta, R_2d

    def align(self, ego_boxes, sender_boxes, cav_id="Unknown"):
        if len(ego_boxes) < self.min_boxes_required or len(sender_boxes) < self.min_boxes_required:
            return np.eye(4, dtype=np.float64), (0.0, 0.0, 0.0)

        # 1. Size-Aware Hungarian Matching (Uses TRUE dimensions to prevent bad pairings)
        s_centers = sender_boxes[:, :2]
        e_centers = ego_boxes[:, :2]
        s_dims = sender_boxes[:, 3:5]
        e_dims = ego_boxes[:, 3:5]

        dist_matrix = np.linalg.norm(s_centers[:, None, :] - e_centers[None, :, :], axis=-1)
        size_penalty = np.sum(np.abs(s_dims[:, None, :] - e_dims[None, :, :]), axis=-1)
        
        cost_matrix = dist_matrix + (size_penalty * 2.0)
        s_ind, e_ind = linear_sum_assignment(cost_matrix)

        matched_s_corners = []
        matched_e_corners = []

        for s_i, e_i in zip(s_ind, e_ind):
            if dist_matrix[s_i, e_i] <= self.max_match_dist:
                # 2. Extract CANONICAL corners for pure geometric SVD math
                e_corners = self._get_canonical_corners(ego_boxes[e_i])
                s_corners = self._get_canonical_corners(sender_boxes[s_i])

                # MANIFOLD-PRESERVING 180-DEGREE FLIP FIX
                dist_normal = np.sum(np.linalg.norm(e_corners - s_corners, axis=-1))
                s_corners_flipped = np.roll(s_corners, 2, axis=0)
                dist_flipped = np.sum(np.linalg.norm(e_corners - s_corners_flipped, axis=-1))
                
                if dist_flipped < dist_normal:
                    s_corners = s_corners_flipped
                    
                matched_s_corners.append(s_corners)
                matched_e_corners.append(e_corners)

        if len(matched_s_corners) < self.min_boxes_required:
            return np.eye(4, dtype=np.float64), (0.0, 0.0, 0.0)

        matched_s_corners = np.array(matched_s_corners) # Shape: (N, 4, 2)
        matched_e_corners = np.array(matched_e_corners) # Shape: (N, 4, 2)
        N = len(matched_s_corners)

        # 3. RANSAC SVD Loop
        best_inliers = []
        best_offsets = (0.0, 0.0, 0.0)
        best_R = np.eye(2)

        s_centroids = np.mean(matched_s_corners, axis=1)
        e_centroids = np.mean(matched_e_corners, axis=1)

        for _ in range(self.ransac_iters):
            idx = np.random.choice(N, min(2, N), replace=False)
            P_s_sample = matched_s_corners[idx].reshape(-1, 2)
            P_e_sample = matched_e_corners[idx].reshape(-1, 2)
            
            dx, dy, dtheta, R_2d = self._solve_kabsch(P_s_sample, P_e_sample)
            
            s_transformed = s_centroids @ R_2d.T + np.array([dx, dy])
            errors = np.linalg.norm(s_transformed - e_centroids, axis=-1)
            
            inliers = np.where(errors < 0.4)[0] # Tightened inlier threshold to 0.4m
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_offsets = (dx, dy, dtheta)
                best_R = R_2d
                if len(inliers) == N:
                    break

        # 4. Final Polish using ALL verified inliers
        if len(best_inliers) >= self.min_boxes_required:
            P_s_inliers = matched_s_corners[best_inliers].reshape(-1, 2)
            P_e_inliers = matched_e_corners[best_inliers].reshape(-1, 2)
            opt_dx, opt_dy, opt_dtheta, final_R = self._solve_kabsch(P_s_inliers, P_e_inliers)
        else:
            opt_dx, opt_dy, opt_dtheta = best_offsets
            final_R = best_R

        # Deadband Filter
        if abs(opt_dx) < 0.05 and abs(opt_dy) < 0.05 and abs(np.degrees(opt_dtheta)) < 0.5:
            return np.eye(4, dtype=np.float64), (0.0, 0.0, 0.0)

        if self.debug:
            print(f"\n[DEBUG RANSAC SVD] CAV: {cav_id} | Inliers: {len(best_inliers)}/{N}")
            print(f"[DEBUG RANSAC SVD] Solved Offsets -> dx: {opt_dx:.3f}m, dy: {opt_dy:.3f}m, yaw: {np.degrees(opt_dtheta):.2f}°")

        T_corr = np.eye(4, dtype=np.float64)
        T_corr[:2, :2] = final_R
        T_corr[0, 3] = opt_dx
        T_corr[1, 3] = opt_dy

        return T_corr, (opt_dx, opt_dy, opt_dtheta)