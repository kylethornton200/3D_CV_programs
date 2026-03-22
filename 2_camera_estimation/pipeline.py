"""
=============================================================================
Multi-camera implementations based on principles of multiple view geometry.
Most ideas are covered in MVG.

Covers:
  1. Normalized 8-point algorithm
  2. 7-point algorithm (minimal solver)
  3. RANSAC robust estimation
  4. Sampson distance
  5. Rank-2 enforcement via SVD
  6. Minimizing Sampson distance via Levenberg-Marquardt
  7. Essential matrix extraction & pose recovery
  8. Triangulation (linear + optimal)
  9. Full 3D reconstruction
  10. Visualization of epipolar lines, matches, and 3D point cloud
=============================================================================
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from typing import Tuple, List


# =============================================================================
# Synthetic Cameras/Data
# =============================================================================

def generate_synthetic_scene(
    n_points: int = 100,
    scene_spread: float = 2.0,
    seed: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic 3D scene with two cameras.
    
    Returns:
        X_world   : (n_points, 3) 3D points
        K1, K2    : (3, 3) intrinsic matrices
        R, t      : relative rotation and translation (camera 2 w.r.t. camera 1)
    """
    rng = np.random.RandomState(seed)
    
    n_random = n_points // 2
    X_random = rng.randn(n_random, 3) * scene_spread
    X_random[:, 2] += 6.0  # Push points in front of cameras
    
    n_cube = n_points - n_random
    faces = []
    for _ in range(n_cube):
        face = rng.randint(6)
        pt = rng.rand(3) * 2 - 1  # [-1, 1]
        if face == 0: pt[0] = -1
        elif face == 1: pt[0] = 1
        elif face == 2: pt[1] = -1
        elif face == 3: pt[1] = 1
        elif face == 4: pt[2] = -1
        else: pt[2] = 1
        faces.append(pt * scene_spread)
    X_cube = np.array(faces)
    X_cube[:, 2] += 6.0
    
    X_world = np.vstack([X_random, X_cube])
    
    K1 = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Camera 2 intrinsics
    K2 = np.array([
        [830, 0, 330],
        [0, 815, 250],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Relative pose
    angle = np.radians(30)
    R = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ], dtype=np.float64)
    
    t = np.array([1.0, 0.1, 0.2], dtype=np.float64)
    
    return X_world, K1, K2, R, t


def project_points(
    X: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """
    Project 3D points using camera P = K [R | t].
    
    Args:
        X : (N, 3) 3D points
        K : (3, 3) intrinsic matrix
        R : (3, 3) rotation matrix
        t : (3,) translation vector
    
    Returns:
        x : (N, 2) projected 2D points
    """
    P = K @ np.hstack([R, t.reshape(3, 1)])
    X_hom = np.hstack([X, np.ones((X.shape[0], 1))])
    x_hom = (P @ X_hom.T).T
    x = x_hom[:, :2] / x_hom[:, 2:3]
    return x


def add_noise_and_outliers(
    x1: np.ndarray, x2: np.ndarray,
    noise_std: float = 1.0,
    outlier_ratio: float = 0.15,
    img_size: Tuple[int, int] = (640, 480),
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Add Gaussian noise and random outliers to correspondences to simulate the way we would theoretically get these pairs.
    
    Returns:
        x1_noisy, x2_noisy : noisy points with outliers appended
        is_inlier          : boolean mask (True for inliers)
    """
    rng = np.random.RandomState(seed)
    n = x1.shape[0]
    n_outliers = int(n * outlier_ratio)
    
    x1_noisy = x1 + rng.randn(n, 2) * noise_std
    x2_noisy = x2 + rng.randn(n, 2) * noise_std
    
    # Generate outliers
    x1_out = rng.rand(n_outliers, 2) * np.array([img_size[0], img_size[1]])
    x2_out = rng.rand(n_outliers, 2) * np.array([img_size[0], img_size[1]])
    
    x1_all = np.vstack([x1_noisy, x1_out])
    x2_all = np.vstack([x2_noisy, x2_out])
    is_inlier = np.zeros(n + n_outliers, dtype=bool)
    is_inlier[:n] = True
    
    perm = rng.permutation(n + n_outliers)
    return x1_all[perm], x2_all[perm], is_inlier[perm]


# =============================================================================
# 1: Normalization (Hartley's Method)
# =============================================================================

def normalize_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hartley normalization: translate centroid to origin, scale so mean
    distance from origin is sqrt(2).

    Args:
        pts : (N, 2) points
    
    Returns:
        pts_norm : (N, 2) normalized points
        T        : (3, 3) normalization transform such that
                   pts_norm_hom = T @ pts_hom
    """
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
    
    if mean_dist < 1e-10:
        mean_dist = 1e-10
    
    scale = np.sqrt(2) / mean_dist
    
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    pts_hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_norm_hom = (T @ pts_hom.T).T
    pts_norm = pts_norm_hom[:, :2]
    
    return pts_norm, T


# =============================================================================
# 2: The 8-Point Algorithm
# =============================================================================

def eight_point_algorithm(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Normalized 8-point algorithm for estimating F.
    
    The epipolar constraint x2^T F x1 = 0 gives a linear system in
    the entries of F. With >= 8 correspondences, solve via SVD.
    
    Hartley normalization is applied for numerical stability
    otherwise it can get lost in floating point operations.
    
    Args:
        x1, x2 : (N, 2) corresponding points, N >= 8
    
    Returns:
        F : (3, 3) fundamental matrix (rank-2 enforced)
    """
    assert x1.shape[0] >= 8, "Need at least 8 correspondences"
    
    x1_norm, T1 = normalize_points(x1)
    x2_norm, T2 = normalize_points(x2)
    
    N = x1_norm.shape[0]
    u1, v1 = x1_norm[:, 0], x1_norm[:, 1]
    u2, v2 = x2_norm[:, 0], x2_norm[:, 1]
    
    A = np.column_stack([
        u2 * u1, u2 * v1, u2,
        v2 * u1, v2 * v1, v2,
        u1, v1, np.ones(N)
    ])

    _, S, Vt = np.linalg.svd(A)
    f = Vt[-1]  # Last row of Vt = smallest singular value
    F = f.reshape(3, 3)
    
    F = enforce_rank2(F)
    
    F = T2.T @ F @ T1
    
    F = F / np.linalg.norm(F)
    
    return F


def enforce_rank2(F: np.ndarray) -> np.ndarray:
    """
    Enforce the rank-2 constraint on F by zeroing the smallest
    singular value.
    
    F = U diag(s1, s2, s3) V^T  -->  F' = U diag(s1, s2, 0) V^T
    """
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0  # Zero the smallest singular value to collapse the axis
    F_rank2 = U @ np.diag(S) @ Vt
    return F_rank2


# =============================================================================
# 3: The 7-Point Algorithm
# =============================================================================

def seven_point_algorithm(x1: np.ndarray, x2: np.ndarray) -> List[np.ndarray]:
    """
    7-point algorithm — minimal solver for F.
    
    With 7 correspondences, the null space of A is 2D:
        F = alpha * F1 + (1 - alpha) * F2
    
    Imposing det(F) = 0 gives a cubic in alpha with 1 or 3 real solutions.
    
    Args:
        x1, x2 : (7, 2) corresponding points
    
    Returns:
        List of valid fundamental matrices (1 or 3)
    """
    assert x1.shape[0] == 7, "Need exactly 7 correspondences"
    
    x1_norm, T1 = normalize_points(x1)
    x2_norm, T2 = normalize_points(x2)
    
    u1, v1 = x1_norm[:, 0], x1_norm[:, 1]
    u2, v2 = x2_norm[:, 0], x2_norm[:, 1]
    
    A = np.column_stack([
        u2 * u1, u2 * v1, u2,
        v2 * u1, v2 * v1, v2,
        u1, v1, np.ones(7)
    ])
    
    # Null space is 2D
    _, S, Vt = np.linalg.svd(A)
    F1 = Vt[-1].reshape(3, 3)
    F2 = Vt[-2].reshape(3, 3)
    
    # det(alpha * F1 + (1-alpha) * F2) = 0
    # This is a cubic polynomial in alpha
    # We evaluate det at several points and fit the cubic
    
    # Use the fact that det(aF1 + bF2) is a homogeneous cubic in (a,b)
    # Let F(a) = a*F1 + (1-a)*F2
    # Evaluate at a = 0, 1, -1, 2 to get 4 values of the cubic
    
    def det_at(alpha):
        return np.linalg.det(alpha * F1 + (1 - alpha) * F2)
    

    alphas_sample = np.array([0.0, 1.0, -1.0, 2])
    dets = np.array([det_at(a) for a in alphas_sample])
    
    V = np.column_stack([
        np.ones(4),
        alphas_sample,
        alphas_sample**2,
        alphas_sample**3
    ])
    
    coeffs = np.linalg.solve(V, dets)  # [c0, c1, c2, c3]
    
    # Find roots of the cubic
    poly_coeffs = coeffs[::-1]
    
    if abs(poly_coeffs[0]) < 1e-12:
        # Degenerate: quadratic or lower
        if abs(poly_coeffs[1]) < 1e-12:
            roots = []
        else:
            roots = np.roots(poly_coeffs[1:])
    else:
        roots = np.roots(poly_coeffs)
    
    # Keep only real roots
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-6]
    
    # Build F for each real root
    results = []
    for alpha in real_roots:
        F = alpha * F1 + (1 - alpha) * F2
        F = enforce_rank2(F)
        # Denormalize
        F = T2.T @ F @ T1
        F = F / np.linalg.norm(F)
        results.append(F)
    
    return results


# =============================================================================
# 4: Error Metrics
# =============================================================================

def algebraic_error(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Algebraic error: x2^T F x1 for each correspondence.
    Not geometrically meaningful but fast.
    """
    x1_hom = np.hstack([x1, np.ones((x1.shape[0], 1))])
    x2_hom = np.hstack([x2, np.ones((x2.shape[0], 1))])
    
    return np.array([x2_hom[i] @ F @ x1_hom[i] for i in range(x1.shape[0])])


def sampson_distance(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    Sampson distance — first-order approximation to the geometric
    (reprojection) error.
    
    d_sampson^2 = (x2^T F x1)^2 / 
                  ((Fx1)_1^2 + (Fx1)_2^2 + (F^Tx2)_1^2 + (F^Tx2)_2^2)
    
    This is the recommended error metric for RANSAC.
    """
    x1_hom = np.hstack([x1, np.ones((x1.shape[0], 1))])
    x2_hom = np.hstack([x2, np.ones((x2.shape[0], 1))])
    
    N = x1.shape[0]

    Fx1 = (F @ x1_hom.T).T       # (N, 3) — lines in image 2
    Ftx2 = (F.T @ x2_hom.T).T    # (N, 3) — lines in image 1
    
    numerator = np.array([x2_hom[i] @ F @ x1_hom[i] for i in range(N)])
    
    denom = (Fx1[:, 0]**2 + Fx1[:, 1]**2 +
             Ftx2[:, 0]**2 + Ftx2[:, 1]**2)
    
    denom = np.maximum(denom, 1e-12)

    return numerator**2 / denom


def symmetric_epipolar_distance(
    F: np.ndarray, x1: np.ndarray, x2: np.ndarray
) -> np.ndarray:
    """
    Symmetric epipolar distance: sum of squared distances from each
    point to the other's epipolar line.
    """
    x1_hom = np.hstack([x1, np.ones((x1.shape[0], 1))])
    x2_hom = np.hstack([x2, np.ones((x2.shape[0], 1))])
    
    N = x1.shape[0]
    
    Fx1 = (F @ x1_hom.T).T
    Ftx2 = (F.T @ x2_hom.T).T
    
    numerator = np.array([x2_hom[i] @ F @ x1_hom[i] for i in range(N)])
    
    d1 = numerator**2 / (Fx1[:, 0]**2 + Fx1[:, 1]**2 + 1e-12)
    d2 = numerator**2 / (Ftx2[:, 0]**2 + Ftx2[:, 1]**2 + 1e-12)
    
    return d1 + d2


# =============================================================================
# 5: RANSAC Robust Estimation
# =============================================================================

def ransac_fundamental(
    x1: np.ndarray, x2: np.ndarray,
    n_iterations: int = 2000,
    threshold: float = 3.0,
    use_7point: bool = True,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSAC estimation of the fundamental matrix.
    
    Uses either the 7-point (minimal) or 8-point algorithm as the
    hypothesis generator, with Sampson distance as the error metric.
    
    Args:
        x1, x2       : (N, 2) putative correspondences
        n_iterations  : max RANSAC iterations
        threshold     : inlier threshold (squared Sampson distance)
        use_7point    : if True, use 7-point; else use 8-point
        seed          : random seed
    
    Returns:
        F_best    : (3, 3) best fundamental matrix
        inliers   : boolean mask of inlier correspondences
    """
    rng = np.random.RandomState(seed)
    N = x1.shape[0]
    sample_size = 7 if use_7point else 8
    
    best_inlier_count = 0
    best_F = None
    best_inliers = np.zeros(N, dtype=bool)
    
    max_iter = n_iterations
    
    for i in range(max_iter):
        idx = rng.choice(N, sample_size, replace=False)
        
        try:
            if use_7point:
                F_candidates = seven_point_algorithm(x1[idx], x2[idx])
            else:
                F_candidates = [eight_point_algorithm(x1[idx], x2[idx])]
        except (np.linalg.LinAlgError, ValueError):
            continue
        
        for F_candidate in F_candidates:
            if F_candidate is None or np.any(np.isnan(F_candidate)):
                continue
            
            # Compute Sampson distance for all points
            errors = sampson_distance(F_candidate, x1, x2)
            inliers = errors < threshold
            n_inliers = np.sum(inliers)
            
            if n_inliers > best_inlier_count:
                best_inlier_count = n_inliers
                best_F = F_candidate.copy()
                best_inliers = inliers.copy()
                
                w = n_inliers / N
                if w > 0:
                    p = 0.999  # desired probability of success
                    denom = 1 - w**sample_size
                    if denom > 0:
                        adaptive_iter = int(np.log(1 - p) / np.log(denom))
                        max_iter = min(n_iterations, adaptive_iter)
        
        if i >= max_iter:
            break
    
    # Recompute F using all inliers with 8-point algorithm
    if best_F is not None and np.sum(best_inliers) >= 8:
        best_F = eight_point_algorithm(x1[best_inliers], x2[best_inliers])
    
    return best_F, best_inliers


# =============================================================================
# SECTION 7: Gold Standard (MLE) Refinement
# =============================================================================

def f_from_params(params: np.ndarray) -> np.ndarray:
    """
    Reconstruct F from 7 parameters.
    
    Parameterization: F = [e']_x @ H, where:
        - e' is parameterized by 2 numbers (e1, e2) with e' = (e1, e2, 1)
        - H is parameterized by the remaining 5 (but we use a different
          parameterization for simplicity: direct entries with rank-2 enforcement)
    
    Here we use a simpler approach: parameterize all 9 entries and
    enforce rank-2 in the residual computation.
    """
    # We use a 9-parameter over-parameterization and enforce rank-2
    F = params[:9].reshape(3, 3)
    F = enforce_rank2(F)
    return F


def gold_standard_refinement(
    F_init: np.ndarray, x1: np.ndarray, x2: np.ndarray,
    max_iter: int = 100
) -> np.ndarray:
    """
    Gold Standard (Maximum Likelihood) refinement of F by minimizing
    the Sampson distance via Levenberg-Marquardt.
    
    This is a good approximation to the full geometric error
    minimization and much simpler to implement.
    
    Args:
        F_init : (3, 3) initial estimate of F
        x1, x2 : (N, 2) inlier correspondences
    
    Returns:
        F_refined : (3, 3) refined fundamental matrix
    """
    x1_hom = np.hstack([x1, np.ones((x1.shape[0], 1))])
    x2_hom = np.hstack([x2, np.ones((x2.shape[0], 1))])
    
    def residuals(params):
        F = params.reshape(3, 3)
        F = enforce_rank2(F)
        
        Fx1 = (F @ x1_hom.T).T
        Ftx2 = (F.T @ x2_hom.T).T
        
        num = np.sum(x2_hom * (F @ x1_hom.T).T, axis=1)
        
        denom = np.sqrt(
            Fx1[:, 0]**2 + Fx1[:, 1]**2 +
            Ftx2[:, 0]**2 + Ftx2[:, 1]**2 + 1e-12
        )
        
        # Sampson error (signed, for least_squares)
        return num / denom
    
    # Initial parameters
    p0 = F_init.flatten()
    
    result = least_squares(
        residuals, p0,
        method='lm',
        max_nfev=max_iter * 9
    )
    
    F_refined = result.x.reshape(3, 3)
    F_refined = enforce_rank2(F_refined)
    F_refined = F_refined / np.linalg.norm(F_refined)
    
    return F_refined


# =============================================================================
# 7: Essential Matrix and Pose Recovery
# =============================================================================

def compute_essential_matrix(
    F: np.ndarray, K1: np.ndarray, K2: np.ndarray
) -> np.ndarray:
    """
    E = K2^T @ F @ K1
    
    The essential matrix encodes only the relative rotation and
    translation direction between calibrated cameras.
    """
    E = K2.T @ F @ K1
    
    # Enforce the constraint that the two nonzero singular values are equal
    U, S, Vt = np.linalg.svd(E)
    S_new = np.array([(S[0] + S[1]) / 2, (S[0] + S[1]) / 2, 0])
    E = U @ np.diag(S_new) @ Vt
    
    return E


def decompose_essential_matrix(
    E: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Decompose E into four possible (R, t) solutions.
    
    E = U diag(1, 1, 0) V^T
    
    The four solutions are:
        (R1, +t), (R1, -t), (R2, +t), (R2, -t)
    where R1 = U W V^T, R2 = U W^T V^T, t = U[:,2]
    
    Returns:
        List of (R, t) tuples
    """
    U, S, Vt = np.linalg.svd(E)
    
    # Ensure proper rotation (det = +1)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    
    solutions = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]
    
    return solutions


def recover_pose(
    E: np.ndarray, K1: np.ndarray, K2: np.ndarray,
    x1: np.ndarray, x2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recover the correct (R, t) from E by testing the cheirality
    constraint.
    
    Args:
        E      : (3, 3) essential matrix
        K1, K2 : intrinsic matrices
        x1, x2 : (N, 2) correspondences (a subset for testing)
    
    Returns:
        R, t : the correct rotation and translation
    """
    solutions = decompose_essential_matrix(E)
    
    best_count = 0
    best_R, best_t = None, None
    
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    
    for R, t in solutions:
        P2 = K2 @ np.hstack([R, t.reshape(3, 1)])
        

        n_test = min(x1.shape[0], 50)
        X_test = triangulate_points(P1, P2, x1[:n_test], x2[:n_test])
        
        if X_test is None:
            continue
        
        # Check cheirality: points must be in front of both cameras
        in_front_1 = X_test[:, 2] > 0
        
        X_cam2 = (R @ X_test.T).T + t
        in_front_2 = X_cam2[:, 2] > 0
        
        count = np.sum(in_front_1 & in_front_2)
        
        if count > best_count:
            best_count = count
            best_R = R
            best_t = t
    
    return best_R, best_t


# =============================================================================
# 8: Triangulation
# =============================================================================

def triangulate_points(
    P1: np.ndarray, P2: np.ndarray,
    x1: np.ndarray, x2: np.ndarray
) -> np.ndarray:
    """
    Linear triangulation (DLT method).
    
    For each correspondence (x1_i, x2_i), solve:
        x1 × (P1 X) = 0
        x2 × (P2 X) = 0
    
    This gives 4 equations in 4 unknowns (homogeneous X).
    Solve via SVD.
    
    Args:
        P1, P2 : (3, 4) camera matrices
        x1, x2 : (N, 2) correspondences
    
    Returns:
        X : (N, 3) triangulated 3D points
    """
    N = x1.shape[0]
    X = np.zeros((N, 3))
    
    for i in range(N):
        u1, v1 = x1[i, 0], x1[i, 1]
        u2, v2 = x2[i, 0], x2[i, 1]
        
        # Build the 4x4 system
        A = np.array([
            u1 * P1[2] - P1[0],
            v1 * P1[2] - P1[1],
            u2 * P2[2] - P2[0],
            v2 * P2[2] - P2[1]
        ])
        
        _, S, Vt = np.linalg.svd(A)
        X_hom = Vt[-1]
        
        if abs(X_hom[3]) < 1e-10:
            X[i] = X_hom[:3]  # Point at infinity
        else:
            X[i] = X_hom[:3] / X_hom[3]
    
    return X


def triangulate_optimal(
    P1: np.ndarray, P2: np.ndarray,
    x1_pt: np.ndarray, x2_pt: np.ndarray,
    F: np.ndarray
) -> np.ndarray:
    """
    Optimal triangulation (Hartley-Sturm method, Section 12.5).
    
    Finds the pair of corresponding points (x1_hat, x2_hat) that
    exactly satisfy the epipolar constraint and minimize the total
    reprojection error.
    
    For a single point pair. This is the Gold Standard for triangulation.
    
    Simplified version: correct the points to lie on epipolar lines,
    then triangulate.
    """
    x1_hom = np.array([x1_pt[0], x1_pt[1], 1.0])
    x2_hom = np.array([x2_pt[0], x2_pt[1], 1.0])
    
    # Compute epipolar lines
    l2 = F @ x1_hom      # line in image 2
    l1 = F.T @ x2_hom    # line in image 1
    
    # Project each point onto the other's epipolar line
    # Closest point on line l = (a,b,c) to point (u,v):
    # (u - a*(au+bv+c)/(a^2+b^2), v - b*(au+bv+c)/(a^2+b^2))
    
    def project_to_line(pt, l):
        a, b, c = l
        u, v = pt
        d = a*u + b*v + c
        norm_sq = a**2 + b**2
        return np.array([u - a*d/norm_sq, v - b*d/norm_sq])
    
    x1_corr = project_to_line(x1_pt, l1)
    x2_corr = project_to_line(x2_pt, l2)
    
    # Triangulate the corrected points
    return triangulate_points(P1, P2,
                             x1_corr.reshape(1, 2),
                             x2_corr.reshape(1, 2))[0]


# =============================================================================
# 9: Epipole Computation
# =============================================================================

def compute_epipoles(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the epipoles from F.
    
    e  = Fe = 0
    e' = F^T e' = 0
    """
    _, _, Vt = np.linalg.svd(F)
    e = Vt[-1]
    e = e / e[2] if abs(e[2]) > 1e-10 else e
    
    _, _, Vt2 = np.linalg.svd(F.T)
    e_prime = Vt2[-1]
    e_prime = e_prime / e_prime[2] if abs(e_prime[2]) > 1e-10 else e_prime
    
    return e, e_prime


# =============================================================================
# 10: Visualization
# =============================================================================

def draw_epipolar_lines(
    F: np.ndarray, x1: np.ndarray, x2: np.ndarray,
    img_size: Tuple[int, int] = (640, 480),
    n_lines: int = 15,
    filename: str = "epipolar_lines.png"
):
    """
    Draw epipolar lines in both images.
    
    For selected points in image 1, draw the corresponding epipolar
    lines in image 2 and vice versa.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    W, H = img_size
    n = min(n_lines, x1.shape[0])
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    
    indices = np.linspace(0, x1.shape[0]-1, n, dtype=int)
    
    for idx, i in enumerate(indices):
        color = colors[idx]
        
        x1_hom = np.array([x1[i, 0], x1[i, 1], 1.0])
        l2 = F @ x1_hom  # l2 = Fx1
        
        x2_hom = np.array([x2[i, 0], x2[i, 1], 1.0])
        l1 = F.T @ x2_hom  # l1 = F^T x2
        
        def draw_line(ax, l, W, H, color):
            a, b, c = l
            if abs(b) > abs(a):
                x_vals = np.array([0, W])
                y_vals = -(a * x_vals + c) / b
            else:
                y_vals = np.array([0, H])
                x_vals = -(b * y_vals + c) / a
            ax.plot(x_vals, y_vals, color=color, alpha=0.6, linewidth=1)
        
        ax1.scatter(x1[i, 0], x1[i, 1], color=color, s=40, zorder=5, edgecolors='k', linewidths=0.5)
        draw_line(ax1, l1, W, H, color)
        
        ax2.scatter(x2[i, 0], x2[i, 1], color=color, s=40, zorder=5, edgecolors='k', linewidths=0.5)
        draw_line(ax2, l2, W, H, color)
    
    e, e_prime = compute_epipoles(F)
    
    if abs(e[2]) > 1e-10 and 0 <= e[0]/e[2] <= W and 0 <= e[1]/e[2] <= H:
        ax1.scatter(e[0]/e[2], e[1]/e[2], color='red', s=200, marker='*',
                   zorder=10, label='Epipole')
        ax1.legend()
    
    if abs(e_prime[2]) > 1e-10 and 0 <= e_prime[0]/e_prime[2] <= W and 0 <= e_prime[1]/e_prime[2] <= H:
        ax2.scatter(e_prime[0]/e_prime[2], e_prime[1]/e_prime[2],
                   color='red', s=200, marker='*', zorder=10, label="Epipole'")
        ax2.legend()
    
    ax1.set_xlim(0, W); ax1.set_ylim(H, 0)
    ax2.set_xlim(0, W); ax2.set_ylim(H, 0)
    ax1.set_title("Image 1: Points & Epipolar Lines from Image 2", fontsize=12)
    ax2.set_title("Image 2: Points & Epipolar Lines from Image 1", fontsize=12)
    ax1.set_aspect('equal'); ax2.set_aspect('equal')
    ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def draw_matches(
    x1: np.ndarray, x2: np.ndarray, inliers: np.ndarray,
    img_size: Tuple[int, int] = (640, 480),
    filename: str = "matches.png"
):
    """Draw inlier/outlier correspondences side by side."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))
    
    W, H = img_size
    
    rect1 = plt.Rectangle((0, 0), W, H, fill=False, edgecolor='gray', linewidth=2)
    rect2 = plt.Rectangle((W + 20, 0), W, H, fill=False, edgecolor='gray', linewidth=2)
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    
    offset = W + 20
    
    # Draw matches
    for i in range(x1.shape[0]):
        color = 'lime' if inliers[i] else 'red'
        alpha = 0.7 if inliers[i] else 0.3
        lw = 1.0 if inliers[i] else 0.5
        
        ax.plot([x1[i, 0], x2[i, 0] + offset],
                [x1[i, 1], x2[i, 1]],
                color=color, alpha=alpha, linewidth=lw)
    
    ax.scatter(x1[:, 0], x1[:, 1], c=['lime' if inl else 'red' for inl in inliers],
              s=15, zorder=5, edgecolors='k', linewidths=0.3)
    ax.scatter(x2[:, 0] + offset, x2[:, 1], c=['lime' if inl else 'red' for inl in inliers],
              s=15, zorder=5, edgecolors='k', linewidths=0.3)
    
    n_inliers = np.sum(inliers)
    n_total = len(inliers)
    ax.set_title(f"Correspondences: {n_inliers}/{n_total} inliers "
                f"({100*n_inliers/n_total:.1f}%)", fontsize=14)
    ax.set_xlim(-10, 2*W + 30)
    ax.set_ylim(H + 10, -10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_3d_reconstruction(
    X_true: np.ndarray, X_reconstructed: np.ndarray,
    R_true: np.ndarray, t_true: np.ndarray,
    R_est: np.ndarray, t_est: np.ndarray,
    filename: str = "reconstruction_3d.png"
):
    """Plot true vs reconstructed 3D points and cameras."""
    fig = plt.figure(figsize=(16, 7))
    
    # --- True Scene ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X_true[:, 0], X_true[:, 1], X_true[:, 2],
               c='steelblue', s=10, alpha=0.6, label='3D Points')
    
    draw_camera(ax1, np.eye(3), np.zeros(3), 'Camera 1 (true)', 'green')
    draw_camera(ax1, R_true, t_true, 'Camera 2 (true)', 'red')
    
    ax1.set_title("Ground Truth", fontsize=14)
    ax1.legend(fontsize=8)
    set_axes_equal(ax1)
    
    # --- Reconstructed Scene ---
    ax2 = fig.add_subplot(122, projection='3d')
    

    if X_reconstructed is not None and len(X_reconstructed) > 0:
        scale = np.median(np.linalg.norm(X_true, axis=1)) / \
                np.maximum(np.median(np.linalg.norm(X_reconstructed, axis=1)), 1e-10)
        X_scaled = X_reconstructed * scale
        
        ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2],
                   c='coral', s=10, alpha=0.6, label='Reconstructed')
    
    draw_camera(ax2, np.eye(3), np.zeros(3), 'Camera 1', 'green')
    if R_est is not None and t_est is not None:
        draw_camera(ax2, R_est, t_est, 'Camera 2 (est)', 'orange')
    
    ax2.set_title("Reconstruction", fontsize=14)
    ax2.legend(fontsize=8)
    set_axes_equal(ax2)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def draw_camera(ax, R, t, label, color, scale=0.5):
    """Draw a simple camera frustum in 3D."""
    # Camera center in world coordinates
    if np.allclose(t, 0):
        C = np.zeros(3)
    else:
        C = -R.T @ t
    
    ax.scatter(*C, color=color, s=100, marker='^', label=label, zorder=5)
    
    # Draw optical axis
    axis_dir = R.T @ np.array([0, 0, 1]) * scale
    ax.quiver(*C, *axis_dir, color=color, arrow_length_ratio=0.2, linewidth=2)


def set_axes_equal(ax):
    """Set equal aspect ratio for 3D plot."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = np.mean(limits, axis=1)
    max_range = np.max(limits[:, 1] - limits[:, 0]) / 2
    ax.set_xlim3d([center[0] - max_range, center[0] + max_range])
    ax.set_ylim3d([center[1] - max_range, center[1] + max_range])
    ax.set_zlim3d([center[2] - max_range, center[2] + max_range])


def plot_error_comparison(
    F_8pt: np.ndarray, F_ransac: np.ndarray, F_refined: np.ndarray,
    x1_inliers: np.ndarray, x2_inliers: np.ndarray,
    filename: str = "error_comparison.png"
):
    """Compare error distributions of different F estimation methods."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    methods = [
        ("8-Point Algorithm", F_8pt),
        ("RANSAC + 8-Point", F_ransac),
        ("Gold Standard (MLE)", F_refined)
    ]
    
    for ax, (name, F) in zip(axes, methods):
        if F is None:
            ax.set_title(f"{name}\n(failed)")
            continue
        
        errors = np.sqrt(sampson_distance(F, x1_inliers, x2_inliers))
        
        ax.hist(errors, bins=40, color='steelblue', edgecolor='white',
                alpha=0.8, density=True)
        ax.axvline(np.mean(errors), color='red', linestyle='--',
                  label=f'Mean: {np.mean(errors):.4f}')
        ax.axvline(np.median(errors), color='orange', linestyle='--',
                  label=f'Median: {np.median(errors):.4f}')
        ax.set_title(f"{name}\nRMS: {np.sqrt(np.mean(errors**2)):.4f} px", fontsize=12)
        ax.set_xlabel("Sampson Error (px)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_svd_analysis(F: np.ndarray, filename: str = "svd_analysis.png"):
    """Visualize the SVD decomposition of F."""
    U, S, Vt = np.linalg.svd(F)
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    
    matrices = [
        ("U (left singular vectors)", U),
        (f"Σ = diag({S[0]:.4f}, {S[1]:.4f}, {S[2]:.2e})", np.diag(S)),
        ("V^T (right singular vectors)", Vt),
        ("F = UΣV^T", F)
    ]
    
    for ax, (title, M) in zip(axes, matrices):
        im = ax.imshow(M, cmap='RdBu_r', aspect='equal',
                       vmin=-np.max(np.abs(M)), vmax=np.max(np.abs(M)))
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.7)
        
        # Annotate values
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{M[i,j]:.3f}', ha='center', va='center',
                       fontsize=7, color='k')
    
    plt.suptitle("SVD Decomposition of F: Rank-2 Structure", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


# =============================================================================
# SECTION 12: Comparison with OpenCV
# =============================================================================

def opencv_fundamental(
    x1: np.ndarray, x2: np.ndarray, method: int = cv2.FM_8POINT
) -> np.ndarray:
    """Compute F using OpenCV for comparison."""
    F, mask = cv2.findFundamentalMat(x1, x2, method)
    if F is not None:
        F = F / np.linalg.norm(F)
    return F


def compare_matrices(F1: np.ndarray, F2: np.ndarray, name1: str, name2: str):
    """
    Compare two fundamental matrices.
    Since F is defined up to scale, compare the normalized versions.
    """
    if F1 is None or F2 is None:
        print("  Cannot compare: one matrix is None")
        return
    
    F1_n = F1 / np.linalg.norm(F1)
    F2_n = F2 / np.linalg.norm(F2)
    
    diff1 = np.linalg.norm(F1_n - F2_n)
    diff2 = np.linalg.norm(F1_n + F2_n)
    diff = min(diff1, diff2)
    
    print(f"  ||{name1} - {name2}|| = {diff:.6f}")
    
    S1 = np.linalg.svd(F1_n, compute_uv=False)
    S2 = np.linalg.svd(F2_n, compute_uv=False)
    print(f"    Singular values {name1}: [{S1[0]:.6f}, {S1[1]:.6f}, {S1[2]:.2e}]")
    print(f"    Singular values {name2}: [{S2[0]:.6f}, {S2[1]:.6f}, {S2[2]:.2e}]")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("=" * 72)
    print(" EPIPOLAR GEOMETRY PIPELINE")
    print(" Chapter 9 — Multiple View Geometry (Hartley & Zisserman)")
    print("=" * 72)
    
    # ------------------------------------------------------------------
    # STEP 1: Generate synthetic scene
    # ------------------------------------------------------------------
    print("\n[STEP 1] Generating synthetic scene...")
    X_world, K1, K2, R_true, t_true = generate_synthetic_scene(n_points=500)
    
    print(f"  3D points: {X_world.shape[0]}")
    print(f"  K1 (focal length): {K1[0,0]:.0f} px")
    print(f"  K2 (focal length): {K2[0,0]:.0f} px")
    print(f"  Rotation angle: {np.degrees(np.arccos((np.trace(R_true)-1)/2)):.1f}°")
    print(f"  Translation: [{t_true[0]:.2f}, {t_true[1]:.2f}, {t_true[2]:.2f}]")
    
    # Project to both cameras
    # Camera 1: P1 = K1 [I | 0]
    x1_true = project_points(X_world, K1, np.eye(3), np.zeros(3))
    # Camera 2: P2 = K2 [R | t]
    x2_true = project_points(X_world, K2, R_true, t_true)
    
    # Filter points within image bounds
    img_size = (640, 480)
    valid = ((x1_true[:, 0] > 0) & (x1_true[:, 0] < img_size[0]) &
             (x1_true[:, 1] > 0) & (x1_true[:, 1] < img_size[1]) &
             (x2_true[:, 0] > 0) & (x2_true[:, 0] < img_size[0]) &
             (x2_true[:, 1] > 0) & (x2_true[:, 1] < img_size[1]))
    
    x1_true = x1_true[valid]
    x2_true = x2_true[valid]
    X_world = X_world[valid]
    print(f"  Visible points: {x1_true.shape[0]}")
    # Add noise and outliers
    x1_noisy, x2_noisy, is_inlier = add_noise_and_outliers(
        x1_true, x2_true, noise_std=0.8, outlier_ratio=0.20
    )
    print(f"  Total correspondences (with outliers): {x1_noisy.shape[0]}")
    print(f"  True inlier ratio: {np.mean(is_inlier)*100:.1f}%")
    
    # ------------------------------------------------------------------
    # STEP 2: Compute ground-truth F
    # ------------------------------------------------------------------
    print("\n[STEP 2] Computing ground-truth fundamental matrix...")
    
    # F_true from the essential matrix: E = [t]_x R, F = K2^{-T} E K1^{-1}
    t_skew = np.array([
        [0, -t_true[2], t_true[1]],
        [t_true[2], 0, -t_true[0]],
        [-t_true[1], t_true[0], 0]
    ])
    E_true = t_skew @ R_true
    F_true = np.linalg.inv(K2).T @ E_true @ np.linalg.inv(K1)
    F_true = F_true / np.linalg.norm(F_true)
    
    print(f"  F_true rank: {np.linalg.matrix_rank(F_true, tol=1e-8)}")
    S_true = np.linalg.svd(F_true, compute_uv=False)
    print(f"  Singular values: [{S_true[0]:.6f}, {S_true[1]:.6f}, {S_true[2]:.2e}]")
    
    # Verify: x2^T F x1 should be ~0 for true correspondences
    errors_true = np.abs(algebraic_error(F_true, x1_true, x2_true))
    print(f"  Algebraic error on true points: mean={np.mean(errors_true):.2e}, "
          f"max={np.max(errors_true):.2e}")
    
    # ------------------------------------------------------------------
    # STEP 3: 8-Point Algorithm (on all points, including outliers)
    # ------------------------------------------------------------------
    print("\n[STEP 3] 8-Point Algorithm (all points, with outliers)...")
    F_8pt_all = eight_point_algorithm(x1_noisy, x2_noisy)
    
    sampson_8pt = np.sqrt(sampson_distance(F_8pt_all, x1_true, x2_true))
    print(f"  Sampson RMS on true inliers: {np.sqrt(np.mean(sampson_8pt**2)):.4f} px")
    compare_matrices(F_true, F_8pt_all, "F_true", "F_8pt_all")
    
    # ------------------------------------------------------------------
    # STEP 4: 8-Point Algorithm (on true inliers only — best case)
    # ------------------------------------------------------------------
    print("\n[STEP 4] 8-Point Algorithm (true inliers only)...")
    x1_in = x1_noisy[is_inlier]
    x2_in = x2_noisy[is_inlier]
    F_8pt_clean = eight_point_algorithm(x1_in, x2_in)
    
    sampson_clean = np.sqrt(sampson_distance(F_8pt_clean, x1_true, x2_true))
    print(f"  Sampson RMS on true inliers: {np.sqrt(np.mean(sampson_clean**2)):.4f} px")
    compare_matrices(F_true, F_8pt_clean, "F_true", "F_8pt_clean")
    
    # ------------------------------------------------------------------
    # STEP 5: RANSAC with 7-Point Algorithm
    # ------------------------------------------------------------------
    print("\n[STEP 5] RANSAC + 7-Point Algorithm...")
    F_ransac, ransac_inliers = ransac_fundamental(
        x1_noisy, x2_noisy,
        n_iterations=3000,
        threshold=3.0,
        use_7point=True
    )
    
    if F_ransac is not None:
        sampson_ransac = np.sqrt(sampson_distance(F_ransac, x1_true, x2_true))
        print(f"  RANSAC inliers found: {np.sum(ransac_inliers)}/{len(ransac_inliers)}")
        print(f"  Inlier detection accuracy: "
              f"{np.mean(ransac_inliers == is_inlier)*100:.1f}%")
        print(f"  Sampson RMS on true inliers: "
              f"{np.sqrt(np.mean(sampson_ransac**2)):.4f} px")
        compare_matrices(F_true, F_ransac, "F_true", "F_ransac")
    else:
        print("  RANSAC failed!")
    
    # ------------------------------------------------------------------
    # STEP 6: Gold Standard (MLE) Refinement
    # ------------------------------------------------------------------
    print("\n[STEP 6] Gold Standard (MLE) Refinement...")
    if F_ransac is not None:
        x1_ransac_in = x1_noisy[ransac_inliers]
        x2_ransac_in = x2_noisy[ransac_inliers]
        
        F_refined = gold_standard_refinement(F_ransac, x1_ransac_in, x2_ransac_in)
        
        sampson_refined = np.sqrt(sampson_distance(F_refined, x1_true, x2_true))
        print(f"  Sampson RMS on true inliers: "
              f"{np.sqrt(np.mean(sampson_refined**2)):.4f} px")
        compare_matrices(F_true, F_refined, "F_true", "F_refined")
    else:
        F_refined = None
    
    # ------------------------------------------------------------------
    # STEP 7: OpenCV Comparison
    # ------------------------------------------------------------------
    print("\n[STEP 7] OpenCV comparison...")
    F_cv_8pt = opencv_fundamental(x1_in, x2_in, cv2.FM_8POINT)
    F_cv_ransac, _ = cv2.findFundamentalMat(x1_noisy, x2_noisy, cv2.FM_RANSAC, 3.0)
    if F_cv_ransac is not None:
        F_cv_ransac = F_cv_ransac / np.linalg.norm(F_cv_ransac)
    
    if F_cv_8pt is not None:
        compare_matrices(F_true, F_cv_8pt, "F_true", "F_cv_8pt")
    if F_cv_ransac is not None:
        compare_matrices(F_true, F_cv_ransac, "F_true", "F_cv_ransac")
    
    # ------------------------------------------------------------------
    # STEP 8: Essential Matrix & Pose Recovery
    # ------------------------------------------------------------------
    print("\n[STEP 8] Essential Matrix & Pose Recovery...")
    F_best = F_refined if F_refined is not None else F_ransac
    
    E_est = compute_essential_matrix(F_best, K1, K2)
    S_E = np.linalg.svd(E_est, compute_uv=False)
    print(f"  E singular values: [{S_E[0]:.4f}, {S_E[1]:.4f}, {S_E[2]:.2e}]")
    print("  (Should be [σ, σ, 0] for a valid essential matrix)")
    
    R_est, t_est = recover_pose(E_est, K1, K2, x1_in, x2_in)
    
    if R_est is not None:
        # Compare with ground truth
        angle_err = np.degrees(np.arccos(
            np.clip((np.trace(R_est @ R_true.T) - 1) / 2, -1, 1)
        ))
        
        # Translation direction error (t is only up to scale)
        t_true_dir = t_true / np.linalg.norm(t_true)
        t_est_dir = t_est / np.linalg.norm(t_est)
        t_angle = np.degrees(np.arccos(np.clip(
            abs(np.dot(t_true_dir, t_est_dir)), 0, 1
        )))
        
        print(f"  Rotation error: {angle_err:.4f}°")
        print(f"  Translation direction error: {t_angle:.4f}°")
    else:
        print("  Pose recovery failed!")
    
    # ------------------------------------------------------------------
    # STEP 9: Triangulation & 3D Reconstruction
    # ------------------------------------------------------------------
    print("\n[STEP 9] Triangulation & 3D Reconstruction...")
    
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    
    if R_est is not None and t_est is not None:
        P2 = K2 @ np.hstack([R_est, t_est.reshape(3, 1)])
        
        # Linear (DLT) triangulation
        X_dlt = triangulate_points(P1, P2, x1_in, x2_in)
        
        # Filter out bad reconstructions
        valid_mask = (np.abs(X_dlt) < 100).all(axis=1) & (X_dlt[:, 2] > 0)
        X_dlt_valid = X_dlt[valid_mask]
        
        print(f"  DLT triangulated points: {X_dlt_valid.shape[0]}/{x1_in.shape[0]}")
        
        # Compute reprojection error
        x1_reproj = project_points(X_dlt_valid, K1, np.eye(3), np.zeros(3))
        x2_reproj = project_points(X_dlt_valid, K2, R_est, t_est)
        
        reproj_err_1 = np.sqrt(np.mean(
            np.sum((x1_in[valid_mask] - x1_reproj)**2, axis=1)
        ))
        reproj_err_2 = np.sqrt(np.mean(
            np.sum((x2_in[valid_mask] - x2_reproj)**2, axis=1)
        ))
        
        print(f"  Reprojection RMS (image 1): {reproj_err_1:.4f} px")
        print(f"  Reprojection RMS (image 2): {reproj_err_2:.4f} px")
        
        # Optimal triangulation for comparison
        print("\n  Optimal triangulation (Hartley-Sturm) on first 20 points...")
        n_optimal = min(20, x1_in.shape[0])
        X_optimal = np.zeros((n_optimal, 3))
        for i in range(n_optimal):
            X_optimal[i] = triangulate_optimal(
                P1, P2, x1_in[i], x2_in[i], F_best
            )
        
        x1_reproj_opt = project_points(X_optimal, K1, np.eye(3), np.zeros(3))
        reproj_err_opt = np.sqrt(np.mean(
            np.sum((x1_in[:n_optimal] - x1_reproj_opt)**2, axis=1)
        ))
        print(f"  Optimal reprojection RMS (image 1): {reproj_err_opt:.4f} px")
    else:
        X_dlt_valid = None
    
    # ------------------------------------------------------------------
    # STEP 10: Epipole Computation
    # ------------------------------------------------------------------
    print("\n[STEP 10] Epipole Analysis...")
    e, e_prime = compute_epipoles(F_best)
    print(f"  Epipole in image 1: ({e[0]/e[2]:.1f}, {e[1]/e[2]:.1f})")
    print(f"  Epipole in image 2: ({e_prime[0]/e_prime[2]:.1f}, "
          f"{e_prime[1]/e_prime[2]:.1f})")
    
    # Verify: Fe = 0, F^T e' = 0
    print(f"  ||F e||  = {np.linalg.norm(F_best @ e):.2e} (should be ~0)")
    print(f"  ||F^T e'|| = {np.linalg.norm(F_best.T @ e_prime):.2e} (should be ~0)")
    
    # ------------------------------------------------------------------
    # STEP 11: Error Metric Comparison
    # ------------------------------------------------------------------
    print("\n[STEP 11] Error Metric Comparison on Inliers...")
    
    metrics = {
        "Algebraic": np.abs(algebraic_error(F_best, x1_in, x2_in)),
        "Sampson": np.sqrt(sampson_distance(F_best, x1_in, x2_in)),
        "Symmetric Epipolar": np.sqrt(symmetric_epipolar_distance(F_best, x1_in, x2_in))
    }
    
    for name, errors in metrics.items():
        print(f"  {name:25s}: mean={np.mean(errors):.6f}  "
              f"median={np.median(errors):.6f}  max={np.max(errors):.6f}")
    
    # ------------------------------------------------------------------
    # VISUALIZATIONS
    # ------------------------------------------------------------------
    print("\n[VISUALIZATIONS] Generating plots...")
    
    output_dir = "2_camera_estimation/output/"
    
    draw_epipolar_lines(F_best, x1_in, x2_in, img_size,
                       filename=f"{output_dir}/01_epipolar_lines.png")
    
    draw_matches(x1_noisy, x2_noisy, ransac_inliers if F_ransac is not None else is_inlier,
                img_size, filename=f"{output_dir}/02_matches.png")
    
    if X_dlt_valid is not None:
        plot_3d_reconstruction(X_world, X_dlt_valid,
                             R_true, t_true, R_est, t_est,
                             filename=f"{output_dir}/03_reconstruction_3d.png")
    
    plot_error_comparison(F_8pt_clean, F_ransac, F_refined,
                         x1_in, x2_in,
                         filename=f"{output_dir}/04_error_comparison.png")
    
    plot_svd_analysis(F_best, filename=f"{output_dir}/05_svd_analysis.png")
    
    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(" SUMMARY")
    print("=" * 72)
    print(f"""
    Scene: {X_world.shape[0]} 3D points, {x1_noisy.shape[0]} correspondences ({np.mean(~is_inlier)*100:.0f}% outliers)
    
    Method                     Sampson RMS (px)    Matrix Error
    ─────────────────────────  ──────────────────  ────────────
    8-Point (with outliers)    {np.sqrt(np.mean(sampson_distance(F_8pt_all, x1_true, x2_true))):.4f}              {min(np.linalg.norm(F_true/np.linalg.norm(F_true) - F_8pt_all/np.linalg.norm(F_8pt_all)), np.linalg.norm(F_true/np.linalg.norm(F_true) + F_8pt_all/np.linalg.norm(F_8pt_all))):.6f}
    8-Point (clean inliers)    {np.sqrt(np.mean(sampson_distance(F_8pt_clean, x1_true, x2_true))):.4f}              {min(np.linalg.norm(F_true/np.linalg.norm(F_true) - F_8pt_clean/np.linalg.norm(F_8pt_clean)), np.linalg.norm(F_true/np.linalg.norm(F_true) + F_8pt_clean/np.linalg.norm(F_8pt_clean))):.6f}
    RANSAC + 7-Point           {np.sqrt(np.mean(sampson_distance(F_ransac, x1_true, x2_true))):.4f}              {min(np.linalg.norm(F_true/np.linalg.norm(F_true) - F_ransac/np.linalg.norm(F_ransac)), np.linalg.norm(F_true/np.linalg.norm(F_true) + F_ransac/np.linalg.norm(F_ransac))):.6f}""")
    
    if F_refined is not None:
        print(f"Gold Standard (MLE)\t{np.sqrt(np.mean(sampson_distance(F_refined, x1_true, x2_true))):.4f}\t{min(np.linalg.norm(F_true/np.linalg.norm(F_true) - F_refined/np.linalg.norm(F_refined)), np.linalg.norm(F_true/np.linalg.norm(F_true) + F_refined/np.linalg.norm(F_refined))):.6f}")
    
    if R_est is not None:
        print(f"""
Pose Recovery:  
    Rotation error:              {angle_err:.4f}°
    Translation direction error: {t_angle:.4f}°
  
Triangulation:
    Reprojection RMS (image 1):  {reproj_err_1:.4f} px
    Reprojection RMS (image 2):  {reproj_err_2:.4f} px
""")
    
    print("  Output files saved to /mnt/user-data/outputs/")
    print("=" * 72)


if __name__ == "__main__":
    main()