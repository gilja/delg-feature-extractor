import numpy as np
from typing import List, Dict

from scipy.spatial import cKDTree  # type: ignore
from skimage import measure
from skimage import transform


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """pip
    Computes cosine similarity between two global descriptor vectors.

    Returns:
        A float value between -1 and 1 (1 = identical, 0 = orthogonal).
    """
    a = np.array(vec1)
    b = np.array(vec2)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def local_feature_match(
    f1: Dict,
    f2: Dict,
    ratio_thresh: float = 0.8,
    ransac_residual_threshold: float = 10.0,
    min_inliers: int = 10,
) -> bool:
    """
    Determines whether two images match based on their local features.

    Args:
        f1 (Dict): Dictionary containing 'locations' and 'descriptors' for image 1.
        f2 (Dict): Dictionary containing 'locations' and 'descriptors' for image 2.
        ratio_thresh (float): Lowe's ratio test threshold.
        ransac_residual_threshold (float): RANSAC residual threshold for geometric verification.
        min_inliers (int): Minimum number of inliers required to consider images as matching.

    Returns:
        bool: True if images match, False otherwise.
    """
    # Step 1: Extract descriptors and locations from both images
    desc1 = np.array(f1["descriptors"])
    loc1 = np.array(f1["locations"])
    desc2 = np.array(f2["descriptors"])
    loc2 = np.array(f2["locations"])

    # Early exit if there are too few local features.
    if desc1.shape[0] < 3 or desc2.shape[0] < 3:
        return False

    # Step 2: Match descriptors using KD-tree and Lowe's ratio test
    index_tree = cKDTree(desc2)
    distances, indices = index_tree.query(desc1, k=2, workers=-1)

    matched_query_points = []
    matched_index_points = []

    for i, row in enumerate(distances):
        if row[0] < ratio_thresh * row[1]:
            matched_query_points.append(loc1[i])
            matched_index_points.append(loc2[indices[i][0]])

    # Step 3: Run RANSAC to find an affine transformation and count inliers
    matched_query_points = np.array(matched_query_points)
    matched_index_points = np.array(matched_index_points)

    # Early exit if not enough putative matches
    if matched_query_points.shape[0] < 3:
        return False

    _, inliers = measure.ransac(
        (matched_index_points, matched_query_points),
        transform.AffineTransform,
        min_samples=3,
        residual_threshold=ransac_residual_threshold,
        max_trials=1000,
    )

    num_inliers = np.sum(inliers)

    # Step 4: Decision based on inliers
    print("Num inliers: {num_inliers}")
    if num_inliers >= min_inliers:
        return True

    return False
