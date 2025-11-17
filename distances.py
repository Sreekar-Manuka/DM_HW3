import numpy as np


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Euclidean (L2) distance between two 1D vectors.

    Parameters
    ----------
    x, y : np.ndarray
        1D numpy arrays of the same length.

    Returns
    -------
    float
        Euclidean distance between x and y.
    """
    diff = x - y
    return float(np.sqrt(np.dot(diff, diff)))


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute cosine distance = 1 - cosine similarity between two 1D vectors.

    Handles zero vectors safely:
    - If both vectors are all zeros, distance is defined as 0.0.
    - If one is zero and the other is non-zero, distance is defined as 1.0.
    """
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)

    if x_norm == 0.0 and y_norm == 0.0:
        return 0.0
    if x_norm == 0.0 or y_norm == 0.0:
        return 1.0

    sim = float(np.dot(x, y) / (x_norm * y_norm))
    # Numerical safety: clip to [-1, 1]
    sim = max(min(sim, 1.0), -1.0)
    return 1.0 - sim


def generalized_jaccard_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute generalized Jaccard distance for non-negative numeric vectors.

    Jaccard similarity is defined as:
        sum(min(x_i, y_i)) / sum(max(x_i, y_i))

    Distance = 1 - similarity.

    If both vectors are all zeros (denominator = 0), we define similarity = 1
    => distance = 0.
    """
    # Ensure non-negative; if dataset can contain negatives, this could be
    # relaxed or handled differently, but here we assume non-negative inputs.
    x_nonneg = np.maximum(x, 0.0)
    y_nonneg = np.maximum(y, 0.0)

    min_sum = float(np.sum(np.minimum(x_nonneg, y_nonneg)))
    max_sum = float(np.sum(np.maximum(x_nonneg, y_nonneg)))

    if max_sum == 0.0:
        # Both are effectively zero vectors
        return 0.0

    similarity = min_sum / max_sum
    return 1.0 - similarity
