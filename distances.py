import numpy as np


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Euclidean (L2) distance between two 1D vectors.  """
    diff = x - y
    return float(np.sqrt(np.dot(diff, diff)))


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute cosine distance = 1 - cosine similarity between two 1D vectors. """
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
    """Compute generalized Jaccard distance for non-negative numeric vectors."""
    
    x_nonneg = np.maximum(x, 0.0)
    y_nonneg = np.maximum(y, 0.0)

    min_sum = float(np.sum(np.minimum(x_nonneg, y_nonneg)))
    max_sum = float(np.sum(np.maximum(x_nonneg, y_nonneg)))

    if max_sum == 0.0:
       
        return 0.0

    similarity = min_sum / max_sum
    return 1.0 - similarity
