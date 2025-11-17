import time
from typing import Callable, Tuple

import numpy as np


DistanceFunc = Callable[[np.ndarray, np.ndarray], float]


def _initialize_centroids(
    X: np.ndarray,
    K: int,
    random_state: int = 42,
) -> np.ndarray:
    """Randomly choose K samples from X as initial centroids."""
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    indices = rng.choice(n_samples, size=K, replace=False)
    return X[indices].astype(float, copy=True)


def _assign_clusters(
    X: np.ndarray,
    centroids: np.ndarray,
    distance_func: DistanceFunc,
) -> np.ndarray:
    """Assign each sample in X to the index of the nearest centroid."""
    n_samples = X.shape[0]
    K = centroids.shape[0]
    labels = np.empty(n_samples, dtype=int)

    for i in range(n_samples):
        x_i = X[i]
        best_k = 0
        best_dist = float("inf")
        for k in range(K):
            d = distance_func(x_i, centroids[k])
            if d < best_dist:
                best_dist = d
                best_k = k
        labels[i] = best_k

    return labels


def _update_centroids(
    X: np.ndarray,
    labels: np.ndarray,
    K: int,
    random_state: int = 42,
) -> np.ndarray:
    """Recompute centroids as the mean of assigned samples.

    If a cluster becomes empty, reinitialize its centroid to a random sample.
    """
    n_features = X.shape[1]
    centroids = np.zeros((K, n_features), dtype=float)

    rng = np.random.RandomState(random_state)

    for k in range(K):
        members = X[labels == k]
        if len(members) == 0:
            
            idx = rng.randint(0, X.shape[0])
            centroids[k] = X[idx]
        else:
            centroids[k] = np.mean(members, axis=0)

    return centroids


def _compute_sse(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    distance_func: DistanceFunc,
) -> float:
    """Compute Sum of Squared Errors for current clustering."""
    sse = 0.0
    for i in range(X.shape[0]):
        c = centroids[labels[i]]
        d = distance_func(X[i], c)
        sse += d * d
    return float(sse)


def kmeans(
    X: np.ndarray,
    K: int,
    distance_func: DistanceFunc,
    max_iter: int = 500,
    stop_mode: str = "combined",
    tol: float = 1e-4,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float, int, float]:
    """Run K-means clustering. """
    start_time = time.time()

    centroids = _initialize_centroids(X, K, random_state=random_state)
    labels = _assign_clusters(X, centroids, distance_func)
    prev_sse = _compute_sse(X, labels, centroids, distance_func)

    for it in range(1, max_iter + 1):
        old_centroids = centroids.copy()

        
        centroids = _update_centroids(X, labels, K, random_state=random_state)

        
        labels = _assign_clusters(X, centroids, distance_func)

        
        sse = _compute_sse(X, labels, centroids, distance_func)

        
        if stop_mode == "fixed":
            
            if it == max_iter:
                prev_sse = sse
                break
        elif stop_mode == "no_change":
            shift = float(np.linalg.norm(centroids - old_centroids))
            if shift <= tol:
                prev_sse = sse
                break
        elif stop_mode == "sse_increase":
            if sse > prev_sse:
                
                centroids = old_centroids
                sse = prev_sse
                break
            prev_sse = sse
        elif stop_mode == "combined":
            shift = float(np.linalg.norm(centroids - old_centroids))
            if shift <= tol:
                prev_sse = sse
                break
            if sse > prev_sse:
                centroids = old_centroids
                sse = prev_sse
                break
            prev_sse = sse
        else:
            raise ValueError(f"Unknown stop_mode: {stop_mode}")

    end_time = time.time()
    total_time = float(end_time - start_time)

    final_sse = float(prev_sse if stop_mode in {"sse_increase", "combined"} else sse)
    n_iter = it

    return centroids, labels, final_sse, n_iter, total_time
