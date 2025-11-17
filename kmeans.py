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
            # Reinitialize to a random sample
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
    """Run K-means clustering.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    K : int
        Number of clusters.
    distance_func : callable
        Function that takes two 1D vectors and returns a scalar distance.
    max_iter : int, default 500
        Maximum number of iterations.
    stop_mode : {"no_change", "sse_increase", "combined", "fixed"}
        Stopping condition mode.
    tol : float, default 1e-4
        Tolerance for centroid movement in "no_change" and "combined" modes.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    centroids : np.ndarray
        Final centroids of shape (K, n_features).
    labels : np.ndarray
        Cluster assignment for each sample, shape (n_samples,).
    final_sse : float
        SSE of the final clustering.
    n_iter : int
        Number of iterations executed.
    total_time : float
        Wall-clock time taken in seconds.
    """
    start_time = time.time()

    centroids = _initialize_centroids(X, K, random_state=random_state)
    labels = _assign_clusters(X, centroids, distance_func)
    prev_sse = _compute_sse(X, labels, centroids, distance_func)

    for it in range(1, max_iter + 1):
        old_centroids = centroids.copy()

        # Update step
        centroids = _update_centroids(X, labels, K, random_state=random_state)

        # Assignment step
        labels = _assign_clusters(X, centroids, distance_func)

        # Compute SSE
        sse = _compute_sse(X, labels, centroids, distance_func)

        # Check stopping conditions
        if stop_mode == "fixed":
            # Ignore all conditions, run exactly max_iter iterations
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
                # Revert to previous centroids/labels/SSE
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
