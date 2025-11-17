import time
from typing import Callable, Dict, List, Tuple

import numpy as np

from distances import (
    euclidean_distance,
    cosine_distance,
    generalized_jaccard_distance,
)
from kmeans import kmeans


DistanceFunc = Callable[[np.ndarray, np.ndarray], float]


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    X = np.loadtxt("data.csv", delimiter=",")
    y = np.loadtxt("label.csv", delimiter=",")
    
    X = X.reshape(10000, 784)
    y = y.reshape(10000,)
    return X, y


def compute_cluster_accuracy(
    labels: np.ndarray,
    true_labels: np.ndarray,
    K: int,
) -> float:
    """Compute clustering accuracy using majority vote per cluster."""
    n_samples = labels.shape[0]
    mapping = np.zeros(K, dtype=int)

    for k in range(K):
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            mapping[k] = 0
            continue
        cluster_true = true_labels[idx].astype(int)
        counts = np.bincount(cluster_true)
        mapping[k] = int(np.argmax(counts))

    pred_labels = np.array([mapping[labels[i]] for i in range(n_samples)], dtype=int)
    accuracy = float(np.mean(pred_labels == true_labels))
    return accuracy


def run_q1_q2_q3(
    X: np.ndarray,
    y: np.ndarray,
    K: int,
    metrics: Dict[str, DistanceFunc],
):
    q1_results = []
    q2_results = []
    q3_results = []

    for name, dist_func in metrics.items():
        centroids, labels, final_sse, n_iter, total_time = kmeans(
            X,
            K,
            dist_func,
            max_iter=500,
            stop_mode="combined",
            tol=1e-4,
            random_state=42,
        )

        accuracy = compute_cluster_accuracy(labels, y, K)

        q1_results.append((name, final_sse, n_iter))
        q2_results.append((name, accuracy))
        q3_results.append((name, n_iter, total_time))

    return q1_results, q2_results, q3_results


def run_q4(
    X: np.ndarray,
    y: np.ndarray,
    K: int,
    metrics: Dict[str, DistanceFunc],
):
    results = []

    for name, dist_func in metrics.items():
        
        _, _, sse_no_change, _, _ = kmeans(
            X,
            K,
            dist_func,
            max_iter=500,
            stop_mode="no_change",
            tol=1e-4,
            random_state=42,
        )

        
        _, _, sse_sse_increase, _, _ = kmeans(
            X,
            K,
            dist_func,
            max_iter=500,
            stop_mode="sse_increase",
            tol=1e-4,
            random_state=42,
        )

        
        _, _, sse_fixed, _, _ = kmeans(
            X,
            K,
            dist_func,
            max_iter=100,
            stop_mode="fixed",
            tol=1e-4,
            random_state=42,
        )

        results.append((name, sse_no_change, sse_sse_increase, sse_fixed))

    return results


def format_table(headers: List[str], rows: List[Tuple]) -> str:
    """Return a string with a simple aligned table for console output."""
    headers = [str(h) for h in headers]
    rows_str = [[str(cell) for cell in row] for row in rows]

    col_widths = [len(h) for h in headers]
    for row in rows_str:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(row_vals):
        return " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row_vals))

    sep = "-+-".join("-" * w for w in col_widths)

    lines = [fmt_row(headers), sep]
    for row in rows_str:
        lines.append(fmt_row(row))

    return "\n".join(lines)


def format_markdown_table(headers: List[str], rows: List[Tuple]) -> str:
    header_line = " | ".join(headers)
    sep_line = " | ".join(["---"] * len(headers))
    row_lines = [" | ".join(str(cell) for cell in row) for row in rows]
    return "\n".join([header_line, sep_line] + row_lines)


def save_report(
    q1_results,
    q2_results,
    q3_results,
    q4_results,
    report_path: str = "task1_report.md",
):
    lines: List[str] = []

    lines.append("# Task 1 Report: K-means Clustering\n")

    # Q1
    lines.append("## Q1 — Compare SSE across distance metrics\n")
    lines.append(
        format_markdown_table(
            ["Metric", "SSE", "Iterations"],
            [(m, f"{sse:.4f}", it) for m, sse, it in q1_results],
        )
    )
    lines.append("")

    best_q1 = min(q1_results, key=lambda x: x[1])
    lines.append(
        f"**Lowest SSE** is achieved by **{best_q1[0]}** with SSE = {best_q1[1]:.4f}.\n"
    )

    # Q2
    lines.append("\n## Q2 — Clustering accuracy for each metric\n")
    lines.append(
        format_markdown_table(
            ["Metric", "Accuracy"],
            [(m, f"{acc:.4f}") for m, acc in q2_results],
        )
    )
    lines.append("")

    best_q2 = max(q2_results, key=lambda x: x[1])
    lines.append(
        f"**Highest accuracy** is achieved by **{best_q2[0]}** with accuracy = {best_q2[1]:.4f}.\n"
    )

    # Q3
    lines.append("\n## Q3 — Iteration count and running time\n")
    lines.append(
        format_markdown_table(
            ["Metric", "Iterations", "Time (seconds)"],
            [(m, it, f"{t:.4f}") for m, it, t in q3_results],
        )
    )
    lines.append("")

    slowest = max(q3_results, key=lambda x: x[2])
    fastest = min(q3_results, key=lambda x: x[2])
    lines.append(
        f"**Slowest metric**: {slowest[0]} ({slowest[2]:.4f} s). "
        f"**Fastest metric**: {fastest[0]} ({fastest[2]:.4f} s).\n"
    )

    # Q4
    lines.append("\n## Q4 — Effect of different stopping conditions\n")
    lines.append(
        format_markdown_table(
            ["Metric", "No-Change SSE", "SSE-Increase SSE", "Fixed-100-Iter SSE"],
            [
                (
                    m,
                    f"{s1:.4f}",
                    f"{s2:.4f}",
                    f"{s3:.4f}",
                )
                for (m, s1, s2, s3) in q4_results
            ],
        )
    )
    lines.append("")

    # Q5 summary
    lines.append("\n## Q5 — Final Summary\n")


    metrics = {}
    for m, sse, _ in q1_results:
        metrics.setdefault(m, {})["sse"] = sse
    for m, acc in q2_results:
        metrics.setdefault(m, {})["acc"] = acc

    best_metric = max(
        metrics.items(),
        key=lambda kv: (kv[1].get("acc", 0.0), -kv[1].get("sse", float("inf"))),
    )[0]

    lines.append(f"- **Best overall metric**: {best_metric}.\n")
    lines.append(
        f"- **Fastest converging metric**: {fastest[0]} based on wall-clock time.\n"
    )

    
    flat_q4 = []
    for m, s1, s2, s3 in q4_results:
        flat_q4.append((m, "no_change", s1))
        flat_q4.append((m, "sse_increase", s2))
        flat_q4.append((m, "fixed_100", s3))
    best_stop = min(flat_q4, key=lambda x: x[2])

    lines.append(
        "- **Best stopping condition (by SSE)**: "
        f"{best_stop[1]} with metric {best_stop[0]} (SSE = {best_stop[2]:.4f}).\n"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    X, y = load_data()
    K = 10

    metrics: Dict[str, DistanceFunc] = {
        "Euclidean": euclidean_distance,
        "Cosine": cosine_distance,
        "Jaccard": generalized_jaccard_distance,
    }

    print("Running Q1, Q2, Q3 experiments (stop_mode = combined)...")
    q1_results, q2_results, q3_results = run_q1_q2_q3(X, y, K, metrics)

    print("\nQ1 — SSE across metrics (combined stop mode):")
    print(format_table(["Metric", "SSE", "Iterations"], q1_results))

    print("\nQ2 — Clustering accuracy:")
    print(format_table(["Metric", "Accuracy"], q2_results))

    print("\nQ3 — Iterations and running time:")
    print(format_table(["Metric", "Iterations", "Time (seconds)"], q3_results))

    print("\nRunning Q4 experiments (different stopping conditions)...")
    q4_results = run_q4(X, y, K, metrics)

    print("\nQ4 — Effect of stopping conditions:")
    print(
        format_table(
            ["Metric", "No-Change SSE", "SSE-Increase SSE", "Fixed-100-Iter SSE"],
            q4_results,
        )
    )

    print("\nSaving report to task1_report.md ")
    save_report(q1_results, q2_results, q3_results, q4_results)
    print("Done.")


if __name__ == "__main__":
    main()
