import os
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate


def load_ratings_dataset(csv_path: str = "ratings_small.csv"):
    """Load ratings_small.csv and return a Surprise Dataset object. """
    df = pd.read_csv(csv_path)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)
    return data


def run_cv(
    algo,
    data,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[float, float]:
    """Run k-fold cross-validation on a Surprise algorithm."""
    results = cross_validate(
        algo,
        data,
        measures=["RMSE", "MAE"],
        cv=n_splits,
        verbose=False,
        n_jobs=1,
    )

    mean_rmse = float(np.mean(results["test_rmse"]))
    mean_mae = float(np.mean(results["test_mae"]))
    return mean_mae, mean_rmse


def format_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Return a simple aligned text table as a string for console output."""
    headers = [str(h) for h in headers]
    rows_str = [[str(cell) for cell in row] for row in rows]

    col_widths = [len(h) for h in headers]
    for row in rows_str:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(vals: Sequence[str]) -> str:
        return " | ".join(vals[i].ljust(col_widths[i]) for i in range(len(vals)))

    sep = "-+-".join("-" * w for w in col_widths)

    lines = [fmt_row(headers), sep]
    for row in rows_str:
        lines.append(fmt_row(row))

    return "\n".join(lines)


def format_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Return a  Markdown table string."""
    header_line = " | ".join(headers)
    sep_line = " | ".join(["---"] * len(headers))
    row_lines = [" | ".join(str(cell) for cell in row) for row in rows]
    return "\n".join([header_line, sep_line] + row_lines)


def make_bar_plot(
    x_labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    filename: str,
):
    """Create and save a bar plot."""
    plt.figure(figsize=(6, 4))
    x_pos = np.arange(len(x_labels))
    plt.bar(x_pos, values, color="skyblue")
    plt.xticks(x_pos, x_labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def make_line_plot(
    x_values: Sequence[float],
    y_values: Sequence[float],
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
):
    """Create and save a line plot."""
    plt.figure(figsize=(6, 4))
    plt.plot(x_values, y_values, marker="o", color="steelblue")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
