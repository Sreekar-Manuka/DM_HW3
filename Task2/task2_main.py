from typing import Dict, List, Sequence, Tuple

import numpy as np

from surprise import KNNBasic, SVD

from recsys_utils import (
    format_markdown_table,
    format_table,
    load_ratings_dataset,
    make_bar_plot,
    make_line_plot,
    run_cv,
)


def run_base_models(data):
    """Q2c / Q2d: Evaluate PMF, User-CF, Item-CF with cosine similarity."""
    results = []

    
    algo_pmf = SVD(random_state=42)
    mae_pmf, rmse_pmf = run_cv(algo_pmf, data)
    results.append(("PMF", mae_pmf, rmse_pmf))

    
    sim_user_cosine = {"name": "cosine", "user_based": True}
    algo_user = KNNBasic(k=40, sim_options=sim_user_cosine)
    mae_user, rmse_user = run_cv(algo_user, data)
    results.append(("User-CF", mae_user, rmse_user))

    
    sim_item_cosine = {"name": "cosine", "user_based": False}
    algo_item = KNNBasic(k=40, sim_options=sim_item_cosine)
    mae_item, rmse_item = run_cv(algo_item, data)
    results.append(("Item-CF", mae_item, rmse_item))

    return results


def run_similarity_study(data):
    """Q2e: Study similarity metrics for User-CF and Item-CF."""
    similarities = ["cosine", "msd", "pearson"]

    user_results = []  
    item_results = []  

    for sim in similarities:
        
        sim_user = {"name": sim, "user_based": True}
        algo_user = KNNBasic(k=40, sim_options=sim_user)
        mae_user, rmse_user = run_cv(algo_user, data)
        user_results.append((sim, mae_user, rmse_user))

        
        sim_item = {"name": sim, "user_based": False}
        algo_item = KNNBasic(k=40, sim_options=sim_item)
        mae_item, rmse_item = run_cv(algo_item, data)
        item_results.append((sim, mae_item, rmse_item))

    
    rmse_table = []
    for sim in similarities:
        rmse_user = next(r[2] for r in user_results if r[0] == sim)
        rmse_item = next(r[2] for r in item_results if r[0] == sim)
        rmse_table.append((sim, rmse_user, rmse_item))

    # Bar plots
    user_rmse_vals = [r[2] for r in user_results]
    item_rmse_vals = [r[2] for r in item_results]

    make_bar_plot(
        x_labels=similarities,
        values=user_rmse_vals,
        title="User-CF: RMSE vs Similarity Metric",
        ylabel="RMSE",
        filename="user_cf_similarity_rmse.png",
    )

    make_bar_plot(
        x_labels=similarities,
        values=item_rmse_vals,
        title="Item-CF: RMSE vs Similarity Metric",
        ylabel="RMSE",
        filename="item_cf_similarity_rmse.png",
    )

    return user_results, item_results, rmse_table


def run_k_neighbors_study(data):
    """Q2f / Q2g: Study the effect of k on RMSE for User-CF and Item-CF."""
    k_values = [5, 10, 20, 40, 60, 80]

    user_rmse = []
    item_rmse = []

    for k in k_values:
        sim_user = {"name": "cosine", "user_based": True}
        algo_user = KNNBasic(k=k, sim_options=sim_user)
        _, rmse_user = run_cv(algo_user, data)
        user_rmse.append(rmse_user)

        sim_item = {"name": "cosine", "user_based": False}
        algo_item = KNNBasic(k=k, sim_options=sim_item)
        _, rmse_item = run_cv(algo_item, data)
        item_rmse.append(rmse_item)

    # Line plots
    make_line_plot(
        x_values=k_values,
        y_values=user_rmse,
        title="User-CF: RMSE vs k",
        xlabel="k (neighbors)",
        ylabel="RMSE",
        filename="user_cf_k_rmse.png",
    )

    make_line_plot(
        x_values=k_values,
        y_values=item_rmse,
        title="Item-CF: RMSE vs k",
        xlabel="k (neighbors)",
        ylabel="RMSE",
        filename="item_cf_k_rmse.png",
    )

    table_rows = [(k, user_rmse[i], item_rmse[i]) for i, k in enumerate(k_values)]
    return k_values, user_rmse, item_rmse, table_rows


def save_report(
    base_results,
    user_sim_results,
    item_sim_results,
    sim_rmse_table,
    k_values,
    user_k_rmse,
    item_k_rmse,
    k_table_rows,
    report_path: str = "task2_report.md",
):
    lines: List[str] = []

    lines.append("# Task 2 Report: Recommender Systems with scikit-surprise\n")

    
    lines.append("## Introduction\n")
   

    
    lines.append("## Q2c — Base Models Evaluation\n")
    lines.append(
        format_markdown_table(
            ["Model", "MAE", "RMSE"],
            [(m, f"{mae:.4f}", f"{rmse:.4f}") for (m, mae, rmse) in base_results],
        )
    )
    lines.append("")

    
    best_base = min(base_results, key=lambda x: x[2])
    lines.append("## Q2d — Best Base Model\n")
    lines.append(
        f"The best-performing base model in terms of RMSE is **{best_base[0]}** "
        f"with RMSE = {best_base[2]:.4f} and MAE = {best_base[1]:.4f}. "
       
    )

   
    lines.append("## Q2e — Similarity Metrics Study\n")
    lines.append("### RMSE Comparison for User-CF and Item-CF\n")
    lines.append(
        format_markdown_table(
            ["Similarity", "User-CF RMSE", "Item-CF RMSE"],
            [
                (sim, f"{user_rmse:.4f}", f"{item_rmse:.4f}")
                for (sim, user_rmse, item_rmse) in sim_rmse_table
            ],
        )
    )
    lines.append("")

    best_user_sim = min(user_sim_results, key=lambda x: x[2])
    best_item_sim = min(item_sim_results, key=lambda x: x[2])

    

    
    lines.append("## Q2f — Effect of Number of Neighbors k\n")
    lines.append(
        format_markdown_table(
            ["k", "User-CF RMSE", "Item-CF RMSE"],
            [
                (k, f"{user_k_rmse[i]:.4f}", f"{item_k_rmse[i]:.4f}")
                for i, k in enumerate(k_values)
            ],
        )
    )
    lines.append("")


    best_k_user_index = int(np.argmin(user_k_rmse))
    best_k_item_index = int(np.argmin(item_k_rmse))
    k_user_star = k_values[best_k_user_index]
    k_item_star = k_values[best_k_item_index]

    lines.append("\n## Q2g — Best k \n")
    lines.append(
        f"For User-CF, the best number of neighbors is **k = {k_user_star}** "
        f"with RMSE = {user_k_rmse[best_k_user_index]:.4f}. "
        f"For Item-CF, the best number of neighbors is **k = {k_item_star}** "
        f"with RMSE = {item_k_rmse[best_k_item_index]:.4f}. "
        "These optimal k values may differ because user neighborhoods and "
        "item neighborhoods can have different density and variability. "
        "User-based models may benefit from a smaller or larger neighborhood "
        "depending on how diverse user preferences are, while item-based "
        "models depend on how many similar items exist for each movie.\n"
    )


    lines.append("\n## Final Conclusion\n")

    lines.append(
        f"- **Best base model overall (by RMSE)**: {best_base[0]} (RMSE = {best_base[2]:.4f}).\n"
    )
    lines.append(
        f"- **Best similarity for User-CF**: {best_user_sim[0]} (RMSE = {best_user_sim[2]:.4f}).\n"
    )
    lines.append(
        f"- **Best similarity for Item-CF**: {best_item_sim[0]} (RMSE = {best_item_sim[2]:.4f}).\n"
    )
    lines.append(
        f"- **Best k for User-CF**: k = {k_user_star}.\n"
        f"  \n- **Best k for Item-CF**: k = {k_item_star}.\n"
    )
   

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    
    data = load_ratings_dataset("ratings_small.csv")

    
    base_results = run_base_models(data)

    print("Q2c — Base Models (5-fold CV)")
    print(
        format_table(
            ["Model", "MAE", "RMSE"],
            [(m, f"{mae:.4f}", f"{rmse:.4f}") for (m, mae, rmse) in base_results],
        )
    )
    print("")

    
    user_sim_results, item_sim_results, sim_rmse_table = run_similarity_study(data)
    print("Q2e — Similarity Metrics Study (RMSE)")
    print(
        format_table(
            ["Similarity", "User-CF RMSE", "Item-CF RMSE"],
            [
                (sim, f"{user_rmse:.4f}", f"{item_rmse:.4f}")
                for (sim, user_rmse, item_rmse) in sim_rmse_table
            ],
        )
    )
    print("")

    
    k_values, user_k_rmse, item_k_rmse, k_table_rows = run_k_neighbors_study(data)
    print("Q2f — Effect of k on RMSE")
    print(
        format_table(
            ["k", "User-CF RMSE", "Item-CF RMSE"],
            [
                (k, f"{user_k_rmse[i]:.4f}", f"{item_k_rmse[i]:.4f}")
                for i, k in enumerate(k_values)
            ],
        )
    )
    print("")

    
    print("Saving Task 2 report to task2_report.md ")
    save_report(
        base_results,
        user_sim_results,
        item_sim_results,
        sim_rmse_table,
        k_values,
        user_k_rmse,
        item_k_rmse,
        k_table_rows,
    )
    print("Done.")


if __name__ == "__main__":
    main()
