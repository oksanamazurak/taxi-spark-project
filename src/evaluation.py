from typing import List
from pyspark.sql import DataFrame
import os

# Plotting uses pandas + matplotlib/seaborn on driver
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def _ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_regression_comparison(results: List[DataFrame], output_dir: str = "/app/output") -> None:
    _ensure_output_dir(output_dir)
    if not results:
        print("[save_regression_comparison] No regression results provided.")
        return

    # Union Spark DataFrames safely
    df_all = results[0]
    for df in results[1:]:
        df_all = df_all.unionByName(df, allowMissingColumns=True)

    # Save CSV
    csv_path = os.path.join(output_dir, "regression_metrics.csv")
    df_all.coalesce(1).write.mode("overwrite").option("header", True).csv(csv_path)
    print(f"[save_regression_comparison] Saved CSV to: {csv_path}")

    # Plot RMSE and R2
    pdf = df_all.toPandas()
    if pdf.empty:
        print("[save_regression_comparison] Empty pandas dataframe, skip plotting.")
        return

    sns.set(style="whitegrid")

    # RMSE
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(data=pdf, x="model_name", y="rmse", palette="Blues_d")
    ax.set_title("Regression: RMSE by model")
    ax.set_xlabel("")
    ax.set_ylabel("RMSE")
    plt.xticks(rotation=15)
    rmse_png = os.path.join(output_dir, "regression_rmse.png")
    plt.tight_layout()
    plt.savefig(rmse_png)
    plt.close()
    print(f"[save_regression_comparison] Saved plot: {rmse_png}")

    # R2
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(data=pdf, x="model_name", y="r2", palette="Greens_d")
    ax.set_title("Regression: R2 by model")
    ax.set_xlabel("")
    ax.set_ylabel("R2")
    plt.xticks(rotation=15)
    r2_png = os.path.join(output_dir, "regression_r2.png")
    plt.tight_layout()
    plt.savefig(r2_png)
    plt.close()
    print(f"[save_regression_comparison] Saved plot: {r2_png}")


def save_classification_comparison(results: List[DataFrame], output_dir: str = "/app/output") -> None:
    _ensure_output_dir(output_dir)
    if not results:
        print("[save_classification_comparison] No classification results provided.")
        return

    # Union Spark DataFrames safely
    df_all = results[0]
    for df in results[1:]:
        df_all = df_all.unionByName(df, allowMissingColumns=True)

    # Save CSV
    csv_path = os.path.join(output_dir, "classification_metrics.csv")
    df_all.coalesce(1).write.mode("overwrite").option("header", True).csv(csv_path)
    print(f"[save_classification_comparison] Saved CSV to: {csv_path}")

    # Plot metrics
    pdf = df_all.toPandas()
    if pdf.empty:
        print("[save_classification_comparison] Empty pandas dataframe, skip plotting.")
        return

    metrics = [
        ("accuracy", "Accuracy", "Purples_d"),
        ("w_precision", "Weighted Precision", "Oranges_d"),
        ("w_recall", "Weighted Recall", "Reds_d"),
        ("w_f1", "Weighted F1", "Blues_d"),
    ]

    sns.set(style="whitegrid")
    for col, title, palette in metrics:
        if col not in pdf.columns:
            continue
        plt.figure(figsize=(9, 4))
        ax = sns.barplot(data=pdf, x="model_name", y=col, palette=palette)
        ax.set_title(f"Classification: {title} by model")
        ax.set_xlabel("")
        ax.set_ylabel(title)
        plt.xticks(rotation=15)
        out_png = os.path.join(output_dir, f"classification_{col}.png")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print(f"[save_classification_comparison] Saved plot: {out_png}")
