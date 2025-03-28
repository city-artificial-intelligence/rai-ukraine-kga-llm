# ruff: noqa: T201
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.constants import GT_COL_DIVIDER, RESULTS_SEPARATOR
from src.formatting import format_gt_pairs_filepath, format_storing_pathes_from_run_path
from src.utils import save_run_results


def get_predictions_with_gt(
    run_path: str, dataset_name: str, set_name: str, model: str, exp_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_positive_pairs_path = format_gt_pairs_filepath(dataset_name, set_name)
    pred_results_path, _, _ = format_storing_pathes_from_run_path(run_path, set_name, model, exp_name)
    gt_df = get_gt_pairs(gt_positive_pairs_path)
    pred_df = get_pred_pairs(pred_results_path)

    return extend_preds_with_labels_info(pred_df, gt_df)


def plot_usage_histograms(
    tokens_usage: np.array, confidences: np.array, do_plot: bool = True, do_print: bool = True, suptitle: str = ""
) -> None:
    tokens_usage = np.array(tokens_usage)
    if do_print:
        print(f"Mean input tockens: {tokens_usage[:,0].mean():.1f}")
        print(f"Mean output tockens: {tokens_usage[:,1].mean():.1f}")
        print(f"Total input tockens: {tokens_usage[:,0].sum()}")
        print(f"Total output tockens: {tokens_usage[:,1].sum()}")

    if do_plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(suptitle)
        axes[0].hist(tokens_usage[:, 0], bins=20)
        axes[0].set_title("Input tockens usage")
        axes[1].hist(tokens_usage[:, 1], bins=20)
        axes[1].set_title("Output tockens usage")
        axes[2].hist(confidences, bins=50)
        axes[2].set_title("Confidence distribution")
        plt.show()
    return


def analyze_results(
    df: pd.DataFrame,
    print_results: bool = True,
    plot_confusion_matrix: bool = True,
    subtitle: str = "",
    cm_save_path: Path | None = None,
    stats_path: Path | None = None,
) -> dict:
    accuracy = accuracy_score(df["Label"], df["Prediction"])
    precision = precision_score(df["Label"], df["Prediction"])
    recall = recall_score(df["Label"], df["Prediction"])
    f1 = f1_score(df["Label"], df["Prediction"])
    conf_matrix = confusion_matrix(df["Label"], df["Prediction"])

    metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1}

    if print_results:
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")

    if stats_path:
        save_run_results(list(metrics.items()), stats_path, RESULTS_SEPARATOR, columns=["Metric", "Value"])

    if plot_confusion_matrix:
        plt.figure(figsize=(4, 4))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"{subtitle}")
        if cm_save_path:
            if not cm_save_path.parent.exists():
                cm_save_path.parent.mkdir(parents=True)
            plt.savefig(cm_save_path)
        plt.show()

    return {**metrics, "confusion matrix": conf_matrix}


def get_gt_pairs(gt_positive_pairs_path: Path | str, sep: str = GT_COL_DIVIDER) -> pd.DataFrame:
    gt_positive_pairs = pd.read_csv(gt_positive_pairs_path, sep=sep, header=None)
    if len(gt_positive_pairs.columns) == 3:
        gt_positive_pairs.columns = ["Source", "Target", "Label"]
    else:
        gt_positive_pairs.columns = ["Source", "Target", "=", "Label", "notes"]
    return gt_positive_pairs


def get_pred_pairs(pred_results_path: Path | str, sep: str = RESULTS_SEPARATOR) -> pd.DataFrame:
    return pd.read_csv(pred_results_path, sep=sep)


def extend_preds_with_labels_info(df: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
    df["Label"] = False
    for _, row in gt_df.iterrows():
        source = row["Source"]
        target = row["Target"]
        mask = (df["Source"] == source) & (df["Target"] == target) | (df["Source"] == target) & (df["Target"] == source)
        df.loc[mask, "Label"] = True

    df["Type"] = df.apply(
        lambda x: "TP"
        if x["Label"] and x["Prediction"]
        else "TN"
        if not x["Label"] and not x["Prediction"]
        else "FP"
        if not x["Label"] and x["Prediction"]
        else "FN",
        axis=1,
    )
    return df
