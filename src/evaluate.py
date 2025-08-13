# ruff: noqa: T201, T201, T203
from __future__ import annotations

from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from config.config import RUN_DIR
from src.constants import GT_COL_DIVIDER, RESULTS_SEPARATOR
from src.formatting import format_gt_pairs_filepath, format_run_metrics_path, format_storing_pathes_from_run_path
from src.onto_access import OntologyAccess
from src.onto_object import OntologyEntryAttr
from src.utils import save_run_results


def plot_usage_histograms(
    tokens_usage: np.array, confidences: np.array, do_plot: bool = True, do_print: bool = True, suptitle: str = ""
) -> None:
    tokens_usage = np.array(tokens_usage)
    if do_print:
        print(f"Mean input tokens: {tokens_usage[:,0].mean():.1f}")
        print(f"Mean output tokens: {tokens_usage[:,1].mean():.1f}")
        print(f"Total input tokens: {tokens_usage[:,0].sum()}")
        print(f"Total output tokens: {tokens_usage[:,1].sum()}")

    if do_plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(suptitle)
        axes[0].hist(tokens_usage[:, 0], bins=20)
        axes[0].set_title("Input tokens usage")
        axes[1].hist(tokens_usage[:, 1], bins=20)
        axes[1].set_title("Output tokens usage")
        axes[2].hist(confidences, bins=50)
        axes[2].set_title("Confidence distribution")
        plt.show()
    return


def save_analysis_results(
    df: pd.DataFrame,
    print_results: bool = True,
    plot_confusion_matrix: bool = True,
    subtitle: str = "",
    cm_save_path: Path | None = None,
    stats_path: Path | None = None,
) -> dict:
    conf_matrix = confusion_matrix(df["Label"], df["Prediction"])

    metrics = {
        "Accuracy": accuracy_score(df["Label"], df["Prediction"]),
        "Precision": precision_score(df["Label"], df["Prediction"]),
        "Recall": recall_score(df["Label"], df["Prediction"]),
        "F1 Score": f1_score(df["Label"], df["Prediction"]),
        "Specificity": conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1]),
        "Sensitivity": conf_matrix[1][1] / (conf_matrix[1][0] + conf_matrix[1][1]),
    }

    metrics["Youden's index"] = metrics["Sensitivity"] + metrics["Specificity"] - 1

    if print_results:
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")

    if stats_path:
        save_run_results(list(metrics.items()), stats_path, RESULTS_SEPARATOR, columns=["Metric", "Value"])

    if plot_confusion_matrix or cm_save_path:
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

        if plot_confusion_matrix:
            plt.show()
        plt.close()

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
    gt_pairs = {frozenset((row["Source"], row["Target"])) for _, row in gt_df.iterrows()}
    df["Label"] = df.apply(lambda row: frozenset((row["Source"], row["Target"])) in gt_pairs, axis=1)

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


def get_predictions_with_gt(
    run_path: str, dataset_name: str, set_name: str, model: str, prompt_name: str, suffix: str = ""
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_positive_pairs_path = format_gt_pairs_filepath(dataset_name, set_name)
    pred_results_path, _, _ = format_storing_pathes_from_run_path(run_path, set_name, model, prompt_name, suffix)
    gt_df = get_gt_pairs(gt_positive_pairs_path)
    pred_df = get_pred_pairs(pred_results_path)

    return extend_preds_with_labels_info(pred_df, gt_df)


def store_run_metrics_df(
    prompt_functions: list, run_path: Path, dataset_name: str, set_name: str, model: str, suffix: str = ""
) -> pd.DataFrame:
    results_data = []
    for prompt_function in prompt_functions:
        try:
            _, stats_path, _ = format_storing_pathes_from_run_path(run_path, set_name, model, prompt_function, suffix)
            stats = pd.read_csv(stats_path, index_col=0).T
            results_data.append(
                {
                    "Prompt": prompt_function,
                    **{key: stats[key].to_numpy()[0] for key in stats.columns},
                    "Dataset": dataset_name,
                    "SubSet": set_name,
                    "Model": model,
                }
            )
        except FileNotFoundError:
            print(f"File not found for {prompt_function} in {run_path}")
            continue

    results_df = pd.DataFrame(results_data).round(4)

    results_df.to_csv(format_run_metrics_path(run_path, suffix), index=False)
    return results_df


def read_run_metrics_df(run_subdir: str, suffix: str = "") -> pd.DataFrame:
    run_metrics_path = format_run_metrics_path(RUN_DIR / run_subdir, suffix)
    return pd.read_csv(run_metrics_path)


def print_results_entry(
    res_df: pd.DataFrame, onto_tgt: OntologyAccess, onto_src: OntologyAccess, pair_type: str = "FP", idx: int = 0
) -> None:
    source_uri = res_df[res_df["Type"] == pair_type].iloc[idx]["Source"]
    target_uri = res_df[res_df["Type"] == pair_type].iloc[idx]["Target"]

    source_entry = OntologyEntryAttr(source_uri, onto_src)
    target_entry = OntologyEntryAttr(target_uri, onto_tgt)

    print(f"Processing pair {idx} of type {pair_type}")
    pprint("Source Entry:\n")
    pprint(source_entry.annotation)
    pprint("Target Entry:\n")
    pprint(target_entry.annotation)

    print(f"Parent of Source Concept: {source_entry.get_parents_preferred_names()}")
    print(f"Parent of Target Concept: {target_entry.get_parents_preferred_names()}")
