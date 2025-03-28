import time
from pathlib import Path

from config.config import DATA_DIR, RUN_DIR


def format_gt_pairs_filepath(dataset_path: str, set_name: str) -> Path:
    return DATA_DIR / dataset_path / set_name / "refs_equiv/full.tsv"


def format_oracle_pairs_filepath(dataset_name: str, set_name: str) -> Path:
    return DATA_DIR / dataset_name / set_name / f"{dataset_name}-{set_name}-logmap_mappings_to_ask_oracle_user_llm.txt"


def format_run_path(suffix: str = "") -> Path:
    date = time.strftime("%Y-%m-%d")
    time_now = time.strftime("%H-%M-%S")
    return RUN_DIR / (f"{date}_{time_now}" if not suffix else f"{date}_{time_now}_{suffix}")


def format_storing_pathes_from_run_path(
    run_path: str, set_name: str, model: str, exp_name: str
) -> tuple[Path, Path, Path]:
    path_to_results = run_path / f"{set_name}_{model}_{exp_name}_results.csv"
    path_to_stats = run_path / f"{set_name}_{model}_{exp_name}_stats.txt"
    path_to_diagrams = run_path / f"{set_name}_{model}_{exp_name}.png"
    return path_to_results, path_to_stats, path_to_diagrams
