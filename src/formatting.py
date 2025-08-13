from __future__ import annotations

import time
from pathlib import Path

from config.config import DATA_DIR, OUTPUTS_DIR, RUN_DIR


def format_subsets_ontologies_paths(dataset_name: str, set_name: str) -> tuple[Path, Path]:
    onto_data_dir = DATA_DIR / dataset_name / set_name
    source_ontology, target_ontology = set_name.split("-")

    if dataset_name == "anatomy":
        target_filename = f"{source_ontology}.owl"
        source_filename = f"{target_ontology}.owl"

    if dataset_name == "bioml-2024":
        if "." in target_ontology:
            suffix = target_ontology.split(".")[-1]
            source_ontology = f"{source_ontology}.{suffix}"

        source_filename = f"{source_ontology}.owl"
        target_filename = f"{target_ontology}.owl"

    if dataset_name == "largebio":
        onto_data_dir = DATA_DIR / dataset_name
        source_filename = f"oaei_{source_ontology.upper()}_whole_ontology.owl"
        target_filename = f"oaei_{target_ontology.upper()}_whole_ontology.owl"

    if dataset_name == "largebio_small":
        source_filename = f"oaei_{source_ontology.upper()}_small_overlapping_{target_ontology}.owl"
        target_filename = f"oaei_{target_ontology.upper()}_small_overlapping_{source_ontology}.owl"

    return onto_data_dir / source_filename, onto_data_dir / target_filename


def format_oracle_pairs_filepath(dataset_name: str, set_name: str) -> Path:
    return DATA_DIR / dataset_name / set_name / f"{dataset_name}-{set_name}-oasystem_mappings_to_ask_oracle_user_llm.txt"


def format_oracle_pairs_precomputed_dir(dataset_name: str, set_name: str, suffix: str = "") -> Path:
    return DATA_DIR / dataset_name / set_name / f"oracle_pairs{suffix}"


def format_oracle_pairs_with_prompts_path(
    prompt_function: callable | None, oracle_pairs_dir: Path, suffix: str = ""
) -> Path:
    if prompt_function is None:
        return oracle_pairs_dir / f"oasystem_mappings{suffix}.csv"
    return oracle_pairs_dir / f"oasystem_mappings-{prompt_function.__name__}{suffix}.csv"


def format_run_path(suffix: str = "") -> Path:
    date = time.strftime("%Y-%m-%d")
    time_now = time.strftime("%H-%M-%S")
    return RUN_DIR / (f"{date}_{time_now}" if not suffix else f"{date}_{time_now}{suffix}")


def format_predictions_run_path(dataset: str, set_name: str, model: str, exp_type: str, exp_spec: str = "") -> Path:
    return OUTPUTS_DIR / dataset / set_name / model / exp_type / exp_spec


def format_storing_pathes_from_run_path(
    run_path: str, set_name: str, model: str, prompt_name: str, suffix: str = ""
) -> tuple[Path, Path, Path]:
    path_to_results = run_path / f"{set_name}_{model}_{prompt_name}_results{suffix}.csv"
    path_to_stats = run_path / f"{set_name}_{model}_{prompt_name}_stats{suffix}.txt"
    path_to_diagrams = run_path / f"{set_name}_{model}_{prompt_name}{suffix}.png"
    return path_to_results, path_to_stats, path_to_diagrams


def format_gt_pairs_filepath(dataset_path: str, set_name: str) -> Path:
    return DATA_DIR / dataset_path / set_name / "refs_equiv/full.tsv"


def format_run_metrics_path(save_dir: Path, suffix: str = "") -> Path:
    return save_dir / f"results{suffix}.csv"


def format_combined_metrics_path(suffix: str = "") -> Path:
    return RUN_DIR / f"all_runs_metrics{suffix}.csv"
