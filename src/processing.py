from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.constants import LOGGER, PAIRS_SEPARATOR, BinaryOutputFormat, LLMCallOutput
from src.formatting import (
    format_oracle_pairs_filepath,
    format_oracle_pairs_precomputed_dir,
    format_oracle_pairs_with_prompts_path,
)
from src.LLM_servers.openai import OpenAIServer
from src.onto_access import OntologyAccess
from src.onto_object import OntologyEntryAttr
from src.utils import calculate_logpropgs_confidence, retry


def save_oracle_pairs_with_prompts(
    oracle_candidate_pairs_path: Path,
    src_onto_path: OntologyAccess,
    tgt_onto_path: OntologyAccess,
    prompt_functions: list[callable],
    save_dir: Path,
    sep: str,
    max_workers: int = 1,
) -> None:
    base_pairs_df = pd.read_csv(oracle_candidate_pairs_path, sep=sep, header=None)

    for row in tqdm(base_pairs_df.iterrows(), total=base_pairs_df.shape[0], desc="Adding prompts"):
        src_uri, tgt_uri = row[1][0], row[1][1]
        src_entity = OntologyEntryAttr(src_uri, src_onto_path)
        tgt_entity = OntologyEntryAttr(tgt_uri, tgt_onto_path)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(prompt_function, src_entity, tgt_entity): prompt_function
                for prompt_function in prompt_functions
            }
            for future in futures:
                prompt_function = futures[future]
                prompt = future.result()
                base_pairs_df.loc[row[0], prompt_function.__name__] = prompt

    save_dir.mkdir(parents=True, exist_ok=True)
    for prompt_function in [*prompt_functions, None]:
        new_file_path = format_oracle_pairs_with_prompts_path(prompt_function, save_dir)
        columns = base_pairs_df.columns.tolist()[:2] + ([prompt_function.__name__] if prompt_function else [])
        base_pairs_df[columns].to_csv(new_file_path, sep=sep, index=False, header=False)

    return base_pairs_df


def try_load_precomputed_oracle_pairs(
    dataset_name: str, set_name: str, prompt_function: callable | None, sep: str = PAIRS_SEPARATOR, suffix: str = ""
) -> list[tuple[str]]:
    oracle_pairs_dir = format_oracle_pairs_precomputed_dir(dataset_name, set_name)

    # Try path with prompts
    if prompt_function is not None:
        pairs_path = format_oracle_pairs_with_prompts_path(prompt_function, oracle_pairs_dir, suffix=suffix)
        if pairs_path.exists():
            pairs_df = pd.read_csv(pairs_path, sep=sep, header=None)
            return list(zip(pairs_df[0], pairs_df[1], pairs_df[2]))

    # Try path without prompts
    pairs_path = format_oracle_pairs_with_prompts_path(
        prompt_function=None, oracle_pairs_dir=oracle_pairs_dir, suffix=suffix
    )
    if not pairs_path.exists():
        # Fall back to legacy format
        pairs_path = format_oracle_pairs_filepath(dataset_name, set_name)
        if not pairs_path.exists():
            msg = f"Oracle pairs file not found: {pairs_path}"
            raise FileNotFoundError(msg)

    pairs_df = pd.read_csv(pairs_path, sep=sep, header=None)
    return list(zip(pairs_df[0], pairs_df[1]))


def extract_llm_compeletion_answer(response: LLMCallOutput) -> bool:
    """Extract the completion answer from the LLM response."""
    if isinstance(response.parsed, BinaryOutputFormat):
        return response.parsed.answer
    raise NotImplementedError()


@retry(max_retries=1)
def process_sample(
    candidates_pair: tuple[str],
    llm_oracle: OpenAIServer,
    onto_src: OntologyAccess,
    onto_tgt: OntologyAccess,
    model: str,
    prompt_function: Callable,
) -> tuple[list, tuple[int, int], float]:
    """Process a single line: generate a prompt, send it to the LLM, and extract the response.

    If an exception occurs, the function is retried once.

    Returns:
        tuple[list, tuple[int, int], float]: The results, token usage, and confidence.
        - results: list of source, target, answer, confidence
        - token_usage: tuple of input tokens, output tokens
        - confidence: float

    """
    source, target = candidates_pair[:2]

    if len(candidates_pair) == 2:
        prompt = prompt_function(OntologyEntryAttr(source, onto_src), OntologyEntryAttr(target, onto_tgt))
    else:
        prompt = candidates_pair[2]

    try:
        response: LLMCallOutput = llm_oracle.ask_sync_question(prompt, model)
        answer = extract_llm_compeletion_answer(response)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        confidence = calculate_logpropgs_confidence(response.logprobs)

        token_usage = (input_tokens, output_tokens)
        return [source, target, answer, confidence], token_usage, confidence

    except Exception as e:  # noqa: BLE001
        LOGGER.error(f"Error processing line: {candidates_pair}: {e}", exc_info=True)
        return [source, target, "error", str(np.nan)], (np.nan, np.nan), np.nan


def parallel_samples_process(
    candidate_pairs: list[tuple[str]],
    llm_oracle: OpenAIServer,
    onto_src: OntologyAccess | None,
    onto_tgt: OntologyAccess | None,
    model: str,
    max_workers: int,
    prompt_function: Callable,
) -> tuple[list[tuple], list[tuple[int, int]], list[float]]:
    """Process lines in parallel, with progress tracking using tqdm.

    Args:
        candidate_pairs (list[tuple[str]]): The list of candidate pairs to process.
        llm_oracle: The LLM oracle server.
        onto_src: The source ontology.
        onto_tgt The target ontology.
        model: The model to use.
        max_workers: The number of workers to use.
        prompt_function: The prompt function to use to generate the prompt, having the signature:
            prompt_function(src: OntologyEntryAttr, tgt: OntologyEntryAttr) -> str.

    Returns:
        tuple[list[tuple], list[tuple[int, int]], list[float]]:
            - results: list of tuples, each with source, target, answer, confidence
            - tokens_usage: list of tuples, each with input tokens, output tokens,
            - confidences: list of floats

    """
    results = []
    tokens_usage = []
    confidences = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_line = {
            executor.submit(process_sample, pair, llm_oracle, onto_src, onto_tgt, model, prompt_function): pair
            for pair in candidate_pairs
        }

        for future in tqdm(
            as_completed(future_to_line), total=len(future_to_line), desc=f"Processing Lines {prompt_function.__name__}"
        ):
            result, token_usage, confidence = future.result()
            results.append(result)
            tokens_usage.append(token_usage)
            confidences.append(confidence)

    return results, tokens_usage, confidences
