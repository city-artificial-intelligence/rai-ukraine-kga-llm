from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np
from tqdm import tqdm

from src.constants import LOGGER, BinaryOutputFormat, LLMCallOutput
from src.LLM_servers.openai import OpenAIServer
from src.onto_access import OntologyAccess
from src.onto_object import OntologyEntryAttr
from src.utils import calculate_logpropgs_confidence, retry


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
    source, target = candidates_pair
    try:
        try:
            prompt = prompt_function(OntologyEntryAttr(source, onto_src), OntologyEntryAttr(target, onto_tgt))
        except AssertionError:
            prompt = prompt_function(OntologyEntryAttr(target, onto_src), OntologyEntryAttr(source, onto_tgt))

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


def samples_process(
    candidate_pairs: list[tuple[str]],
    llm_oracle: OpenAIServer,
    onto_src: OntologyAccess,
    onto_tgt: OntologyAccess,
    model: str,
    prompt_function: Callable,
) -> tuple[list[tuple], list[tuple[int, int]], list[float]]:
    """Process lines sequentially."""
    results = []
    tokens_usage = []
    confidences = []

    for line in tqdm(candidate_pairs, desc=f"Processing Lines {prompt_function.__name__}"):
        result, token_usage, confidence = process_sample(line, llm_oracle, onto_src, onto_tgt, model, prompt_function)
        results.append(result)
        tokens_usage.append(token_usage)
        confidences.append(confidence)

    return results, tokens_usage, confidences


def parallel_samples_process(
    candidate_pairs: list[tuple[str]],
    llm_oracle: OpenAIServer,
    onto_src: OntologyAccess,
    onto_tgt: OntologyAccess,
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
