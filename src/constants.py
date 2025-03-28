from __future__ import annotations

import logging

from pydantic import BaseModel

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class BinaryOutputFormat(BaseModel):
    answer: bool


class BinaryOutputFormatWithReasoning(BaseModel):
    answer: bool
    reasoning: str


class TokensUsage(BaseModel):
    input_tokens: int | None
    output_tokens: int | None


class LLMCallOutput(BaseModel):
    message: str
    usage: TokensUsage
    logprobs: list | None
    parsed: BaseModel | None


PAIRS_SEPARATOR = "|"
GT_COL_DIVIDER = "\t"
RESULTS_SEPARATOR = ","
