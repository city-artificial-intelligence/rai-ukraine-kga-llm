from __future__ import annotations

import os
from typing import Any

import openai
from pydantic import BaseModel

from src.constants import BinaryOutputFormat, LLMCallOutput, TokensUsage


class OpenAIServer:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, **kwargs: dict[str, Any]) -> None:
        """Initialize the server with an OpenAI API key and optional parameters.

        Parameters
        ----------
        api_key (str | None): API key. If not provided, it will be fetched from the environment variable OPENAI_API_KEY.
        base_url (str | None): Base URL for the API. If not provided, it will be None and used the default OpenAI URL.
        **kwargs (Any): Optional keyword arguments to set additional attributes.

        Attributes
        ----------
        base_url (str | None): Base URL for the API. (by default - OpenAI URL, but can be a custom URL, e.g. for Gemini)
        client (OpenAI): Instance of the OpenAI client.
        chat_context (list): List to store the chat context.
        system_prompt_text (str or None): Text for the system prompt.
        response_format (type): Format of the response, default is BinaryOutputFormat.
        top_p (float): Top-p sampling parameter, default is 1.0.
        temperature (float): Sampling temperature, default is 0.5.
        max_tokens (int): Maximum number of tokens in the response, default is 100.
        logprobs (bool): Whether to include log probabilities in the response, default is True.
        top_logprobs (int): Number of top log probabilities to include, default is 3.

        """
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        self.base_url = base_url

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.chat_context = []

        self.response_format = BinaryOutputFormat
        self.top_p = 0.3
        self.temperature = 0
        self.max_tokens = 100
        self.logprobs = True
        self.top_logprobs = 3

        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_system_context(self, message: str) -> None:
        """Add or update the system context in the chat history.

        If a system message exists, it will be updated; otherwise, it will be added.
        """
        if len(self.chat_context) == 0 or self.chat_context[0]["role"] != "system":
            self.chat_context.insert(0, self.wrap_text_message(message, "system"))
        else:
            self.chat_context[0] = self.wrap_text_message(message, "system")

    def add_context(self, message: str, role: str) -> None:
        """Add a message to the chat context."""
        self.chat_context.append(self.wrap_text_message(message, role))

    def set_response_format(self, response_format: BaseModel | dict) -> None:
        """Set the response format for the server."""
        self.response_format = response_format

    def wrap_text_message(self, message: str, role: str) -> dict:
        """Wrap a text message with a role."""
        return {"role": role, "content": message}

    def ask_sync_question(self, message: str, model: str = "gpt-4o-mini") -> LLMCallOutput:
        """Send a user query to the OpenAI model and optionally update the chat context.

        Args:
            message (str): The user message to process.
            model (str): The model to use for the query.

        Returns:
            LLMServerOutput: Wrapper for the response message, usage, logprobs, and parsed output.

        Raises:
            Exception: If an error occurs during the query.

        """
        try:
            messages = [*self.chat_context, self.wrap_text_message(message, "user")]

            inference_kwargs = {
                "model": model,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "response_format": self.response_format,
            }
            if self.base_url is None:
                inference_kwargs["max_tokens"] = self.max_tokens
                inference_kwargs["logprobs"] = self.logprobs
                inference_kwargs["top_logprsobs"] = self.top_logprobs

            response = self.client.beta.chat.completions.parse(**inference_kwargs)

            output_message = response.choices[0].message.content
            usage = TokensUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
            try:
                logprobs = response.choices[0].logprobs.model_dump()["content"]
            except AttributeError:
                logprobs = []
            parsed_output = response.choices[0].message.parsed
            return LLMCallOutput(message=output_message, usage=usage, logprobs=logprobs, parsed=parsed_output)

        except Exception as e:
            raise e

    def clear_context(self) -> None:
        """Clear the chat context."""
        self.chat_context = []
