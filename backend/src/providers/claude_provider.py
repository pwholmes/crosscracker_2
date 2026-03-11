"""
Claude LLM provider implementation via Anthropic API.

Handles communication with Claude models through the Anthropic SDK.
Since Claude doesn't expose logprobs like Ollama, we rely on LLM-scored confidence only.
"""

import logging
import os
from typing import Any
from dotenv import load_dotenv
from llm_provider import LLMProvider
from model import Entry
from anthropic import Anthropic, AsyncAnthropic


logger = logging.getLogger("src.providers.claude")


def _extract_text_from_response(response: Any) -> str:
    """Extract text content from Anthropic API response message.
    
    Handles the various content block types that the API may return.
    """
    if not response.content or len(response.content) == 0:
        return ""
    
    content_block = response.content[0]
    
    # Check if it's a TextBlock with text attribute
    if hasattr(content_block, 'text'):
        text_attr = getattr(content_block, 'text', None)
        if isinstance(text_attr, str):
            return text_attr
    
    # Check if it's a dict-like object  
    if isinstance(content_block, dict):
        # Type-safe extraction from dict
        text_value = content_block.get("text")  # type: ignore
        if isinstance(text_value, str):
            return text_value
    
    return ""


class ClaudeProvider(LLMProvider):
    """
    LLM provider implementation for Anthropic Claude models.
    """

    # Tune generation creativity by search level (0=conservative, 2=exploratory)
    CANDIDATE_GENERATION_TUNING_PARAMS: list[float] = [0.25, 0.70, 1.00]
    
    def __init__(self):
        """Initialize Claude provider from environment variables."""
        load_dotenv()
        api_key = os.environ.get("CLAUDE_API_KEY")
        if not api_key:
            raise ValueError("CLAUDE_API_KEY environment variable is not set")
        
        self.model_name: str = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5")
        self.client = Anthropic(api_key=api_key)
        self.async_client = AsyncAnthropic(api_key=api_key)
        logger.debug(f"Initialized Claude provider: model={self.model_name}")


    def _request_generation(self, prompt: str, search_level: int, entry: Entry) -> tuple[str, Any]:
        """Issue a generation request to Claude and return response text plus raw payload."""
        logger.debug(f"[LLM GENERATE] Entry: {entry.entry_id} | Clue: '{entry.clue}' | Length: {entry.length} | Pattern: {entry.pattern} | Search level {search_level}")
        if entry.hints:
            hints_str = ", ".join(f"'{clue}'->{answer}" for clue, answer, _ in entry.hints)
            logger.debug(f"[LLM GENERATE HINTS] {hints_str}")

        # Mirror Ollama behavior: generation creativity scales by search level.
        clamped_level = max(0, min(search_level, len(self.CANDIDATE_GENERATION_TUNING_PARAMS) - 1))
        temperature = self.CANDIDATE_GENERATION_TUNING_PARAMS[clamped_level]

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        response_text = _extract_text_from_response(response)
        return response_text, response

    def _request_scoring(self, prompt: str, search_level: int, entry: Entry) -> str:
        """Issue a scoring request to Claude and return raw text output."""
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return _extract_text_from_response(response)

    def _request_verify(self, prompt: str, entry: Entry) -> str:
        """Issue a verification request to Claude and return raw text output."""
        logger.debug(f"[LLM VERIFY] Clue: '{entry.clue}' | Length: {entry.length}")
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=10,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return _extract_text_from_response(response)

