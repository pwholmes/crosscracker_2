"""
Ollama LLM provider implementation.

Handles communication with local Ollama instances running LLMs like llama3.1:8b.
"""

import requests
import logging
import os
import math
from typing import Any, cast
from dotenv import load_dotenv

from config import (
    OLLAMA_CANDIDATE_GENERATION_TUNING_PARAMS,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_TOP_LOGPROBS,
)
from llm_provider import LLMProvider
from model import Entry, Candidate

logger = logging.getLogger("src.providers.ollama")


class OllamaProvider(LLMProvider):
    """
    LLM provider implementation for local Ollama instances.
    """
    
    def __init__(self):
        """Initialize Ollama provider from environment variables."""
        load_dotenv()
        self.model_name: str = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
        self.url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
        logger.debug(f"Initialized Ollama provider: model={self.model_name}, url={self.url}")

    def _request_generation(self, prompt: str, search_level: int, entry: Entry) -> tuple[str, Any]:
        """Issue a generation request to Ollama and return response text plus raw payload."""
        logger.debug(f"[LLM GENERATE] Entry: {entry.entry_id} | Clue: '{entry.clue}' | Length: {entry.length} | Pattern: {entry.pattern} | Search level {search_level}")
        if entry.hints:
            hints_str = ", ".join(f"'{clue}'->{answer}" for clue, answer, _ in entry.hints)
            logger.debug(f"[LLM GENERATE HINTS] {hints_str}")

        tuning_params = cast(list[dict[str, float | int]], OLLAMA_CANDIDATE_GENERATION_TUNING_PARAMS)

        response = requests.post(
            self.url,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "logprobs": True,
                "top_logprobs": OLLAMA_TOP_LOGPROBS,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": tuning_params[search_level]
            },
            timeout=OLLAMA_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        response_text: str = result.get("response", "")
        return response_text, result

    def _request_scoring(self, prompt: str, search_level: int, entry: Entry) -> str:
        """Issue a scoring request to Ollama and return raw text output."""
        response = requests.post(
            self.url,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "keep_alive": OLLAMA_KEEP_ALIVE
            },
            timeout=OLLAMA_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        output = result.get("response", "")
        return str(output)

    def _request_verify(self, prompt: str, entry: Entry) -> str:
        """Issue a verification request to Ollama and return raw text output."""
        logger.debug(f"[LLM VERIFY] Clue: '{entry.clue}' | Answer length: {entry.length}")
        response = requests.post(
            self.url,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "keep_alive": OLLAMA_KEEP_ALIVE
            },
            timeout=OLLAMA_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        output = result.get("response", "")
        return str(output)

    def _include_pattern_in_scoring_task(self) -> bool:
        """Ollama scoring prompt historically includes PATTERN in the task line."""
        return True

    def _add_provider_candidates(
        self,
        entry: Entry,
        candidates: dict[str, Candidate],
        raw_response: Any,
        search_level: int
    ) -> None:
        """Add logprob-derived candidates from Ollama generation output."""
        logprobs: list[dict[str, Any]] = []
        if isinstance(raw_response, dict):
            typed_response = cast(dict[str, Any], raw_response)
            raw_logprobs = typed_response.get("logprobs", [])
            if isinstance(raw_logprobs, list):
                for item in cast(list[Any], raw_logprobs):
                    if isinstance(item, dict):
                        logprobs.append(cast(dict[str, Any], item))
        logprob_results: list[tuple[str, float]] = self._aggregate_logprobs(logprobs, entry.length)

        formatted_output = ", ".join([f"{word} ({int(confidence)})" for word, confidence in logprob_results])
        logger.debug(f"[LOGPROBS RESULT] {formatted_output}")

        for answer, confidence in logprob_results:
            if len(answer) != entry.length:
                continue
            if not answer.isalpha():
                continue
            candidates[answer] = Candidate(
                entry_id=entry.entry_id,
                answer=answer,
                logprob_confidence=confidence
            )

    @staticmethod
    def _aggregate_logprobs(
        logprobs_data: list[dict[str, Any]], 
        target_length: int,
        min_confidence: float = 0.0
    ) -> list[tuple[str, float]]:
        """Aggregate token-level log probabilities into word-level confidence scores."""
        word_probs: dict[str, float] = {} 
        current_tokens: list[str] = []
        current_logprob_sum: float = 0.0

        # Sentinel forces the loop to process the final word buffer
        sentinel: dict[str, Any] = {"token": "\n", "logprob": 0.0}
        rejected_words: list[str] = []
        for entry in logprobs_data + [sentinel]:
            token: str = str(entry.get("token", ""))
            logprob_val: Any = entry.get("logprob", 0.0)
            logprob: float = float(logprob_val) if isinstance(logprob_val, (int, float)) else 0.0

            is_delimiter = "\n" in token or "\\n" in token or token.strip() in {",", ";", ""}
            
            if is_delimiter:
                if current_tokens:
                    word = "".join(current_tokens).upper()
                    word = "".join(filter(str.isalpha, word))

                    if len(word) == target_length:
                        word_probs[word] = word_probs.get(word, 0.0) + math.exp(current_logprob_sum)
                    else:
                        rejected_words.append(word)

                current_tokens = []
                current_logprob_sum = 0.0
            else:
                if len("".join(current_tokens)) < target_length:
                    current_tokens.append(token.strip())
                    current_logprob_sum += logprob

        if rejected_words:
            logger.debug(f"[LOGPROBS] Rejecting words of wrong length: {', '.join(w for w in rejected_words)}")
                
        # Cap at 1.0 (100%) and apply min_confidence filter
        final_results = [
            (word, min(1.0, prob) * 100)
            for word, prob in word_probs.items()
            if (min(1.0, prob) * 100) >= min_confidence
        ]
        final_results = [(word, round(conf, 1)) for word, conf in final_results]

        return sorted(final_results, key=lambda x: x[1], reverse=True)
