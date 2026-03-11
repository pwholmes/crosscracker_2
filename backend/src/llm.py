"""
LLM Factory and public interface.

This module provides a factory that initializes the appropriate LLM provider
based on environment configuration, and exposes a backward-compatible LLM API.
"""

from collections.abc import Callable
import logging

from config import LLM_DEFAULT_CONFIDENCE, LLM_MAX_CANDIDATES, LLM_MAX_SEARCH_LEVEL, LLM_PROVIDER_NAME
from model import Entry, Candidate
from providers import OllamaProvider, ClaudeProvider
from llm_provider import LLMProvider

logger = logging.getLogger("src.llm")

# Module-level hook variable for testing/demo overrides
_generate_candidates_hook: Callable[[Entry, str, int], list[Candidate]] | None = None


class LLM:
    """
    Factory and public interface for crossword solver LLM interactions.
    
    Delegates to the configured provider (Ollama or Claude) based on LLM_PROVIDER env var.
    """
    
    MAX_SEARCH_LEVEL: int = LLM_MAX_SEARCH_LEVEL
    DEFAULT_CONFIDENCE: float = LLM_DEFAULT_CONFIDENCE
    MAX_CANDIDATES: list[int] = LLM_MAX_CANDIDATES

    _provider: LLMProvider | None = None

    @staticmethod
    def initialize() -> None:
        """
        Initialize the LLM provider based on environment configuration.
        
        LLM_PROVIDER env var controls which backend is used:
        - "ollama" (default): Use local Ollama instance
        - "claude": Use Anthropic Claude API
        """
        provider_name = LLM_PROVIDER_NAME
        
        if provider_name == "claude":
            logger.info("Initializing Claude provider")
            LLM._provider = ClaudeProvider()
        elif provider_name == "ollama":
            logger.info("Initializing Ollama provider")
            LLM._provider = OllamaProvider()
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}. Use 'ollama' or 'claude'.")

    @staticmethod
    def _ensure_initialized() -> LLMProvider:
        """Ensure provider is initialized, initializing if needed."""
        if LLM._provider is None:
            LLM.initialize()
        assert LLM._provider is not None, "Provider should be initialized"
        return LLM._provider

    @staticmethod
    def set_generate_candidates_hook(
            hook: Callable[[Entry, str, int], list[Candidate]] | None
    ) -> None:
        """
        Install or clear a custom candidate generation hook.
        
        When a hook is installed, generate_candidates() will call the hook instead
        of the real LLM logic. This is primarily used for:
        - Simulated puzzles with pre-defined candidate lists
        - Test fixtures that need deterministic behavior
        - UI demos that don't require an actual LLM connection
        
        Pass None to clear the hook and restore default (stub) behavior.
        """
        global _generate_candidates_hook
        _generate_candidates_hook = hook

    @staticmethod
    def generate_candidates(
            entry: Entry,
            pattern: str,
            search_level: int
    ) -> list[Candidate]:
        """
        Generate candidate answers for a crossword clue.
        
        If a hook is installed, uses the hook. Otherwise delegates to the active provider.
        
        :param entry: The crossword clue entry
        :param pattern: Pattern of known letters (e.g., "A..B.")
        :param search_level: How creative to be (0=conservative, higher=more speculative)
        :return: List of candidate answers sorted by confidence
        """
        if _generate_candidates_hook is not None:
            return _generate_candidates_hook(entry, pattern, search_level)

        provider = LLM._ensure_initialized()
        
        # Call provider's generation and scoring in sequence
        candidates = provider.provider_generate_candidates(entry, pattern, search_level)
        if not candidates:
            return []
        
        # Score the candidates
        scored_candidates = provider.provider_score_candidates(entry, candidates, search_level)

        # Debug print: show all candidates and their aggregate confidence levels
        logger.debug("[LLM FINAL RESPONSE]: " + ", ".join(f"{c.answer} ({c.confidence:.1f})" for c in scored_candidates))
        return scored_candidates

    @staticmethod
    def verify_answer(entry: Entry, answer: str) -> bool:
        """
        Verify if a given answer is plausible for a crossword clue.
        
        Used when a crossing clue fills in the final letter(s) of an entry.
        
        :param entry: The crossword entry
        :param answer: The answer to verify
        :return: True if plausible, False otherwise
        """
        provider = LLM._ensure_initialized()
        return provider.provider_verify_answer(entry, answer)
