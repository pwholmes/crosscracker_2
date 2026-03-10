"""
Provider implementations for CrossCracker LLM backend.
"""

from .ollama_provider import OllamaProvider
from .claude_provider import ClaudeProvider

__all__ = ["OllamaProvider", "ClaudeProvider"]
