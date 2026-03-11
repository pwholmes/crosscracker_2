from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


def _get_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _get_json(name: str, default: Any) -> Any:
    value = os.getenv(name)
    if value is None:
        return default
    return json.loads(value)


class ClaudeGenerationTuning(TypedDict):
    temperature: float
    top_k: int


class OllamaGenerationTuning(TypedDict):
    temperature: float
    top_p: float
    top_k: int


MODEL_LLM_CONFIDENCE_WEIGHT = _get_float("MODEL_LLM_CONFIDENCE_WEIGHT", 0.6)
MODEL_BACKTRACK_PENALTY = _get_int("MODEL_BACKTRACK_PENALTY", 25)
MODEL_VERIFICATION_PENALTY = _get_int("MODEL_VERIFICATION_PENALTY", 5)
MODEL_MIN_SELECTABLE_CONFIDENCE = _get_int("MODEL_MIN_SELECTABLE_CONFIDENCE", 25)

HEURISTICS_SELECTION_CONFIDENCE_WEIGHT = _get_float("HEURISTICS_SELECTION_CONFIDENCE_WEIGHT", 0.25)
HEURISTICS_SELECTION_LENGTH_WEIGHT = _get_float("HEURISTICS_SELECTION_LENGTH_WEIGHT", 0.2)
HEURISTICS_SELECTION_COMPLETENESS_WEIGHT = _get_float("HEURISTICS_SELECTION_COMPLETENESS_WEIGHT", 0.35)
HEURISTICS_SELECTION_CONSTRAINT_WEIGHT = _get_float("HEURISTICS_SELECTION_CONSTRAINT_WEIGHT", 0.2)
HEURISTICS_FALLBACK_UNFILLED_RATIO_WEIGHT = _get_float("HEURISTICS_FALLBACK_UNFILLED_RATIO_WEIGHT", 0.5)

LLM_PROVIDER_NAME = _get_str("LLM_PROVIDER", "ollama").lower()
LLM_MAX_SEARCH_LEVEL = _get_int("LLM_MAX_SEARCH_LEVEL", 2)
LLM_DEFAULT_CONFIDENCE = _get_float("LLM_DEFAULT_CONFIDENCE", 30.0)
LLM_MAX_CANDIDATES = _get_json("LLM_MAX_CANDIDATES", [7, 12, 15])

OLLAMA_GENERATION_LEVEL_0_TEMPERATURE = _get_float("OLLAMA_GENERATION_LEVEL_0_TEMPERATURE", 0.25)
OLLAMA_GENERATION_LEVEL_0_TOP_P = _get_float("OLLAMA_GENERATION_LEVEL_0_TOP_P", 0.8)
OLLAMA_GENERATION_LEVEL_0_TOP_K = _get_int("OLLAMA_GENERATION_LEVEL_0_TOP_K", 10)
OLLAMA_GENERATION_LEVEL_1_TEMPERATURE = _get_float("OLLAMA_GENERATION_LEVEL_1_TEMPERATURE", 0.70)
OLLAMA_GENERATION_LEVEL_1_TOP_P = _get_float("OLLAMA_GENERATION_LEVEL_1_TOP_P", 0.9)
OLLAMA_GENERATION_LEVEL_1_TOP_K = _get_int("OLLAMA_GENERATION_LEVEL_1_TOP_K", 30)
OLLAMA_GENERATION_LEVEL_2_TEMPERATURE = _get_float("OLLAMA_GENERATION_LEVEL_2_TEMPERATURE", 1.00)
OLLAMA_GENERATION_LEVEL_2_TOP_P = _get_float("OLLAMA_GENERATION_LEVEL_2_TOP_P", 0.95)
OLLAMA_GENERATION_LEVEL_2_TOP_K = _get_int("OLLAMA_GENERATION_LEVEL_2_TOP_K", 60)
OLLAMA_CANDIDATE_GENERATION_TUNING_PARAMS: list[OllamaGenerationTuning] = [
    {
        "temperature": OLLAMA_GENERATION_LEVEL_0_TEMPERATURE,
        "top_p": OLLAMA_GENERATION_LEVEL_0_TOP_P,
        "top_k": OLLAMA_GENERATION_LEVEL_0_TOP_K,
    },
    {
        "temperature": OLLAMA_GENERATION_LEVEL_1_TEMPERATURE,
        "top_p": OLLAMA_GENERATION_LEVEL_1_TOP_P,
        "top_k": OLLAMA_GENERATION_LEVEL_1_TOP_K,
    },
    {
        "temperature": OLLAMA_GENERATION_LEVEL_2_TEMPERATURE,
        "top_p": OLLAMA_GENERATION_LEVEL_2_TOP_P,
        "top_k": OLLAMA_GENERATION_LEVEL_2_TOP_K,
    },
]
OLLAMA_TIMEOUT_SECONDS = _get_int("OLLAMA_TIMEOUT_SECONDS", 120)
OLLAMA_KEEP_ALIVE = _get_str("OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_TOP_LOGPROBS = _get_int("OLLAMA_TOP_LOGPROBS", 5)

CLAUDE_GENERATION_LEVEL_0_TEMPERATURE: float = _get_float("CLAUDE_GENERATION_LEVEL_0_TEMPERATURE", 0.25)
CLAUDE_GENERATION_LEVEL_0_TOP_K: int = _get_int("CLAUDE_GENERATION_LEVEL_0_TOP_K", 10)
CLAUDE_GENERATION_LEVEL_1_TEMPERATURE: float = _get_float("CLAUDE_GENERATION_LEVEL_1_TEMPERATURE", 0.70)
CLAUDE_GENERATION_LEVEL_1_TOP_K: int = _get_int("CLAUDE_GENERATION_LEVEL_1_TOP_K", 30)
CLAUDE_GENERATION_LEVEL_2_TEMPERATURE: float = _get_float("CLAUDE_GENERATION_LEVEL_2_TEMPERATURE", 1.00)
CLAUDE_GENERATION_LEVEL_2_TOP_K: int = _get_int("CLAUDE_GENERATION_LEVEL_2_TOP_K", 60)
CLAUDE_CANDIDATE_GENERATION_TUNING_PARAMS: list[ClaudeGenerationTuning] = [
    {
        "temperature": CLAUDE_GENERATION_LEVEL_0_TEMPERATURE,
        "top_k": CLAUDE_GENERATION_LEVEL_0_TOP_K,
    },
    {
        "temperature": CLAUDE_GENERATION_LEVEL_1_TEMPERATURE,
        "top_k": CLAUDE_GENERATION_LEVEL_1_TOP_K,
    },
    {
        "temperature": CLAUDE_GENERATION_LEVEL_2_TEMPERATURE,
        "top_k": CLAUDE_GENERATION_LEVEL_2_TOP_K,
    },
]

CHROMA_HOST = _get_str("CHROMA_HOST", "localhost")
CHROMA_PORT = _get_int("CHROMA_PORT", 8001)
CHROMA_COLLECTION_NAME = _get_str("CHROMA_COLLECTION_NAME", "crossword_1")
CHROMA_EXECUTION_PROVIDER = _get_str("CHROMA_EXECUTION_PROVIDER", "CUDAExecutionProvider")

SOLVER_MAX_BACKTRACKS_BEFORE_FALLBACK = _get_int("SOLVER_MAX_BACKTRACKS_BEFORE_FALLBACK", 3)

RECORDINGS_DIR = Path(_get_str("RECORDINGS_DIR", str(BASE_DIR / "backend" / "recordings")))
CHECKPOINTS_DIR = Path(_get_str("CHECKPOINTS_DIR", str(BASE_DIR / "backend" / "checkpoints")))
FRONTEND_DIR = Path(_get_str("FRONTEND_DIR", str(BASE_DIR / "frontend")))
SERVER_PLAY_INTERVAL_SECONDS = _get_float("SERVER_PLAY_INTERVAL_SECONDS", 0)