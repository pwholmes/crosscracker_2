"""Puzzle package initialization.

Auto-imports all puzzle modules to trigger registration.
"""

from . import puzzle_9x9  # pyright: ignore[reportUnusedImport]
from . import simulated_9x9  # pyright: ignore[reportUnusedImport]
from .registry import get_default_puzzle_id, list_puzzles, load_puzzle, register_puzzle

__all__ = ["get_default_puzzle_id", "list_puzzles", "load_puzzle", "register_puzzle"]

