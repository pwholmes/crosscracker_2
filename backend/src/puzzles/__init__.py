"""Puzzle package initialization.

Auto-imports all puzzle modules to trigger registration.
"""

from . import simple_9x9  # pyright: ignore[reportUnusedImport]
from . import simulated_9x9  # pyright: ignore[reportUnusedImport]
from . import NYT_2026_01_26_15x15 #pyright: ignore[reportUnusedImport]
from .registry import get_default_puzzle_id, list_puzzles, load_puzzle, register_puzzle

__all__ = ["get_default_puzzle_id", "list_puzzles", "load_puzzle", "register_puzzle"]

