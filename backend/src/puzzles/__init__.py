"""Puzzle package initialization.

Auto-imports all puzzle modules to trigger registration.
"""

from puzzles import crosshare_daybreak_15_9x9  # pyright: ignore[reportUnusedImport]
from puzzles import simulated_9x9  # pyright: ignore[reportUnusedImport]
from puzzles import nyt_2026_01_26_15x15 #pyright: ignore[reportUnusedImport]
from puzzles import nyt_2025_09_22_15x15 #pyright: ignore[reportUnusedImport]
from puzzles.registry import get_default_puzzle_id, list_puzzles, load_puzzle, register_puzzle

__all__ = ["get_default_puzzle_id", "list_puzzles", "load_puzzle", "register_puzzle"]

