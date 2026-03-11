"""Puzzle package initialization.

Auto-discovers and imports all puzzle modules to trigger registration.
"""

import importlib
import pkgutil
from pathlib import Path

from puzzles.registry import get_default_puzzle_id, list_puzzles, load_puzzle, register_puzzle

# Auto-discover and import all puzzle modules in this package
_puzzles_dir = Path(__file__).parent
for module_info in pkgutil.iter_modules([str(_puzzles_dir)]):
    # Skip the registry module and any utility/private modules
    if module_info.name in ("registry", "py.typed"):
        continue
    try:
        importlib.import_module(f"puzzles.{module_info.name}")  # pyright: ignore[reportUnusedImport]
    except Exception as e:
        print(f"Warning: Failed to import puzzle module {module_info.name}: {e}")

__all__ = ["get_default_puzzle_id", "list_puzzles", "load_puzzle", "register_puzzle"]

