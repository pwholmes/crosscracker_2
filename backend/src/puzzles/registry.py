"""Puzzle registry.

Stores puzzle factories and optional simulation hooks.

Simulated puzzles can provide a `generate_candidates` hook that the server can
install into `llm_interface` at load time, allowing UI-driven demo/fixtures.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from model import Grid, Entry


GenerateCandidatesHook = Callable[[Entry, int], list[Any]]


@dataclass(frozen=True)
class PuzzleSpec:
    puzzle_id: str
    title: str
    factory: Callable[[], Grid]
    generate_candidates_hook: GenerateCandidatesHook | None = None


_PUZZLES: dict[str, PuzzleSpec] = {}
_default_puzzle_id: str | None = None


def register_puzzle(
    puzzle_id: str,
    factory: Callable[[], Grid],
    *,
    title: str | None = None,
    generate_candidates_hook: GenerateCandidatesHook | None = None,
    default: bool = False,
) -> None:
    global _default_puzzle_id
    _PUZZLES[puzzle_id] = PuzzleSpec(
        puzzle_id=puzzle_id,
        title=title or puzzle_id,
        factory=factory,
        generate_candidates_hook=generate_candidates_hook,
    )
    if default or _default_puzzle_id is None:
        _default_puzzle_id = puzzle_id


def list_puzzles() -> list[dict[str, str]]:
    specs = sorted(_PUZZLES.values(), key=lambda s: s.puzzle_id)
    return [{"id": s.puzzle_id, "title": s.title} for s in specs]


def get_default_puzzle_id() -> str:
    if _default_puzzle_id is None:
        raise RuntimeError("No puzzles registered")
    return _default_puzzle_id


def load_puzzle(puzzle_id: str) -> tuple[Grid, GenerateCandidatesHook | None]:
    spec = _PUZZLES[puzzle_id]
    return spec.factory(), spec.generate_candidates_hook


def get_puzzle_spec(puzzle_id: str) -> PuzzleSpec | None:
    """Return the PuzzleSpec for a given puzzle_id, or None if not found."""
    return _PUZZLES.get(puzzle_id)