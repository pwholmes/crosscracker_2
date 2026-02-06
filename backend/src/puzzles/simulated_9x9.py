"""Simulated puzzles for UI demos.

These puzzles reuse the 9x9 grid but provide a generate_candidates hook that
returns scored candidates deterministically, enabling repeatable UI replays.
"""

from __future__ import annotations

from __future__ import annotations

from typing import Dict, List, Tuple, TypedDict

from .simple_9x9 import create_grid as create_grid_9x9
from .registry import GenerateCandidatesHook, register_puzzle
from model import Grid, Entry, ScoredCandidate


class CandidateItem(TypedDict):
    answer: str
    confidence: float


def _shift_letters(answer: str) -> str:
    answer = answer.upper()
    out: list[str] = []
    for ch in answer:
        if "A" <= ch <= "Z":
            out.append(chr(((ord(ch) - 65 + 1) % 26) + 65))
        else:
            out.append("X")
    wrong = "".join(out)
    if wrong == answer:
        wrong = "Z" * len(answer)
    return wrong


def _make_hook(backtrack: bool) -> GenerateCandidatesHook:
    grid = create_grid_9x9()

    # clue -> (init, refined)
    by_clue: Dict[str, Tuple[List[CandidateItem], List[CandidateItem]]] = {}

    entry_ids = sorted(grid.entries.keys())
    across_ids = [eid for eid in entry_ids if eid.endswith("A")]
    down_ids = [eid for eid in entry_ids if eid.endswith("D")]
    wrong_top_ids: set[str]
    if backtrack:
        wrong_top_ids = set((across_ids[:3] + down_ids[:3])[:6])
    else:
        wrong_top_ids = set()

    base_conf = 0.92
    step = 0.02

    for i, (eid, entry) in enumerate(sorted(grid.entries.items(), key=lambda kv: kv[0])):
        correct = entry.correct_answer.upper()
        wrong1 = _shift_letters(correct)
        wrong2 = ("Z" * entry.length) if ("Z" * entry.length) != correct else ("Y" * entry.length)
        top_conf = max(0.15, base_conf - (i * step))

        if eid in wrong_top_ids:
            init: List[CandidateItem] = [
                {"answer": wrong1, "confidence": 0.99},
                {"answer": correct, "confidence": 0.90},
                {"answer": wrong2, "confidence": 0.10},
            ]
        else:
            init = [
                {"answer": correct, "confidence": float(top_conf)},
                {"answer": wrong1, "confidence": float(max(0.05, top_conf - 0.35))},
                {"answer": wrong2, "confidence": 0.05},
            ]

        refined: List[CandidateItem] = [
            {"answer": correct, "confidence": float(min(0.99, max(top_conf, 0.8)))},
            {"answer": wrong1, "confidence": 0.10},
            {"answer": wrong2, "confidence": 0.05},
        ]

        by_clue[entry.clue] = (init, refined)

    def hook(entry: Entry, widening_level: int, max_candidates: int) -> list[ScoredCandidate]:
        cands = by_clue.get(entry.clue)
        if cands is None:
            return []

        empty_pattern = set(entry.pattern) == {"."}
        use_init = widening_level == 0 and empty_pattern
        chosen: List[CandidateItem] = cands[0] if use_init else cands[1]
        selected: List[CandidateItem] = chosen[:max_candidates]
        return [ScoredCandidate(answer=item["answer"], confidence=item["confidence"]) for item in selected]

    return hook


def create_grid() -> Grid:
    return create_grid_9x9()


# Register two simulated puzzle variants.
register_puzzle(
    "9x9-sim-easy",
    create_grid,
    title="9x9 (simulated, easy)",
    generate_candidates_hook=_make_hook(backtrack=False),
)

register_puzzle(
    "9x9-sim-backtrack",
    create_grid,
    title="9x9 (simulated, backtracking)",
    generate_candidates_hook=_make_hook(backtrack=True),
)
