"""Full-puzzle candidate ranking/backtracking behavior.

This test simulates an LLM that returns scored candidates for *every* clue.
- Every entry's initial (width=0, empty pattern) candidate list includes the correct answer.
- A few entries intentionally rank an incorrect answer above the correct answer.
- Confidences vary across entries.

The solver should still reach the fully-correct filled grid, exercising both
candidate selection and heuristic backtracking.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, TypedDict
from unittest.mock import patch

from src.puzzles.simple_9x9 import create_grid
from src.solver import Solver
from src.model import Entry, ScoredCandidate


class CandidateItem(TypedDict):
    answer: str
    confidence: float


def _shift_letters(answer: str) -> str:
    """Deterministically create a wrong answer with the same length."""
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


def test_full_puzzle_scored_candidates_exercises_backtracking():
    grid = create_grid()

    # Choose a few entries (across + down) where the WRONG answer outranks the correct
    # answer in the initial (width=0, empty pattern) generation.
    entry_ids = sorted(grid.entries.keys())
    across_ids = [eid for eid in entry_ids if eid.endswith("A")]
    down_ids = [eid for eid in entry_ids if eid.endswith("D")]
    wrong_top_ids = set((across_ids[:3] + down_ids[:3])[:6])

    # Build a per-clue response table:
    #   clue -> (initial_candidates, refined_candidates)
    by_clue: Dict[str, Tuple[List[CandidateItem], List[CandidateItem]]] = {}

    # Provide a range of top confidences across entries.
    base_conf = 0.92
    step = 0.02

    for i, (eid, entry) in enumerate(sorted(grid.entries.items(), key=lambda kv: kv[0])):
        correct = entry.correct_answer.upper()
        assert len(correct) == entry.length

        wrong1 = _shift_letters(correct)
        wrong2 = ("Z" * entry.length) if ("Z" * entry.length) != correct else ("Y" * entry.length)

        # Variety across entries.
        top_conf = max(0.15, base_conf - (i * step))

        # Initial pass (width=0, empty pattern) candidates.
        if eid in wrong_top_ids:
            # Wrong is top and very confident so it gets placed early.
            # Correct is still present.
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

        # Widened / pattern-aware generation should "improve": put correct on top.
        refined: List[CandidateItem] = [
            {"answer": correct, "confidence": float(min(0.99, max(top_conf, 0.8)))},
            {"answer": wrong1, "confidence": 0.10},
            {"answer": wrong2, "confidence": 0.05},
        ]

        by_clue[entry.clue] = (init, refined)

    def mock_generate(entry: Entry, widening_level: int, max_candidates: int | None = None) -> list[ScoredCandidate]:
        if max_candidates is None:
            max_candidates = 5
        cands = by_clue.get(entry.clue)
        assert cands is not None, f"Unexpected clue text: {entry.clue!r}"

        empty_pattern = set(entry.pattern) == {"."}
        use_init = widening_level == 0 and empty_pattern
        chosen: List[CandidateItem] = cands[0] if use_init else cands[1]

        out_dicts: List[CandidateItem] = chosen[:max_candidates]
        # Always return correct-length answers.
        assert all(len(item["answer"]) == len(out_dicts[0]["answer"]) for item in out_dicts)
        
        # Convert to ScoredCandidate objects
        return [ScoredCandidate(answer=item["answer"], confidence=item["confidence"]) for item in out_dicts]

    def mock_verify(clue: str, answer: str) -> bool:
        """Verify answer by checking against correct answer in grid."""
        for entry in grid.entries.values():
            if entry.clue == clue:
                return answer.upper() == entry.correct_answer.upper()
        return False
    
    # Patch the function as imported into the solver module.
    with patch("src.solver.LLM.generate_candidates", side_effect=mock_generate), \
         patch("src.solver.LLM.verify_answer", side_effect=mock_verify):
        solver = Solver(grid)

        events: list[str] = []
        # Drive step-by-step so we can assert we saw backtracking.
        for _ in range(5000):
            ev = solver.step()
            events.append(ev.get("event", ""))
            if ev.get("event") == "solved":
                break
            if ev.get("event") == "failed":
                raise AssertionError(f"Solver failed unexpectedly: {ev}")
        else:
            raise AssertionError("Solver did not terminate within step limit")

    assert "backtrack" in events or "placed_fallback" in events, "Expected at least one heuristic backtrack or fallback"

    # Ensure we end with the fully-correct grid.
    for eid, entry in grid.entries.items():
            assert entry.pattern == entry.correct_answer, f"Entry {eid} solved incorrectly"
