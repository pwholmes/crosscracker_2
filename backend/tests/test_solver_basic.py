# backend/tests/test_solver_basic.py

from unittest.mock import patch

from src.model import Candidate, Cell, Entry, Grid
from src.solver import Solver
from src.puzzles.crosshare_daybreak_15_9x9 import create_grid

def test_solver_can_start():
    grid = create_grid()
    
    def mock_generate(
        entry: Entry,
        widening_level: int,
        max_candidates: int | None = None,
    ) -> list[Candidate]:
        return [Candidate(answer=entry.correct_answer, confidence=1.0)]
    
    def mock_verify(clue: str, answer: str) -> bool:
        """Verify answer by checking against correct answer in grid."""
        for entry in grid.entries.values():
            if entry.clue == clue:
                return answer.upper() == entry.correct_answer.upper()
        return False

    with patch("solver.LLM.generate_candidates", side_effect=mock_generate), \
         patch("solver.LLM.verify_answer", side_effect=mock_verify):
        solver = Solver(grid)
        ev = solver.step()
        assert ev["event"] in {"placed", "solved", "placed_fallback"}


def _create_small_grid() -> Grid:
    cells = [[Cell(row=r, col=c) for c in range(2)] for r in range(3)]
    entries: dict[str, Entry] = {
        "1A": Entry("1A", "Top across", "AB", cells, (0, 0), 2),
        "1D": Entry("1D", "Left down", "AC", cells, (0, 0), 2),
        "2D": Entry("2D", "Right down", "BD", cells, (0, 1), 2),
        "3A": Entry("3A", "Bottom across", "EF", cells, (2, 0), 2),
    }
    return Grid(entries)


def test_verify_entries_checks_crossings_only():
    grid = _create_small_grid()
    solver = Solver(grid, defer_candidate_init=True)

    # Pre-fill crossing entries and an unrelated entry.
    grid.place_candidate(Candidate("1D", "AC", search_level=0))
    grid.place_candidate(Candidate("2D", "BD", search_level=0))
    grid.place_candidate(Candidate("3A", "EF", search_level=0))

    crossing_ids = solver.get_crossing_entry_ids("1A")
    assert crossing_ids == {"1D", "2D"}

    verified_ids: list[str] = []

    def mock_verify(entry: Entry, answer: str) -> bool:
        verified_ids.append(entry.entry_id)
        return True

    # Prepare a dict of entry_id -> current pattern (answer) for crossing entries
    answers = {eid: grid.entries[eid].pattern for eid in crossing_ids}

    with patch("solver.LLM.verify_answer", side_effect=mock_verify):
        verified_entry_ids, failed_entry_ids = solver.verify_answers(answers)

    assert set(verified_entry_ids) == {"1D", "2D"}
    assert failed_entry_ids == []
    assert grid.entries["3A"].verified is False
    assert set(verified_ids) == {"1D", "2D"}
