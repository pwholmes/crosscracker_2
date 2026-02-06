# backend/tests/test_solver_basic.py

from unittest.mock import patch

from src.model import Entry, ScoredCandidate
from src.solver import Solver
from src.puzzles.puzzle_9x9 import create_grid

def test_solver_can_start():
    grid = create_grid()
    
    def mock_generate(
        entry: Entry,
        widening_level: int,
        max_candidates: int | None = None,
    ) -> list[ScoredCandidate]:
        return [ScoredCandidate(answer=entry.correct_answer, confidence=1.0)]
    
    def mock_verify(clue: str, answer: str) -> bool:
        """Verify answer by checking against correct answer in grid."""
        for entry in grid.entries.values():
            if entry.clue == clue:
                return answer.upper() == entry.correct_answer.upper()
        return False

    with patch("src.llm.LLM.generate_candidates", side_effect=mock_generate), \
         patch("src.llm.LLM.verify_answer", side_effect=mock_verify):
        solver = Solver(grid)
        ev = solver.step()
        assert ev["event"] in {"placed", "solved", "placed_fallback"}
