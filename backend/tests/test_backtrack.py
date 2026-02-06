"""Test backtracking when bad candidates are provided."""

from __future__ import annotations

from typing import Dict, Tuple
from unittest.mock import patch
from src.solver import Solver
from src.puzzles.puzzle_9x9 import create_grid
from src.model import Entry, ScoredCandidate


def test_solver_backtracks_on_wrong_candidate():
    """Test that solver backtracks when given an incorrect candidate."""
    grid = create_grid()
    
    # Track how many times we've called generate for "1A"
    call_count: Dict[Tuple[str, str, int], int] = {}
    
    def mock_generate(entry: Entry, widening_level: int, max_candidates: int | None = None) -> list[ScoredCandidate]:
        # For "1A" (Bar that connects rotating wheels -> AXLE)
        if "connects rotating" in entry.clue:
            key = ("1A", entry.pattern, widening_level)
            call_count[key] = call_count.get(key, 0) + 1
            
            # On first call at widening_level 0, return a wrong answer
            if widening_level == 0 and call_count[key] == 1:
                # Return WRONG as first candidate, it won't match pattern or will fail verification
                return [
                    ScoredCandidate(answer="BEND", confidence=0.8),
                    ScoredCandidate(answer="AXLE", confidence=0.7),
                ]
            
        # Fallback: return empty list (stub behavior)
        return []
    
    def mock_verify(clue: str, answer: str) -> bool:
        """Verify answer by checking against correct answer in grid."""
        for entry in grid.entries.values():
            if entry.clue == clue:
                return answer.upper() == entry.correct_answer.upper()
        return False
    
    with patch('src.llm.LLM.generate_candidates', side_effect=mock_generate), \
         patch('src.llm.LLM.verify_answer', side_effect=mock_verify):
        solver = Solver(grid)
        result = solver.solve()
        # Should solve successfully despite the wrong candidate
        assert result is True, "Solver should backtrack and solve the puzzle"
        
        # Verify all entries are verified (solved)
        for entry in grid.entries.values():
            assert entry.verified, f"Entry {entry.entry_id} should be verified"


def test_solver_backtracks_multiple_times():
    """Test that solver backtracks multiple times if needed."""
    grid = create_grid()
    
    call_count: Dict[Tuple[str, str, int], int] = {}
    
    def mock_generate(entry: Entry, widening_level: int, max_candidates: int | None = None) -> list[ScoredCandidate]:
        # Inject wrong answers for multiple clues
        if "connects rotating" in entry.clue:  # 1A
            key = ("1A", entry.pattern, widening_level)
            call_count[key] = call_count.get(key, 0) + 1
            if widening_level == 0 and call_count[key] == 1:
                return [
                    ScoredCandidate(answer="BEND", confidence=0.8),
                    ScoredCandidate(answer="AXLE", confidence=0.7),
                ]
        
        elif "soooo cold" in entry.clue:  # 5A
            key = ("5A", entry.pattern, widening_level)
            call_count[key] = call_count.get(key, 0) + 1
            if widening_level == 0 and call_count[key] == 1:
                return [
                    ScoredCandidate(answer="ZZZ", confidence=0.8),
                    ScoredCandidate(answer="BRR", confidence=0.7),
                ]
        
        # Fallback: return empty list (stub behavior)
        return []
    
    def mock_verify(clue: str, answer: str) -> bool:
        """Verify answer by checking against correct answer in grid."""
        for entry in grid.entries.values():
            if entry.clue == clue:
                return answer.upper() == entry.correct_answer.upper()
        return False
    
    with patch('src.llm.LLM.generate_candidates', side_effect=mock_generate), \
         patch('src.llm.LLM.verify_answer', side_effect=mock_verify):
        solver = Solver(grid)
        result = solver.solve()
        assert result is True, "Solver should backtrack multiple times and solve"
        
        for entry in grid.entries.values():
            assert entry.verified, f"Entry {entry.entry_id} should be verified"
