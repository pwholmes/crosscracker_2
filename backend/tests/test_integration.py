"""Integration tests for LLM module.

These tests require a running Ollama server and should be run separately
from unit tests using: pytest -m integration
"""

import pytest
from src.llm import LLM
from src.model import Entry, Cell, Candidate


def create_test_entry(clue: str, answer: str, length: int) -> Entry:
    """Helper to create a test Entry with a simple grid."""
    grid: list[list[Cell]] = [[Cell(0, i) for i in range(length)]]
    return Entry("1A", clue, answer, grid, (0, 0), length)


@pytest.mark.integration
def test_generate_candidates_integration():
    """Integration test that calls the real LLM endpoint."""
    entry = create_test_entry("Old-fashioned butter maker", "CHURN", 5)
    candidates = LLM.generate_candidates(entry, search_level=0)

    assert isinstance(candidates, list)
    assert len(candidates) > 0
    assert len(candidates) <= LLM.MAX_CANDIDATES[0]
    for cand in candidates:
        assert isinstance(cand, Candidate)
        assert len(cand.answer) == entry.length


@pytest.mark.integration
def test_verify_answer_integration():
    """Integration test that calls the real LLM endpoint."""
    # Test with a clearly correct answer
    # Create a test entry for verification
    from src.model import Cell
    grid = [[Cell(0, i) for i in range(5)]]
    from src.model import Entry
    entry = Entry("1A", "Capital of France", "PARIS", grid, (0, 0), 5)
    result = LLM.verify_answer(entry, "PARIS")
    assert isinstance(result, bool)
    assert result == True