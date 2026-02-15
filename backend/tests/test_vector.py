import pytest
from src.vector import open_database, query_database
from src.model import Entry, Cell


def create_cell_grid(rows: int, columns: int) -> list[list[Cell]]:
    """Create a 2D grid of empty Cell objects with the specified dimensions."""
    grid: list[list[Cell]] = []
    for row in range(rows):
        grid_row: list[Cell] = []
        for col in range(columns):
            grid_row.append(Cell(row=row, col=col))
        grid.append(grid_row)
    return grid

@pytest.mark.integration
def test_query_database():
    """Test that query_database retrieves hints for entries."""
    # Create a small grid
    grid = create_cell_grid(2, 10)
    
    # Create test entries
    entries = [
        Entry("1A", "Old-fashioned butter maker", "CHURN", grid, (0, 0), 5),
        Entry("2A", "Branded cotton swab", "QTIP", grid, (1, 0), 4),
    ]
    
    # Open database and query
    collection = open_database()
    query_database(collection, entries)
    
    # Verify hints were populated
    for entry in entries:
        assert entry.hints is not None, f"Entry {entry.entry_id} should have hints"
        assert len(entry.hints) > 0, f"Entry {entry.entry_id} should have at least one hint"
        
        # Check structure of hints (list of tuples with strings)
        for hint_clue, hint_answer in entry.hints:
            assert isinstance(hint_clue, str), "Hint clue should be a string"
            assert isinstance(hint_answer, str), "Hint answer should be a string"
            assert len(hint_clue) > 0, "Hint clue should not be empty"
            assert len(hint_answer) > 0, "Hint answer should not be empty"
