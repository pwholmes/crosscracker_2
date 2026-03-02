from src.heuristics import BasicStrategy
from src.model import Candidate, Cell, Entry, Grid, Placement


def _create_fallback_test_grid() -> Grid:
    cells = [[Cell(row=r, col=c) for c in range(2)] for r in range(3)]
    entries: dict[str, Entry] = {
        "1A": Entry("1A", "Top across", "AB", cells, (0, 0), 2),
        "2A": Entry("2A", "Middle across", "CD", cells, (1, 0), 2),
        "3A": Entry("3A", "Bottom across", "EF", cells, (2, 0), 2),
        "1D": Entry("1D", "Left down", "ACE", cells, (0, 0), 3),
        "2D": Entry("2D", "Right down", "BDF", cells, (0, 1), 3),
    }
    return Grid(entries)


def _place_entry(entry: Entry, answer: str) -> None:
    placement = Placement(
        entry_id=entry.entry_id,
        candidate=Candidate(entry_id=entry.entry_id, answer=answer),
        pattern=entry.pattern,
        selection_score=0.0,
        is_fallback=False,
    )
    placed = entry.place(placement)
    assert placed


def test_select_best_fallback_target_prefers_more_unfilled_crossings() -> None:
    grid = _create_fallback_test_grid()

    # Place middle and bottom across entries:
    # - 1A then has two unfilled crossings (1D, 2D) and zero filled crossings
    # - 1D/2D each have one unfilled crossing (1A) and two filled crossings (2A, 3A)
    # So 1A should have the highest fallback score.
    _place_entry(grid.entries["2A"], "CD")
    _place_entry(grid.entries["3A"], "EF")

    selected = BasicStrategy.select_best_fallback_target(grid)

    assert selected is not None
    assert selected.entry_id == "1A"


def test_select_best_fallback_target_returns_none_when_no_unfilled_crossings() -> None:
    grid = _create_fallback_test_grid()

    # Fill all downs so every remaining unplaced entry has zero unfilled crossings.
    _place_entry(grid.entries["1D"], "ACE")
    _place_entry(grid.entries["2D"], "BDF")

    selected = BasicStrategy.select_best_fallback_target(grid)

    assert selected is None
