from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Cell:
    row: int
    col: int
    letter: str | None = None
    sources: set[str] = field(default_factory=lambda: set[str]())
    revealed_by_fallback: bool = False


@dataclass
class Entry:
    entry_id: str
    clue: str
    correct_answer: str
    cells: list[Cell]
    hints: list[tuple[str,str]] | None = None
    wrong_answers : list[str] | None = None
    used_fallback: bool = False
    verified: bool = False

    def __init__(
            self,
            entry_id: str,
            clue: str,
            correct_answer: str,
            grid: list[list[Cell]],
            start: tuple[int,int],
            length: int):
        """
        :param self: Description
        :param entry_id: The Entry's designation in the puzzle, given as a number plus the 
            letter A or D for Across or Down, respectively, e.g., "1A", "32D"
        :param clue: The Entry's clue, e.g., "A literary whale hunter"
        :param correct_answer: The Entry's correct answer, used for fallbacks, e.g., "AHAB"
        :param grid: The ENTIRE CROSSWORD GRID!  The constructor extracts the Cells specific
            to this Entry and stores them in its cells member.
        :param start: The 0-based coordinate of the Entry's starting location in the grid
        :param length: The Entry's length.  The length of the correct_answer MUST match this!
        """
        self.entry_id = entry_id
        self.clue = clue
        self.correct_answer = correct_answer
        self.cells = []

        r, c = start
        if entry_id[-1].upper() == "A":  # across
            for i in range(length):
                if c + i >= len(grid[0]):
                    raise ValueError(f"Cell out of bounds at ({r},{c+i}) for {entry_id}")
                self.cells.append(grid[r][c + i])
        elif entry_id[-1].upper() == "D":  # down
            for i in range(length):
                if r + i >= len(grid):
                    raise ValueError(f"Cell out of bounds at ({r+i},{c}) for {entry_id}")
                self.cells.append(grid[r + i][c])
        else:
            raise ValueError(f"Invalid entry_id {entry_id}, must end with 'A' or 'D'")

    @property
    def length(self) -> int:
        return len(self.cells)

    @property
    def pattern(self) -> str:
        return "".join(cell.letter or "." for cell in self.cells)
    

@dataclass
class Candidate:
    entry_id: str
    answer: str
    widening_level: int
    is_fallback: bool = False


@dataclass(frozen=True)
class ScoredCandidate:
    answer: str
    confidence: float
    selection_score: float = 0.0


class Grid:
    def __init__(self, entries: dict[str, Entry]):
        self.entries = entries  # all Entries keyed by entry_id

    def place_candidate(self, candidate: Candidate) -> bool:
        entry = self.entries[candidate.entry_id]
        for cell, ch in zip(entry.cells, candidate.answer):
            if cell.letter is not None and cell.letter != ch:
                return False
        for cell, ch in zip(entry.cells, candidate.answer):
            was_empty = cell.letter is None
            cell.letter = ch
            cell.sources.add(entry.entry_id)
            if candidate.is_fallback and was_empty:
                cell.revealed_by_fallback = True
        return True

    def remove_candidate(self, candidate: Candidate):
        entry = self.entries[candidate.entry_id]
        for cell in entry.cells:
            if entry.entry_id in cell.sources:
                cell.sources.remove(entry.entry_id)
                if not cell.sources:
                    cell.letter = None
                    cell.revealed_by_fallback = False

    def pattern_for_entry(self, entry_id: str) -> str:
        return self.entries[entry_id].pattern


class CandidateCache:
    """Cache candidate lists by (entry_id, pattern, widening_level)."""
    def __init__(self):
        """Initialize the in-memory cache store."""
        self._cache: dict[tuple[str, str, int], list[ScoredCandidate]] = {}

    def get(
        self,
        entry_id: str,
        pattern: str,
        widening_level: int,
    ) -> list[ScoredCandidate] | None:
        """Return cached candidates for the given key, or None if not present."""
        return self._cache.get((entry_id, pattern, widening_level))

    def put(
        self,
        entry_id: str,
        pattern: str,
        widening_level: int,
        candidates: list[ScoredCandidate],
    ) -> None:
        """Store candidates for the given key, overwriting any existing entry."""
        self._cache[(entry_id, pattern, widening_level)] = candidates
