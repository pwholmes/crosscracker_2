from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Cell:
    row: int
    col: int
    letter: str | None = None
    sources: set[str] = field(default_factory=lambda: set[str]())
    revealed_by_fallback: bool = False

@dataclass
class CandidatesAtSearchLevel:
    """ Candidates are generated at a specific search level for each pattern.  """
    search_level: int
    candidates: list[Candidate]


@dataclass
class Placement:
    entry_id: str
    answer: str
    search_level: int
    pattern: str
    confidence: float
    selection_score: float
    is_fallback: bool = False


class Entry:
    entry_id: str
    clue: str
    correct_answer: str
    cells: list[Cell]
    hints: list[tuple[str,str]] | None = None
    placement: Optional[Placement] = None
    backtracks: int = 0
    used_fallback: bool = False
    _candidates_at_search_level: dict[str, CandidatesAtSearchLevel]
    """ The candidate list AND search level for this entry, keyed by pattern.  We store 
    these together as a class instead of keying candidate by pattern and search level
    because we NEVER need to go back to a previous search level for a given entry+pattern. """

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
        self._candidates_at_search_level = {}

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
        """Dynamically generates the Entry's current pattern.  Will return the full
        answer if it has been placed."""
        return "".join(cell.letter or "." for cell in self.cells)
    
    @property
    def search_level(self, pattern: str|None = None) -> int:
        """Returns the CURRENT search level for the specified pattern"""
        if pattern is None:
            pattern = self.pattern
        candidates_at_search_level = self._candidates_at_search_level.get(self.pattern)
        if candidates_at_search_level is None:
            return 0
        return self._candidates_at_search_level[self.pattern].search_level
    
    @property
    def completed(self) -> bool:
        return "." not in self.pattern
    
    @property
    def verified(self) -> bool:
        """An entry is 'verified' if it is complete without having been explicitly placed;
        that is, it was completed entirely via crossing entries"""
        return self.completed and self.placement is None

    def get_candidates(self) -> list[Candidate]:
        from llm import LLM

        if self.pattern in self._candidates_at_search_level:
            return self._candidates_at_search_level[self.pattern].candidates
        
        candidates = LLM.generate_candidates(self, self.search_level)
        self._candidates_at_search_level[self.pattern] = CandidatesAtSearchLevel(self.search_level, candidates)

        return candidates



@dataclass
class Candidate:
    """
    Represents a possible answer for a crossword entry, including scoring and placement context.
    Combines the previous Candidate and ScoredCandidate classes.
    """
    entry_id: str
    answer: str
    search_level: int = 0
    confidence: float = 50.0
    penalty: float = 0.0
    is_fallback: bool = False


class Grid:
    def __init__(self, entries: dict[str, Entry]):
        self.puzzle_id: str | None = None  # Set by server when puzzle is loaded
        self.entries = entries  # all Entries keyed by entry_id
        
        # Calculate grid dimensions from cell positions
        max_row = 0
        max_col = 0
        for entry in entries.values():
            for cell in entry.cells:
                max_row = max(max_row, cell.row)
                max_col = max(max_col, cell.col)
        
        self.width = max_col + 1
        self.height = max_row + 1

    def place_candidate(self, candidate: Candidate) -> bool:
        print(candidate)
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


