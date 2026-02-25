from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger("src.model")

@dataclass
class Cell:
    row: int
    col: int
    letter: str | None = None
    sources: set[str] = field(default_factory=lambda: set[str]())
    revealed_by_fallback: bool = False


@dataclass
class Candidate:
    """
    Represents a possible answer for a crossword entry, including scoring and placement context.
    Combines the previous Candidate and ScoredCandidate classes.
    """
    LLM_CONFIDENCE_WEIGHT = 0.6
    LOGPROB_CONFIDENCE_WEIGHT = 1 - LLM_CONFIDENCE_WEIGHT
    entry_id: str
    answer: str
    search_level: int = 0
    llm_confidence: float = 0.0
    logprob_confidence: float = 0.0
    penalty: float = 0.0
    is_fallback: bool = False

    def merge(self, candidate: Candidate) -> Candidate:
        self.search_level = max(self.search_level, candidate.search_level)
        self.llm_confidence = max(self.llm_confidence, candidate.llm_confidence)
        self.logprob_confidence = max(self.logprob_confidence, candidate.logprob_confidence)
        self.penalty = self.penalty + candidate.penalty
        self.is_fallback = self.is_fallback or candidate.is_fallback
        return self
    
    @property
    def confidence(self) -> float:
        return self.llm_confidence * self.LLM_CONFIDENCE_WEIGHT + \
            self.logprob_confidence * self.LOGPROB_CONFIDENCE_WEIGHT

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
    _candidates: dict[str,Candidate]
    """The pool of all candidates that have been generated for this entry, regardless 
    of whether they fit the current (or any other) pattern.  Keyed by answer."""
    _pattern_levels: dict[str,int]
    """The letter patterns that have been used to generate candidates for this entry,
    paired with the last search level each was used with.  Keyed by pattern."""

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
        self._candidates = {}
        self._pattern_levels = {}

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
        return self._pattern_levels.get(self.pattern, 0)
    
    @property
    def completed(self) -> bool:
        return "." not in self.pattern
    
    @property
    def verified(self) -> bool:
        """An entry is 'verified' if it is complete without having been explicitly placed;
        that is, it was completed entirely via crossing entries"""
        return self.completed and self.placement is None

    def add_candidate(self, answer: str, llm_confidence: float = 0, logprob_confidence: float = 0) -> None:
        candidate = self._candidates.get(answer)
        if candidate is None:
            candidate = Candidate(
                entry_id=self.entry_id,
                answer=answer, 
                llm_confidence=llm_confidence,
                logprob_confidence=logprob_confidence
            )
        else:
            candidate.llm_confidence = llm_confidence
            candidate.logprob_confidence = logprob_confidence

    def get_candidates(self, widen_search: bool = False) -> list[Candidate]:
        """Get the pool of candidates for this entry, generating new candidates as
        necessary based on:
          - Whether we have generated candidates for the current pattern before.
          - If the widen_search parameter is True we will generate new candidates if the
          current search level is less than the maximum.
        NOTE: This list is NOT filtered, even by pattern.  This is more of a "throw as 
        many darts as you can" approach.  It's up to the caller to filter the results."""
        from llm import LLM

        # If the current pattern has not been used to generate candidates, do so now.
        search_level = self._pattern_levels.get(self.pattern, -1)
        if search_level == -1 or (widen_search and search_level < LLM.MAX_SEARCH_LEVEL):
            search_level += 1
            # Call the LLM to generate candidates
            new_candidates = LLM.generate_candidates(self, search_level)
            # Store candidates and the pattern/search level used to generate them
            logger.debug(f"[ENTRY] Storing entry {self.entry_id}, pattern '{self.pattern}', search level {search_level}")
            self._pattern_levels[self.pattern] = search_level
            for new_candidate in new_candidates:
                self._candidates[new_candidate.answer] = new_candidate

        # If we didn't find any candidates and we're at less than the maximum search
        # level, call this function recusrively with the flag to bump the search level.
        if len (self._candidates) == 0:
             if self.search_level < LLM.MAX_SEARCH_LEVEL:
                 logger.debug("[ENTRY GET CANDIDATES]: Recursively calling get_candidates() to bump search level.")
                 return self.get_candidates(True)

        return list(self._candidates.values())

    
    def can_place_answer(self, answer: str, pattern: str|None = None) -> bool:
        if pattern is None:
            pattern = self.pattern
        return Entry.answer_matches_pattern(answer, pattern)

    @staticmethod
    def answer_matches_pattern(answer: str, pattern: str) -> bool:
        if len(answer) != len(pattern):
            return False
        return all(p == "." or p == a for p, a in zip(pattern, answer))

    def num_candidates(self, matching_only: bool = False) -> int:
        if matching_only:
            return sum(self.can_place_answer(c.answer) for c in self._candidates.values())
        else:
            return len(self._candidates)

    def __str__(self):
        return f"Entry {self.entry_id}: Candidates: {str(self._candidates)}"

    def get_crossing_letter(self, crossing_entry: Entry) -> tuple[int | None, str | None]:
        """
        Returns the position (index) in entry where it crosses crossing_entry,
        and the letter contributed by crossing_entry at that cell.
        If no crossing exists, returns (None, None).
        """
        # Find the shared cell
        for idx_entry, cell_entry in enumerate(self.cells):
            for cell_cross in crossing_entry.cells:
                if cell_entry is cell_cross:
                    # Return the index in stuck_entry and the letter
                    return idx_entry, cell_cross.letter
        return None, None


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
        logger.debug(f"[GRID]: Placing candidate: {candidate}")
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


