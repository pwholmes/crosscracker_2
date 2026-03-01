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

    def merge(self, candidate: Candidate) -> Candidate:
        self.search_level = max(self.search_level, candidate.search_level)
        self.llm_confidence = max(self.llm_confidence, candidate.llm_confidence)
        self.logprob_confidence = max(self.logprob_confidence, candidate.logprob_confidence)
        self.penalty = self.penalty + candidate.penalty
        return self
    
    @property
    def confidence(self) -> float:
        return self.llm_confidence * self.LLM_CONFIDENCE_WEIGHT + \
            self.logprob_confidence * self.LOGPROB_CONFIDENCE_WEIGHT

    def __hash__(self) -> int:
        # Hash based on entry_id and answer, which uniquely identify a candidate
        return hash((self.entry_id, self.answer))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Candidate):
            return False
        return (self.entry_id, self.answer) == (other.entry_id, other.answer)


@dataclass
class Placement:
    entry_id: str
    candidate: Candidate
    search_level: int
    pattern: str
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
                cells: list[list[Cell]],
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
                if c + i >= len(cells[0]):
                    raise ValueError(f"Cell out of bounds at ({r},{c+i}) for {entry_id}")
                self.cells.append(cells[r][c + i])
        elif entry_id[-1].upper() == "D":  # down
            for i in range(length):
                if r + i >= len(cells):
                    raise ValueError(f"Cell out of bounds at ({r+i},{c}) for {entry_id}")
                self.cells.append(cells[r + i][c])
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


    def get_candidates(self, widen_search: bool = False, matching_only: bool = True) -> list[Candidate]:
        """Get the pool of candidates for this entry, generating new candidates as
        necessary based on:
          - Whether we have generated candidates for the current pattern before.
          - If the widen_search parameter is True we will generate new candidates if the
          current search level is less than the maximum.
        If matching_only is True, only candidates matching the current pattern are returned."""
        from llm import LLM

        # If the current pattern has not been used to generate candidates, do so now.
        search_level = self._pattern_levels.get(self.pattern, -1)
        if search_level == -1 or (widen_search and search_level < LLM.MAX_SEARCH_LEVEL):
            search_level += 1
            # Call the LLM to generate candidates
            new_candidates = LLM.generate_candidates(self, search_level)
            # Store candidates and the pattern/search level used to generate them
            self._pattern_levels[self.pattern] = search_level
            for new_candidate in new_candidates:
                existing_candidate = self._candidates.get(new_candidate.answer)
                if existing_candidate:
                    logger.debug(f"[ENTRY] Entry {self.entry_id}, pattern '{self.pattern}', search level {search_level}: Merging candidate {new_candidate.answer}")
                    existing_candidate.merge(new_candidate)
                else:
                    logger.debug(f"[ENTRY] Entry {self.entry_id}, pattern '{self.pattern}', search level {search_level}: Storing new candidate {new_candidate}")
                    self._candidates[new_candidate.answer] = new_candidate

        # If we didn't find any candidates and we're at less than the maximum search
        # level, call this function recursively with the flag to bump the search level.
        if len(self._candidates) == 0:
            if self.search_level < LLM.MAX_SEARCH_LEVEL:
                logger.debug("[ENTRY GET CANDIDATES]: Recursively calling get_candidates() to bump search level.")
                return self.get_candidates(True, matching_only)

        candidates = list(self._candidates.values())
        if matching_only:
            candidates = [c for c in candidates if self.can_place_answer(c.answer)]
        return candidates

    
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


    def __str__(self):
        return f"Entry {self.entry_id}: Clue: {self.clue}, Candidates: {str(self._candidates)}"


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


    def place_entry(self, entry: Entry, answer: str, is_fallback: bool) -> bool:
        logger.debug(f"[GRID]: Placing answer: {entry.entry_id} = {answer}")
        
        # Make sure there are no conflicts with existing letters
        for cell, ch in zip(entry.cells, answer):
            if cell.letter is not None and cell.letter != ch:
                logger.error(f"[GRID ERROR] Unable to place candidate; candidate letter {ch} does not match existing letter {cell.letter}")
                return False
        
        # Add new letters to grid, noting if they are placed because this entry is a fallback
        for cell, ch in zip(entry.cells, answer):
            was_empty = cell.letter is None
            cell.letter = ch
            cell.sources.add(entry.entry_id)
            if is_fallback and was_empty:
                cell.revealed_by_fallback = True

        return True


    def remove_entry(self, entry: Entry):
        # Remove from grid all letters not also placed by a crossing entry
        for cell in entry.cells:
            if entry.entry_id in cell.sources:
                cell.sources.remove(entry.entry_id)
                if not cell.sources:
                    cell.letter = None
                    cell.revealed_by_fallback = False


    def pattern_for_entry(self, entry_id: str) -> str:
        return self.entries[entry_id].pattern


    def get_crossing_entry_ids(self, entry_id: str, incomplete_only: bool = False) -> set[str]:
        """Return a set of entry IDs that cross the given entry (share at least one cell)."""
        entry = self.entries[entry_id]
        crossing_ids: set[str] = set()
        for other_id, other_entry in self.entries.items():
            # Skip this entry
            if other_id == entry_id:
                continue
            # If specified, skip completed entries
            if incomplete_only and other_entry.completed:
                continue
            # Select only entries that share a cell with this entry
            if any(cell in other_entry.cells for cell in entry.cells):
                crossing_ids.add(other_id)
        return crossing_ids        