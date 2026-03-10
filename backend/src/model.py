from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar, Final, Any, Optional
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
    LLM_CONFIDENCE_WEIGHT: ClassVar[Final[float]] = 0.6
    LOGPROB_CONFIDENCE_WEIGHT: ClassVar[Final[float]] = 1 - LLM_CONFIDENCE_WEIGHT
    BACKTRACK_PENALTY: ClassVar[Final[int]]  = 25
    VERIFICATION_PENALTY: ClassVar[Final[int]] = 5
    MINIMUM_CONFIDENCE: ClassVar[Final[int]] = 20
    entry_id: str
    answer: str
    search_level: int = 0
    llm_confidence: float = field(default_factory=lambda: float('-inf'))
    logprob_confidence: float = field(default_factory=lambda: float('-inf'))
    backtracks : int = 0
    verification_failures: int = 0

    def merge(self, candidate: Candidate) -> Candidate:
        self.search_level = max(self.search_level, candidate.search_level)
        self.llm_confidence = max(self.llm_confidence, candidate.llm_confidence)
        self.logprob_confidence = max(self.logprob_confidence, candidate.logprob_confidence)
        return self
    
    @property
    def confidence(self) -> float:
        """
        Calculate weighted confidence, intelligently handling unset scores.
        • If only LLM confidence is set, use it directly.
        • If only logprob confidence is set, use it directly.
        • If both are set, use weighted average (60% LLM, 40% logprob).
        • If neither is set, return MINIMUM_CONFIDENCE.
        """
        llm_set = self.llm_confidence > float('-inf')
        logprob_set = self.logprob_confidence > float('-inf')
        
        if not llm_set and not logprob_set:
            # Neither score has been set; return default
            return float(self.MINIMUM_CONFIDENCE)
        elif llm_set and not logprob_set:
            # Only LLM confidence is set; use it directly
            base_confidence = self.llm_confidence
        elif logprob_set and not llm_set:
            # Only logprob confidence is set; use it directly
            base_confidence = self.logprob_confidence
        else:
            # Both are set; use weighted average
            base_confidence = self.llm_confidence * self.LLM_CONFIDENCE_WEIGHT + \
                self.logprob_confidence * self.LOGPROB_CONFIDENCE_WEIGHT
        
        penalized = base_confidence - \
            self.backtracks * Candidate.BACKTRACK_PENALTY - \
            self.verification_failures * Candidate.VERIFICATION_PENALTY
        
        # Only apply minimum floor if base confidence was above threshold
        if base_confidence >= Candidate.MINIMUM_CONFIDENCE:
            return max(Candidate.MINIMUM_CONFIDENCE, penalized)
        else:
            return penalized


    def __hash__(self) -> int:
        # Hash based on entry_id and answer, which uniquely identify a candidate
        return hash((self.entry_id, self.answer))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Candidate):
            return False
        return (self.entry_id, self.answer) == (other.entry_id, other.answer)

    def serialize(self) -> dict[str, Any]:
        """Serialize this candidate to a dict."""
        return {
            "answer": self.answer,
            "search_level": self.search_level,
            "llm_confidence": self.llm_confidence,
            "logprob_confidence": self.logprob_confidence,
            "backtracks": self.backtracks,
            "verification_failures": self.verification_failures
        }
    
    @staticmethod
    def deserialize(entry_id: str, data: dict[str, Any]) -> Candidate:
        """Deserialize a candidate from a dict."""
        return Candidate(
            entry_id=entry_id,
            answer=data["answer"],
            search_level=data.get("search_level", 0),
            llm_confidence=data.get("llm_confidence", float('-inf')),
            logprob_confidence=data.get("logprob_confidence", float('-inf')),
            backtracks=data.get("backtracks", 0),
            verification_failures=data.get("verification_failures", 0),
        )


@dataclass
class Placement:
    entry_id: str
    candidate: Candidate
    pattern: str
    selection_score: float
    is_fallback: bool = False
    
    def serialize(self) -> dict[str, Any]:
        """Serialize this placement to a dict."""
        return {
            "entry_id": self.entry_id,
            "candidate": self.candidate.serialize(),
            "pattern": self.pattern,
            "selection_score": self.selection_score,
            "is_fallback": self.is_fallback,
        }
    
    @staticmethod
    def deserialize(data: dict[str, Any]) -> Placement:
        """Deserialize a placement from a dict."""
        candidate = Candidate.deserialize(data["entry_id"], data["candidate"])
        return Placement(
            entry_id=data["entry_id"],
            candidate=candidate,
            pattern=data["pattern"],
            selection_score=data["selection_score"],
            is_fallback=data.get("is_fallback", False),
        )


class Entry:
    entry_id: str
    clue: str
    correct_answer: str
    cells: list[Cell]
    hints: list[tuple[str,str,float]] | None = None
    placement: Optional[Placement] = None
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

    @property
    def total_backtracks(self) -> int:
        total = 0
        for candidate in self._candidates.values():
            total += candidate.backtracks
        return total

    @property
    def filled_count(self) -> int:
        """Returns the number of filled letters (non-'.') in the current pattern."""
        return sum(1 for ch in self.pattern if ch != ".")

    @property
    def candidate_threshold(self) -> int:
        """Returns the number of viable candidates this entry should have to avoid 
         another call to the LLM to generate more """
        # I know, it's more complicated than it needs to be, but I wanted to make it
        # a "tuneable" formula.  Right now it's just "2 candidates if the entry is
        # completely empty, otherwise 1"
        return max(1, 2 - self.filled_count)


    def get_candidates(self, pattern:str|None = None, widen_search: bool = False, matching_only: bool = True) -> list[Candidate]:
        """Get the pool of candidates for this entry, generating new candidates as
        necessary based on:
          - Whether we have generated candidates for the pattern before.
          - If the widen_search parameter is True we will generate new candidates if the
          current search level is less than the maximum.
        If matching_only is True, only candidates matching the pattern are returned."""
        from llm import LLM

        # If a pattern is specified, use it
        if not pattern:
            pattern = self.pattern

        # If we already have "enough" viable candidates for this entry, don't bother 
        # calling the LLM to generate new ones.
        if self.num_candidates(True) < self.candidate_threshold:
            # If the pattern has not been used to generate candidates, do so now.
            search_level = self._pattern_levels.get(pattern, -1)
            if search_level == -1 or (widen_search and search_level < LLM.MAX_SEARCH_LEVEL):
                search_level += 1
                # Call the LLM to generate candidates
                new_candidates = LLM.generate_candidates(self, pattern, search_level)
                # Store candidates and the pattern/search level used to generate them
                self._pattern_levels[pattern] = search_level
                for new_candidate in new_candidates:
                    existing_candidate = self._candidates.get(new_candidate.answer)
                    if existing_candidate:
                        logger.debug(f"[ENTRY] Entry {self.entry_id}, pattern '{pattern}', search level {search_level}: Merging candidate {new_candidate.answer}")
                        existing_candidate.merge(new_candidate)
                    else:
                        logger.debug(f"[ENTRY] Entry {self.entry_id}, pattern '{pattern}', search level {search_level}: Storing new candidate {new_candidate}")
                        self._candidates[new_candidate.answer] = new_candidate

        # Filter candidates by pattern matching if requested
        candidates = list(self._candidates.values())
        if matching_only:
            candidates = [c for c in candidates if self.can_place_answer(c.answer)]

        # If no matching candidates exist and we can still widen search, do so recursively
        if len(candidates) == 0 and self.search_level < LLM.MAX_SEARCH_LEVEL:
            logger.debug(f"[ENTRY GET CANDIDATES]: No matching candidates for entry {self.entry_id}, pattern '{self.pattern}', recursively widening search.")
            return self.get_candidates(pattern, True, matching_only)

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


    def get_candidate(self, answer: str) -> Candidate | None:
        return self._candidates.get(answer)


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


    def place(self, placement: Placement) -> bool:
        """Place a candidate answer into this entry's cells.
        
        Sets this entry's placement and adds the answer to the grid cells.
        Returns False if there are conflicts with existing letters, True otherwise.
        """
        answer = placement.candidate.answer
        logger.debug(f"[ENTRY]: Placing answer: {self.entry_id} = {answer}")
        
        # Make sure there are no conflicts with existing letters
        for cell, ch in zip(self.cells, answer):
            if cell.letter is not None and cell.letter != ch:
                logger.error(f"[ENTRY ERROR] Unable to place candidate; candidate letter {ch} does not match existing letter {cell.letter}")
                return False
        
        # Add new letters to cells, noting if they are placed because this entry is a fallback
        for cell, ch in zip(self.cells, answer):
            was_empty = cell.letter is None
            cell.letter = ch
            cell.sources.add(self.entry_id)
            if placement.is_fallback and was_empty:
                cell.revealed_by_fallback = True

        # Set the placement
        self.placement = placement
        return True


    def remove(self) -> None:
        """Remove this entry's answer from its cells and clear placement.
        
        Only removes letters not also placed by a crossing entry.
        """
        for cell in self.cells:
            if self.entry_id in cell.sources:
                cell.sources.remove(self.entry_id)
                if not cell.sources:
                    cell.letter = None
                    cell.revealed_by_fallback = False

        # Clear the placement
        self.placement = None


    def serialize(self) -> dict[str, Any]:
        """Serialize this entry's state for checkpointing.
        
        Returns a dict containing candidates, pattern levels, placement, and other state.
        """
        # Serialize all candidates for this entry
        candidates_list: list[dict[str, Any]] = []
        for candidate in self._candidates.values():
            candidates_list.append(candidate.serialize())
        
        # Serialize placement if exists
        placement_data: dict[str, Any] | None = None
        if self.placement:
            placement_data = self.placement.serialize()
        
        return {
            "candidates": candidates_list,
            "pattern_levels": dict(self._pattern_levels),
            "placement": placement_data,
            "used_fallback": self.used_fallback,
        }
    
    def deserialize(self, entry_data: dict[str, Any]) -> None:
        """Restore this entry's state from checkpoint data.
        
        Reconstructs candidates, pattern levels, placement, and other state.
        """
        # Restore candidates (entry._candidates is already empty due to defer_candidate_init=True)
        candidates = entry_data.get("candidates", [])
        for cand_data in candidates:
            candidate = Candidate.deserialize(self.entry_id, cand_data)
            # CRITICAL: Reject any candidate that contains "." - this represents a pattern, not an answer
            if "." in candidate.answer or  ' ' in candidate.answer:
                logger.error(f"[ENTRY ERROR] Entry {self.entry_id}: Deserialized candidate with embedded '.' or ' ': '{candidate.answer}' - REJECTING. Checkpoint may be corrupted!")
                continue
            self._candidates[candidate.answer] = candidate
        
        # Restore pattern levels
        self._pattern_levels = dict(entry_data.get("pattern_levels", {}))
        
        # Restore placement
        placement_data = entry_data.get("placement")
        if placement_data:
            self.placement = Placement.deserialize(placement_data)
        else:
            self.placement = None
        
        # Restore other state
        self.used_fallback = entry_data.get("used_fallback", False)

    def __str__(self):
        return f"Entry {self.entry_id}: Clue: {self.clue}, Candidates: {str(self._candidates)}"


class Grid:
    puzzle_id: str | None
    entries: dict[str,Entry]

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
    
    def serialize(self) -> dict[str, Any]:
        """Serialize this grid's state for checkpointing.
        
        Returns a dict containing puzzle_id, all entry states, and all cell states.
        """
        # Serialize entry state by calling each entry's serialize method
        entries_state: dict[str, Any] = {}
        for entry_id, entry in self.entries.items():
            entries_state[entry_id] = entry.serialize()
        
        # Serialize grid cell state
        cells_state: list[dict[str, Any]] = []
        for entry in self.entries.values():
            for cell in entry.cells:
                cells_state.append({
                    "row": cell.row,
                    "col": cell.col,
                    "letter": cell.letter,
                    "sources": list(cell.sources),
                    "revealed_by_fallback": cell.revealed_by_fallback,
                })
        
        return {
            "puzzle_id": self.puzzle_id,
            "entries_state": entries_state,
            "cells_state": cells_state,
        }
    
    def deserialize(self, checkpoint_data: dict[str, Any]) -> None:
        """Restore this grid's state from checkpoint data.
        
        Clears all cells and reconstructs entry and cell states from checkpoint.
        """
        entries_state: dict[str, dict[str, Any]] = checkpoint_data.get("entries_state", {})
        cells_state: list[dict[str, Any]] = checkpoint_data.get("cells_state", [])
        
        # First clear all grid cells
        for entry in self.entries.values():
            for cell in entry.cells:
                cell.letter = None
                cell.sources.clear()
                cell.revealed_by_fallback = False
        
        # Restore cell states
        cell_map: dict[tuple[int, int], dict[str, Any]] = {}
        for cell_data in cells_state:
            key = (cell_data["row"], cell_data["col"])
            cell_map[key] = cell_data
        
        for entry in self.entries.values():
            for cell in entry.cells:
                key = (cell.row, cell.col)
                if key in cell_map:
                    cell_data = cell_map[key]
                    cell.letter = cell_data.get("letter")
                    cell.sources = set(cell_data.get("sources", []))
                    cell.revealed_by_fallback = cell_data.get("revealed_by_fallback", False)
        
        # Restore entry states using Entry's deserialize method
        for entry_id, entry_data in entries_state.items():
            entry = self.entries.get(entry_id)
            if not entry:
                continue
            
            entry.deserialize(entry_data)        