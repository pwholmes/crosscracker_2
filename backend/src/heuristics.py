import logging
from model import Grid, Entry, Candidate
from collections.abc import Callable

class BasicStrategy:
    @staticmethod
    def select_best_unfilled_entry(entries: dict[str,Entry], attempted_entries: set[str]) -> tuple[str, float] | None:
        """Select the best unfilled entry using a heuristic that balances:
        1. Candidate confidence
        2. Entry completeness (filled letters)
        3. Entry length (longer entries preferred)
        """
        # These should sum to 1.0d
        CONFIDENCE_WEIGHT = 0.3
        LENGTH_WEIGHT = 0.2
        COMPLETENESS_WEIGHT = 0.5

        best_entry_id: str | None = None
        best_score: float = float("-inf")

        for entry in entries.values():
            if "." not in entry.pattern:
                continue
            if entry.entry_id in attempted_entries:
                continue

            cand = BasicStrategy.select_best_candidate(entry)
            if cand is None:
                continue

            # Calculate completeness bonus
            blank_count = entry.pattern.count(".")
            filled_count = entry.length - blank_count
            completeness = filled_count / entry.length

            # Total score = confidence (0-100) + completeness bonus - penalty
            score = CONFIDENCE_WEIGHT * (cand.confidence - cand.penalty) + \
                    LENGTH_WEIGHT * min(entry.length,10)/10 * 100 + \
                    COMPLETENESS_WEIGHT * completeness * 100

            if best_entry_id is None or score > best_score:
                best_entry_id = entry.entry_id
                best_score = score

        if best_entry_id is None:
            return None
        return best_entry_id, best_score

    @staticmethod
    def select_best_candidate(entry: Entry, widen_search: bool = False) -> Candidate | None:
        candidates = entry.get_candidates(widen_search)

        best_candidate = None
        best_effective_confidence = float("-inf")

        for candidate in candidates:
            if len(candidate.answer) != entry.length:
                continue
            if not BasicStrategy._can_place(entry, candidate.answer):
                continue

            # Calculate effective confidence (confidence minus penalty)
            effective_confidence = candidate.confidence - candidate.penalty

            if effective_confidence > best_effective_confidence:
                best_candidate = candidate
                best_effective_confidence = effective_confidence

        return best_candidate

    @staticmethod
    def _can_place(entry: Entry, answer: str) -> bool:
        for cell, ch in zip(entry.cells, answer):
            if cell.letter is not None and cell.letter != ch:
                return False
        return True


    @staticmethod
    def select_best_backtrack_target(grid: Grid, get_crossing_ids_func: Callable[[str], set[str]]) -> str | None:
        """Select a backtrack target by finding the entry that appears most frequently
        as a crossing to unfilled entries.
        
        Algorithm:
        1. For each unplaced entry, identify all placed crossing entries
        2. Award each unplaced crossing entry a point
        3. Select the placed entry with the most points
        4. Break ties by lowest confidence score
        """
        logger = logging.getLogger("src.heuristics")
        entries = grid.entries.values()
        points: dict[str,int] = {}

        # Loop through placed non-fallback entries, assigning a point for every unplaced
        # entry that crosses it.
        for entry in entries:
            if entry.placement == None or entry.used_fallback:
                continue

            total = 0
            crossing_entry_ids = get_crossing_ids_func(entry.entry_id)
            for crossing_entry_id in crossing_entry_ids:
                crossing_entry = grid.entries[crossing_entry_id]
                if not crossing_entry.completed:
                    total += 1
            points[entry.entry_id] = total

        # If no entries were added to the points list, we couldn't find an entry to backtrack
        if not points:
            return None

        # Find the entry with the most points
        max_points = max(points.values())
        # There will always be at least ONE with the max value
        tied_entries = [entry for entry in entries if entry.entry_id in points and points[entry.entry_id] == max_points]

        # Break ties by lowest confidence score
        selected = min(tied_entries, key=lambda rec: rec.placement.confidence if rec.placement is not None else float("inf"))
        logger.debug(
            f"BACKTRACK TARGET SELECTED: entry_id={selected.entry_id} "
            f"points={max_points} "
            f"confidence={selected.placement.confidence:.2f}" if selected.placement is not None else "confidence=None"
        )

        return selected.entry_id

    @staticmethod
    def select_best_fallback_target(grid: Grid, get_crossing_ids_func: Callable[[str], set[str]]) -> Entry | None:
        # These weights should sum to 1
        W_UNFILLED_RATIO = 0.5
        #W_CONFIDENCE = 0.3
        #W_RETRY = 0.2

        best_entry = None
        best_score = float("-inf")

        entries = grid.entries.values()
        for entry in entries:
            if entry.placement is None:
                continue
            crossing_entry_ids = get_crossing_ids_func(entry.entry_id)
            unfilled_count = 0
            filled_count = 0
            for crossing_entry_id in crossing_entry_ids:
                crossing_entry = grid.entries[crossing_entry_id]
                if "." in crossing_entry.pattern:
                    unfilled_count += 1
                else:
                    filled_count += 1
            if unfilled_count == 0:
                continue
            score = W_UNFILLED_RATIO * unfilled_count / (filled_count + 1)
            if score > best_score:
                best_score = score
                best_entry = entry

        return best_entry
