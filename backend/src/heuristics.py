import logging
from model import Grid, Entry, Candidate
from collections.abc import Callable

class BasicStrategy:
    @staticmethod
    def select_best_unfilled_entry(entries: dict[str,Entry], attempted_entries: set[tuple[str, str]]) -> tuple[Entry, Candidate, float] | None:
        """Select the best unfilled entry using a heuristic that balances:
        1. Candidate confidence
        2. Entry completeness (filled letters)
        3. Entry length (longer entries preferred)
        """
        # These should sum to 1.0
        CONFIDENCE_WEIGHT = 0.25
        LENGTH_WEIGHT = 0.2
        COMPLETENESS_WEIGHT = 0.35
        CONSTRAINT_WEIGHT = 0.2

        best_entry: Entry | None = None
        best_candidate: Candidate | None = None
        best_score: float = float("-inf")

        for entry in entries.values():
            if entry.completed:
                continue

            attempted_candidates = {answer for entry_id, answer in attempted_entries if entry_id == entry.entry_id}

            cand = BasicStrategy.select_best_candidate(entry, attempted_candidates)
            if cand is None:
                continue

            # Calculate completeness bonus
            blank_count = entry.pattern.count(".")
            filled_count = entry.length - blank_count
            completeness = filled_count / entry.length

            confidence_score = CONFIDENCE_WEIGHT * (cand.confidence - cand.penalty)
            # This length score favors longer answers
            length_score = LENGTH_WEIGHT * min(entry.length,10)/10 * 100
            # This length score favors shorter answers
            #length_score = LENGTH_WEIGHT * max(0, 100 - 100/9 * (entry.length - 1))
            completeness_score = COMPLETENESS_WEIGHT * completeness * 100
            constraint_weight = CONSTRAINT_WEIGHT * (100 / max(1, entry.num_candidates(True)))
            score =  confidence_score + length_score + completeness_score + constraint_weight

            if best_entry is None or score > best_score:
                best_entry = entry
                best_candidate = cand
                best_score = score

        if best_entry is None or best_candidate is None:
            return None
        return best_entry, best_candidate, best_score


    @staticmethod
    def select_best_candidate(entry: Entry, attempted_candidates: set[str], widen_search: bool = False) -> Candidate | None:
        candidates = entry.get_candidates(widen_search)

        best_candidate = None
        best_effective_confidence = float("-inf")

        for candidate in candidates:
            if (candidate in attempted_candidates):
                continue
            if len(candidate.answer) != entry.length:
                continue
            if not entry.can_place_answer(candidate.answer):
                continue

            # Calculate effective confidence (confidence minus penalty)
            effective_confidence = candidate.confidence - candidate.penalty

            if effective_confidence > best_effective_confidence:
                best_candidate = candidate
                best_effective_confidence = effective_confidence

        return best_candidate


    @staticmethod
    def select_best_backtrack_target(grid: Grid, get_crossing_ids_func: Callable[[str], set[str]], top_n_candidates: int = 5) -> str | None:
        """Select a backtrack target by finding the placed entry most likely to be
        blocking progress on unfilled entries.
        
        Algorithm:
        For each unplaced entry (the "stuck" entry):
          - If it has no candidates, directly blame all placed crossing entries,
            weighted by inverse confidence (shakier crossings get more blame)
          - Otherwise, examine its top N candidates
          - For each placed crossing entry, determine what letter it contributes
            at the shared cell
          - Count how many of the stuck entry's top N candidates conflict with
            that crossing letter
          - Award blame to the crossing entry proportional to the conflict ratio,
            weighted by candidate scarcity
        Select the placed entry with the highest total blame score.
        Break ties by lowest confidence.
        """
        logger = logging.getLogger("src.heuristics")
        entries = grid.entries.values()
        blame: dict[str, float] = {}

        # Initialize blame scores for all eligible entries
        for entry in entries:
            if entry.placement is not None and not entry.used_fallback:
                blame[entry.entry_id] = 0.0

        # Loop through stuck (unplaced) entries and blame their crossing entries
        for entry in entries:
            if entry.completed:
                continue

            candidates: list[Candidate] = entry.get_candidates()[:top_n_candidates]
            crossing_entry_ids = get_crossing_ids_func(entry.entry_id)

            if not candidates:
                # No candidates at all -- directly blame all placed crossing entries,
                # weighted by inverse confidence so shakier entries get more blame
                for crossing_entry_id in crossing_entry_ids:
                    crossing_entry = grid.entries[crossing_entry_id]
                    if crossing_entry.placement is None or crossing_entry.used_fallback:
                        continue
                    blame[crossing_entry_id] = blame.get(crossing_entry_id, 0.0) + (1.0 - crossing_entry.placement.confidence)
                continue

            scarcity_weight = 1.0 + (1.0 / len(candidates))  # more weight when fewer candidates exist

            for crossing_entry_id in crossing_entry_ids:
                crossing_entry = grid.entries[crossing_entry_id]
                if crossing_entry.placement is None or crossing_entry.used_fallback:
                    continue

                # What letter is the crossing entry contributing at the shared cell?
                cross_position, crossing_letter = entry.get_crossing_letter(crossing_entry)
                if cross_position is None or crossing_letter is None:
                    continue

                # How many of this entry's candidates conflict with that letter?
                conflicting = sum(
                    1 for candidate in candidates
                    if candidate.answer[cross_position] != crossing_letter
                )
                conflict_ratio = conflicting / len(candidates)
                blame[crossing_entry_id] = blame.get(crossing_entry_id, 0.0) + conflict_ratio * scarcity_weight

        if not blame:
            return None

        # Filter to only entries that have any blame at all, falling back to all entries if none do
        blamed_entries = {k: v for k, v in blame.items() if v > 0}
        pool = blamed_entries if blamed_entries else blame

        max_blame = max(pool.values())
        tied_entries = [entry for entry in entries if entry.entry_id in pool and pool[entry.entry_id] == max_blame]

        # Break ties by lowest confidence score
        selected = min(tied_entries, key=lambda rec: rec.placement.confidence if rec.placement is not None else float("inf"))

        logger.debug(
            f"BACKTRACK TARGET SELECTED: entry_id={selected.entry_id} "
            f"blame={max_blame:.2f} "
            + (f"confidence={selected.placement.confidence:.2f}" if selected.placement is not None else "confidence=None")
        )

        return selected.entry_id


    @staticmethod
    def select_best_backtrack_target_old(grid: Grid, get_crossing_ids_func: Callable[[str], set[str]]) -> str | None:
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
                if not crossing_entry.completed:
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
