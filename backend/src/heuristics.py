import logging
from model import Grid, Entry, Candidate

class BasicStrategy:
    # Weights should sum to 1.0
    SELECTION_CONFIDENCE_WEIGHT = 0.25
    SELECTION_LENGTH_WEIGHT = 0.2
    SELECTION_COMPLETENESS_WEIGHT = 0.35
    SELECTION_CONSTRAINT_WEIGHT = 0.2
    MIN_CANDIDATE_CONFIDENCE_THRESHOLD = 20

    @staticmethod
    def select_best_unfilled_entry(grid: Grid, attempted_entries: set[tuple[str, str]]) -> tuple[Entry, Candidate, float] | None:
        """Select the best unfilled entry using a heuristic that balances:
        1. Candidate confidence (higher confidence preferred)
        2. Entry completeness (higher percentage of filled letters preferred)
        3. Entry length (longer entries preferred)
        4. Candidate scarcity (entries with fewer candidates preferred)
        """
        best_entry: Entry | None = None
        best_candidate: Candidate | None = None
        best_score: float = float("-inf")

        for entry in grid.entries.values():
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

            confidence_score = BasicStrategy.SELECTION_CONFIDENCE_WEIGHT * (cand.confidence - cand.penalty)
            # This length score favors longer answers
            length_score = BasicStrategy.SELECTION_LENGTH_WEIGHT * min(entry.length,10)/10 * 100
            # This length score favors shorter answers
            #length_score = LENGTH_WEIGHT * max(0, 100 - 100/9 * (entry.length - 1))
            completeness_score = BasicStrategy.SELECTION_COMPLETENESS_WEIGHT * completeness * 100
            constraint_weight = BasicStrategy.SELECTION_CONSTRAINT_WEIGHT * (100 / max(1, entry.num_candidates(True)))
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
        """
        Select the best candidate for an entry:
        1. It must be one we haven't already tried in this pass through the puzzle
        2. It must be the correct length
        3. It must have a certain minimum confidence level
        4. It must be placeable (i.e., no conflicts with crossing entries)
        5. The highest-confidence candidate remaining is selected (after applying 
           a penalty to the confidence if the candidate has been previously 
           backtracked)

        NOTE: get_candidates() will call the LLM to regenerate candidates if the 
        entry has a previously-unseen pattern.  The effect is that every time we 
        place an entry, on the NEXT pass through the puzzle we'll call the LLM 
        roughly once for every letter in that entry, because each of its crossing 
        entries will have a new pattern (except for crossing enries that had
        already been placed).
        """
        candidates = entry.get_candidates(widen_search)

        best_candidate = None
        best_effective_confidence = float("-inf")

        for candidate in candidates:
            if candidate.answer in attempted_candidates:
                continue
            attempted_candidates.add(candidate.answer)
            if len(candidate.answer) != entry.length:
                continue
            if candidate.confidence < BasicStrategy.MIN_CANDIDATE_CONFIDENCE_THRESHOLD:
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
    def select_best_backtrack_target(grid: Grid) -> str | None:
        """Select a backtrack target by finding the placed entry most likely to be
        blocking progress on unfilled entries.
        
        Algorithm:
        For each unplaced entry (the "stuck" entry):
          - If it has no candidates, skip it -- it's hopeless and backtracking won't help
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

        # Loop through stuck (unplaced) entries and blame their crossing entries
        for entry in entries:
            if entry.completed:
                continue

            candidates: list[Candidate] = entry.get_candidates()
            if not candidates:
                continue

            crossing_entry_ids = grid.get_crossing_entry_ids(entry.entry_id)

            #if not candidates:
                # No candidates at all for the stuck entry.  This could be because
                # we're too dumb to think of an answer for the clue, but it could also
                # be because a crossing entry prevented us from  Blame all placed 
                # crossing entries, weighted by inverse confidence so shakier
                # entries get more blame.
            #    for crossing_entry_id in crossing_entry_ids:
            #        crossing_entry = grid.entries[crossing_entry_id]
            #        if crossing_entry.placement is None or crossing_entry.used_fallback:
            #            continue
            #        blame[crossing_entry_id] = blame.get(crossing_entry_id, 0.0) + (1.0 - crossing_entry.placement.confidence)
            #    continue

            # more weight when fewer candidates exist
            scarcity_weight = 1.0 + (1.0 / len(candidates))

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

        # If nothing could be blamed, we can't select a backtrack target.
        if not blame:
            logger.debug(f"BACKTRACK TARGET NOT SELECTED: Couldn't assign blame")
            return None

        # Find the maximum blame value and the entry(ies) that have it
        max_blame = max(blame.values())
        tied_ids = [eid for eid, val in blame.items() if val == max_blame]
        tied_entries = [entry for entry in entries if entry.entry_id in tied_ids]

        # Break ties by lowest confidence score
        selected = min(tied_entries, key=lambda rec: rec.placement.confidence if rec.placement is not None else float("inf"))

        logger.debug(
            f"BACKTRACK TARGET SELECTED: entry_id={selected.entry_id} "
            f"blame={max_blame:.2f} "
            + (f"confidence={selected.placement.confidence:.2f}" if selected.placement is not None else "confidence=None")
        )

        return selected.entry_id


    @staticmethod
    def select_best_backtrack_target_old(grid: Grid) -> str | None:
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
            crossing_entry_ids = grid.get_crossing_entry_ids(entry.entry_id)
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
    def select_best_fallback_target(grid: Grid) -> Entry | None:
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
            crossing_entry_ids = grid.get_crossing_entry_ids(entry.entry_id)
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
