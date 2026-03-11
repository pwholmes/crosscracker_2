import logging
from config import (
    HEURISTICS_FALLBACK_UNFILLED_RATIO_WEIGHT,
    HEURISTICS_SELECTION_COMPLETENESS_WEIGHT,
    HEURISTICS_SELECTION_CONFIDENCE_WEIGHT,
    HEURISTICS_SELECTION_CONSTRAINT_WEIGHT,
    HEURISTICS_SELECTION_LENGTH_WEIGHT,
)
from model import Grid, Entry, Candidate

class BasicStrategy:

    @staticmethod
    def select_best_unfilled_entry(grid: Grid, attempted_entries: set[tuple[str, str]], blacklist: dict[Candidate, str] | None = None) -> tuple[Entry, Candidate, float] | None:
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

            cand = BasicStrategy.select_best_candidate(entry, attempted_candidates, blacklist=blacklist)
            if cand is None:
                continue

            # Calculate completeness bonus
            blank_count = entry.pattern.count(".")
            filled_count = entry.length - blank_count
            completeness = filled_count / entry.length

            confidence_score = HEURISTICS_SELECTION_CONFIDENCE_WEIGHT * cand.confidence
            # This length score favors longer answers
            length_score = HEURISTICS_SELECTION_LENGTH_WEIGHT * min(entry.length,10)/10 * 100
            # This length score favors shorter answers
            #length_score = LENGTH_WEIGHT * max(0, 100 - 100/9 * (entry.length - 1))
            completeness_score = HEURISTICS_SELECTION_COMPLETENESS_WEIGHT * completeness * 100
            constraint_weight = HEURISTICS_SELECTION_CONSTRAINT_WEIGHT * (100 / max(1, entry.num_candidates(True)))
            score =  confidence_score + length_score + completeness_score + constraint_weight

            if best_entry is None or score > best_score:
                best_entry = entry
                best_candidate = cand
                best_score = score

        if best_entry is None or best_candidate is None:
            return None
        return best_entry, best_candidate, best_score


    @staticmethod
    def select_best_candidate(entry: Entry, attempted_candidates: set[str], widen_search: bool = False, blacklist: dict[Candidate, str] | None = None) -> Candidate | None:
        """
        Select the best candidate for an entry:
        1. It must not be in the blacklist (entries that failed previously)
        2. It must be one we haven't already tried in this pass through the puzzle
        3. It must be the correct length
        4. It must have a certain minimum confidence level
        5. It must be placeable (i.e., no conflicts with crossing entries)
        6. The highest-confidence candidate remaining is selected (after applying 
           a penalty to the confidence if the candidate has been previously 
           backtracked)

        NOTE: get_candidates() will call the LLM to regenerate candidates if the 
        entry has a previously-unseen pattern.  The effect is that every time we 
        place an entry, on the NEXT pass through the puzzle we'll call the LLM 
        roughly once for every letter in that entry, because each of its crossing 
        entries will have a new pattern (except for crossing enries that had
        already been placed).
        """
        candidates = entry.get_candidates(entry.pattern, widen_search)

        best_candidate = None
        best_confidence = float("-inf")

        for candidate in candidates:
            # Pattern-aware blacklist: skip only if the entry's pattern hasn't
            # changed since the candidate was blacklisted.  A pattern change
            # means a crossing entry was placed or removed, so the candidate
            # deserves another chance.
            if blacklist is not None and candidate in blacklist:
                if blacklist[candidate] == entry.pattern:
                    continue
            if candidate.answer in attempted_candidates:
                continue
            attempted_candidates.add(candidate.answer)
            if len(candidate.answer) != entry.length:
                continue
            if candidate.confidence < Candidate.MIN_SELECTABLE_CONFIDENCE:
                continue
            if not entry.can_place_answer(candidate.answer):
                continue

            # Calculate effective confidence (confidence minus penalty)
            if candidate.confidence > best_confidence:
                best_candidate = candidate
                best_confidence = candidate.confidence

        return best_candidate


    @staticmethod
    def select_best_backtrack_target(grid: Grid) -> str | None:
        """Select a backtrack target by finding the placed entry most likely to be
        blocking progress on unfilled entries.
        
        Algorithm:
        For each unplaced entry (the "stuck" entry):
          - If it has no candidates, try generating some.  If we can't generate any,
            skip it -- it's hopeless and backtracking won't help
          - Otherwise, examine its top N candidates
          - For each placed crossing entry, determine what letter it contributes
            at the shared cell
          - Count how many of the stuck entry's top N candidates conflict with
            that crossing letter
          - Award blame to the crossing entry proportional to the
            confidence-weighted conflict ratio, scaled inversely by
            the crossing entry's placement confidence
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

            crossing_entry_ids = grid.get_crossing_entry_ids(entry.entry_id)

            candidates: list[Candidate] = entry.get_candidates(matching_only=False)
            if not candidates:
                logger.debug(f"Entry {entry.entry_id} has zero candidates - high priority for backtracking analysis")
                            
                # Try regenerating candidates for the stuck entry using a pattern
                # with each (potentially invalid) non-fallback crossing entry 
                # removed in turn.
                for crossing_entry_id in crossing_entry_ids:
                    crossing_entry = grid.entries[crossing_entry_id]
                    if crossing_entry.placement is None or crossing_entry.used_fallback:
                        continue

                    cross_position, crossing_letter = entry.get_crossing_letter(crossing_entry)
                    if cross_position is None or crossing_letter is None:
                        continue

                    # Remove this crossing entry from the pattern
                    pattern_list = list(entry.pattern)
                    pattern_list[cross_position] ='.'
                    pattern = ''.join(pattern_list)

                    if entry.get_candidates(pattern=pattern, matching_only=True):
                        logger.debug(f"Removing crossing entry {crossing_entry_id} helped {entry.entry_id} generate new candidates!")

                candidates = entry.get_candidates(matching_only=False)
                if not candidates:
                    logger.debug(f"No viable candidates could be generated for entry {entry.entry_id} even when removing crossing entries; skipping for backtrack analysis and leaving for potential fallback")
                    continue

            for crossing_entry_id in crossing_entry_ids:
                crossing_entry = grid.entries[crossing_entry_id]
                if crossing_entry.placement is None:
                    #logger.debug(f"Entry {entry.entry_id} not placed, so can't be blamed; skipping")
                    continue
                if crossing_entry.used_fallback:
                    #logger.debug(f"Entry {entry.entry_id} was a fallback and MUST be correct, so can't be blamed; skipping")
                    continue

                # What letter is the crossing entry contributing at the shared cell?
                cross_position, crossing_letter = entry.get_crossing_letter(crossing_entry)
                if cross_position is None or crossing_letter is None:
                    continue

                # Compute confidence-weighted blame.
                # For each candidate that conflicts with this crossing letter,
                # weight the conflict by the blocked candidate's confidence
                # (high-confidence candidates being blocked = stronger signal).
                # Then scale inversely by the blocker's placement confidence
                # (low-confidence placements deserve more blame).
                blocker_confidence = crossing_entry.placement.candidate.confidence
                blocker_weight = 100.0 / max(blocker_confidence, 1.0)

                weighted_conflicts = sum(
                    candidate.confidence
                    for candidate in candidates
                    if candidate.answer[cross_position] != crossing_letter
                )
                total_confidence = sum(c.confidence for c in candidates)
                if total_confidence <= 0:
                    continue
                conflict_ratio = weighted_conflicts / total_confidence

                blame[crossing_entry_id] = blame.get(crossing_entry_id, 0.0) + conflict_ratio * blocker_weight

        # If nothing could be blamed, we can't select a backtrack target.
        if not blame:
            logger.debug(f"BACKTRACK TARGET NOT SELECTED: Couldn't assign blame")
            return None

        # Log the blame
        for eid, val in blame.items():
            logger.debug(f"SELECT BACKTRACK BLAME: Entry {eid}, blame {val}")

        # Find the maximum blame value and the entry(ies) that have it
        max_blame = max(blame.values())
        tied_ids = [eid for eid, val in blame.items() if val == max_blame]
        tied_entries = [entry for entry in entries if entry.entry_id in tied_ids]

        # Break ties by lowest confidence score
        selected = min(tied_entries, key=lambda rec: rec.placement.candidate.confidence if rec.placement is not None else float("inf"))

        logger.debug(
            f"BACKTRACK TARGET SELECTED: entry_id={selected.entry_id} "
            f"blame={max_blame:.2f} "
            + (f"confidence={selected.placement.candidate.confidence:.2f}" if selected.placement is not None else "confidence=None")
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
        selected = min(tied_entries, key=lambda rec: rec.placement.candidate.confidence if rec.placement is not None else float("inf"))
        logger.debug(
            f"BACKTRACK TARGET SELECTED: entry_id={selected.entry_id} "
            f"points={max_points} "
            f"confidence={selected.placement.candidate.confidence:.2f}" if selected.placement is not None else "confidence=None"
        )

        return selected.entry_id

    @staticmethod
    def select_best_fallback_target(grid: Grid) -> Entry | None:
        """Select the best unplaced entry to assign a fallback answer to.
        
        Fallback targets are entries that should receive a default/heuristic-based
        answer when the normal candidate generation fails or produces no viable options.
        This function selects which unplaced entry would be most beneficial to fill,
        helping break deadlocks in the solve process.
        
        The selection heuristic prioritizes entries that:
        - Have many unfilled crossing entries (high unfilled_count)
        - Have few filled crossing entries (low filled_count)
        
        This strategy targets "blocking" entries: entries whose placement would free up
        the most stuck crossing entries, maximizing the chance that fallback placement
        will unblock forward progress.
        
        Algorithm:
        1. Iterate through all unplaced entries
        2. For each unplaced entry, count its filled vs unfilled crossing entries
        3. Skip entries with no unfilled crossings (they don't block anything)
        4. Score each entry based on the ratio of unfilled to filled crossing entries
        5. Return the entry with the highest score
        
        :param grid: The crossword Grid to analyze
        :return: The Entry that should receive a fallback answer, or None if no
                 unplaced entries exist or all unplaced entries have no unfilled crossings
        """
        # These weights should sum to 1
        #W_CONFIDENCE = 0.3
        #W_RETRY = 0.2

        best_entry = None
        best_score = float("-inf")

        entries = grid.entries.values()
        for entry in entries:
            if entry.placement is not None:
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
            score = HEURISTICS_FALLBACK_UNFILLED_RATIO_WEIGHT * unfilled_count / (filled_count + 1)
            if score > best_score:
                best_score = score
                best_entry = entry

        return best_entry
