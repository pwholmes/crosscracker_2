from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import asyncio
import logging
from .llm import LLM
from .model import Candidate, CandidateCache, Entry, Grid, ScoredCandidate

logger = logging.getLogger("src.solver")

@dataclass
class AttemptState:
    current_width: int = 0
    generated_pattern: str = ""
    candidates: list[ScoredCandidate] | None = None
    next_index: int = 0


@dataclass
class PlacedRecord:
    entry_id: str
    answer: str
    width_used: int
    pattern_at_placement: str
    confidence_at_placement: float
    score_at_placement: float
    is_fallback: bool = False


class Solver:
    """A solver that can be driven step-by-step.

    Strategy:
    - Pre-generate candidates once for every clue at width 0 using an empty pattern.
    - In each pass, repeatedly pick the unfilled entry with the highest-confidence
      *placeable* candidate and place it.
    - If an entry has no fitting candidates at its current width, widen for that entry
      (only) and regenerate using the current pattern.
    - Only after a full pass makes no placements do we backtrack (heuristic), and if
      needed, apply fallbacks (removing conflicting non-fallback entries).
    """

    # Backtracks for an entry before forcing its fallback
    MAX_BACKTRACKS_BEFORE_FALLBACK: int = 3

    def __init__(self, grid: Grid, *, defer_candidate_init: bool = False, record: bool = False):
        self.grid = grid
        self.entries = grid.entries
        self.cache = CandidateCache()
        self._attempts: dict[str, AttemptState] = {eid: AttemptState() for eid in self.entries}
        self._placed: dict[str, PlacedRecord] = {}
        self._penalties: dict[tuple[str, str, int], dict[str, float]] = {}
        self._entry_backtracks: dict[str, int] = {}
        self._attempted_this_pass: set[str] = set()
        self._stall_passes: int = 0
        self._recording: list[dict[str, Any]] | None = [] if record else None
        # Initialize candidates for all entries at width 0 with empty pattern
        if not defer_candidate_init:
            self._initialize_candidates_at_width(0)

    def record_event(self, event: dict[str, Any]) -> None:
        """Record an event to the recording if recording is enabled.
        
        This is the single point through which all events flow for recording.
        Separated from broadcasting so the server can handle both in a unified way.
        """
        if self._recording is not None:
            self._recording.append(event.copy())

    def _initialize_entry_candidates(self, entry_id: str, widening_level: int) -> None:
        """Initialize (or reinitialize) candidates for a single entry at a given widening level."""
        entry = self.entries[entry_id]
        attempt = self._attempts[entry_id]
        attempt.current_width = widening_level
        attempt.generated_pattern = "." * entry.length
        attempt.candidates = self._get_candidates(entry_id, attempt.generated_pattern, widening_level)
        attempt.next_index = 0

    def _initialize_candidates_at_width(self, widening_level: int) -> None:
        """Initialize (or reinitialize) candidates for all entries at a given widening level."""
        for eid in self.entries.keys():
            self._initialize_entry_candidates(eid, widening_level)

    async def async_initialize_with_progress(
        self, progress_callback: Callable[[int, int], Any]
    ) -> None:
        """Async reinitialization with progress callback for UI feedback.
        
        This is optional and mainly used by the server to show progress to connected clients.
        The Solver is already fully initialized after __init__, so this reinitializes if needed.
        """
        total = len(self.entries)
        for idx, eid in enumerate(self.entries.keys(), 1):
            self._initialize_entry_candidates(eid, 0)
            result = progress_callback(idx, total)
            if asyncio.iscoroutine(result):
                await result
            await asyncio.sleep(0)

    def solve(self) -> bool:
        """Run to completion (blocking)."""
        while True:
            ev = self.step()
            if ev.get("event") == "solved":
                return True
            if ev.get("event") == "failed":
                return False

    def step(self) -> dict[str, Any]:
        """Perform exactly one solver action and return an event dict."""
        # Check if grid is completely filled first
        if self._all_filled():
            return self._finalize_event({"event": "solved"}, [])

        while True:
            # Select the best unfilled entry
            selection = self._select_best_unfilled_entry()
            if selection is None:
                return self._handle_stall([])
            entry_id, selection_score = selection

            # Mark this entry as having been attempted in this pass
            self._attempted_this_pass.add(entry_id)

            # Get the best candidate answer for this entry            
            scored_candidate = self._select_best_candidate(entry_id)
            if scored_candidate is None:
                continue

            # Check if any crossing entries would be completed by the placement of this answer
            crossing_entries = self._predict_crossing_entries(entry_id, scored_candidate.answer)
            if crossing_entries:
                logger.debug(
                    f"VERIFY: Checking crossing entries for {entry_id}='{scored_candidate.answer}': "
                    f"{list(crossing_entries.keys())}"
                )
            # Verify the crossing entries
            verified_entry_ids, failed_entry_ids = self.verify_answers(crossing_entries)
            if failed_entry_ids:
                # Reject this candidate and continue trying others
                self._reject_candidate(entry_id, scored_candidate.answer)
                continue
            
            # Verification passed - now actually place the entry
            candidate = Candidate(entry_id=entry_id, answer=scored_candidate.answer, widening_level=0)
            self.grid.place_candidate(candidate)
            self.entries[entry_id].verified = True
            self._record_placement(
                entry_id=entry_id,
                answer=candidate.answer,
                width_used=0,
                pattern_at_placement=self.entries[entry_id].pattern,
                confidence=1.0,
                score=selection_score,
                is_fallback=False,
            )

            # Mark crossing entries as verified now that placement is committed
            for verified_entry_id in verified_entry_ids:
                self.entries[verified_entry_id].verified = True
            
            self._attempted_this_pass.clear()
            self._stall_passes = 0

            rec = self._placed.get(candidate.entry_id)
            confidence = rec.confidence_at_placement if rec is not None else None
            pattern_at_placement = rec.pattern_at_placement if rec is not None else None
            score_at_placement = rec.score_at_placement if rec is not None else None
            logger.debug(
                f"PLACED entry={entry_id} answer='{candidate.answer}' "
                f"confidence={f'{confidence:.2f}' if confidence is not None else 'N/A'} "
                f"score={f'{score_at_placement:.2f}' if score_at_placement is not None else 'N/A'} "
                f"pattern={pattern_at_placement} widening_level={candidate.widening_level}"
            )
            return self._finalize_event(
                {
                    "event": "placed",
                    "candidate": {
                        "entry_id": candidate.entry_id,
                        "answer": candidate.answer,
                        "widening_level": candidate.widening_level,
                        "confidence": confidence,
                        "score": score_at_placement,
                        "pattern": pattern_at_placement,
                    },
                },
                verified_entry_ids,
            )

    def _select_best_unfilled_entry(self) -> tuple[str, float] | None:
        """Select the best unfilled entry using a heuristic that balances:
        1. Candidate confidence
        2. Entry completeness (filled letters)
        3. Entry length (longer entries preferred)
        """
        # These should sum to 1.0
        CONFIDENCE_WEIGHT = 0.3
        LENGTH_WEIGHT = 0.2
        COMPLETENESS_WEIGHT = 0.5

        best_entry_id: str | None = None
        best_score: float = float("-inf")

        for eid, entry in self.entries.items():
            if "." not in entry.pattern:
                continue
            if eid in self._attempted_this_pass:
                continue

            cand = self._select_best_candidate(eid)
            if cand is None:
                continue

            # Get penalty for this candidate if it was backtracked
            attempt = self._attempts[eid]
            key = (eid, attempt.generated_pattern, attempt.current_width)
            penalties = self._penalties.get(key, {})
            penalty = penalties.get(cand.answer, 0.0)
            
            # Calculate completeness bonus
            blank_count = entry.pattern.count(".")
            filled_count = entry.length - blank_count
            completeness = filled_count / entry.length
            
            # Total score = confidence (0-100) + completeness bonus - penalty
            score = CONFIDENCE_WEIGHT * (cand.confidence - penalty) + \
                    LENGTH_WEIGHT * min(entry.length,10)/10 * 100 + \
                    COMPLETENESS_WEIGHT * completeness * 100
            
            if best_entry_id is None or score > best_score:
                best_entry_id = eid
                best_score = score

        if best_entry_id is None:
            return None
        return best_entry_id, best_score

    def _select_best_candidate(self, entry_id: str) -> ScoredCandidate | None:
        entry = self.entries[entry_id]
        attempt = self._attempts[entry_id]
        pattern = entry.pattern

        if attempt.candidates is None or attempt.generated_pattern != pattern:
            attempt.generated_pattern = pattern
            attempt.candidates = self._get_candidates(entry_id, pattern, attempt.current_width)
            attempt.next_index = 0

        key = (entry_id, attempt.generated_pattern, attempt.current_width)
        penalties = self._penalties.get(key, {})

        best_cand = None
        best_effective_confidence = float("-inf")

        for cand in attempt.candidates[attempt.next_index :]:
            if len(cand.answer) != entry.length:
                continue
            if not self._can_place(entry, cand.answer):
                continue
            
            # Calculate effective confidence (confidence minus penalty)
            penalty = penalties.get(cand.answer, 0.0)
            effective_confidence = cand.confidence - penalty
            
            if effective_confidence > best_effective_confidence:
                best_cand = cand
                best_effective_confidence = effective_confidence
        
        return best_cand

    # Old _place_entry logic (commented out for reference)
    # def _place_entry(self, entry_id: str, selection_score: float) -> Candidate | None:
    #     entry = self.entries[entry_id]
    #     attempt = self._attempts[entry_id]
    #     while attempt.current_width <= LLM.MAX_WIDENING:
    #         pattern = entry.pattern
    #         if attempt.candidates is None or attempt.generated_pattern != pattern:
    #             attempt.generated_pattern = pattern
    #             attempt.candidates = self._get_candidates(entry_id, pattern, attempt.current_width)
    #             attempt.next_index = 0
    #         key = (entry_id, attempt.generated_pattern, attempt.current_width)
    #         penalties = self._penalties.setdefault(key, {})
    #         while attempt.next_index < len(attempt.candidates):
    #             cand = attempt.candidates[attempt.next_index]
    #             attempt.next_index += 1
    #             if len(cand.answer) != entry.length:
    #                 penalties[cand.answer] = penalties.get(cand.answer, 0.0) + 10.0
    #                 continue
    #             candidate = Candidate(entry_id=entry_id, answer=cand.answer, widening_level=attempt.current_width)
    #             if not self.grid.place_candidate(candidate):
    #                 penalties[cand.answer] = penalties.get(cand.answer, 0.0) + 10.0
    #                 continue
    #             self.entries[entry_id].verified = True
    #             self._record_placement(
    #                 entry_id=entry_id,
    #                 answer=cand.answer,
    #                 width_used=attempt.current_width,
    #                 pattern_at_placement=pattern,
    #                 confidence=cand.confidence,
    #                 score=selection_score,
    #                 is_fallback=False,
    #             )
    #             return candidate
    #         attempt.current_width += 1
    #         attempt.candidates = None
    #         attempt.next_index = 0
    #     return None

    def _try_apply_fallback_and_create_event(
        self,
        newly_verified: list[str],
        entry: Entry | None = None,
        extra_event_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Apply a fallback (for a specific entry or any entry) and create the event.
        
        Returns the placed_fallback event dict if successful, None otherwise.
        """
        fallback = self._apply_fallback_with_conflict_removal(entry)
        if fallback is None:
            return None
        
        rec, conflicts_removed = fallback
        event: dict[str, Any] = {
            "event": "placed_fallback",
            "candidate": {
                "entry_id": rec.entry_id,
                "answer": rec.answer,
                "widening_level": 0,
                "confidence": rec.confidence_at_placement,
                "pattern": rec.pattern_at_placement,
            },
            "conflicts_removed": conflicts_removed,
        }
        
        # Merge in any extra fields (e.g., verification_failed)
        if extra_event_fields:
            event.update(extra_event_fields)
        
        return self._finalize_event(event, newly_verified)

    def _handle_stall(self, newly_verified: list[str]) -> dict[str, Any]:
        """Handle a stall: no placements were made in this pass.
        
        Strategy:
        1. Try to backtrack a placed entry to open up new possibilities
        2. If that entry has been backtracked MAX_BACKTRACKS_BEFORE_FALLBACK times, 
           apply its fallback instead of continuing to thrash
        3. If no backtrack target exists, try to apply any available fallback
        4. If no fallback possible, puzzle has failed
        """
        self._stall_passes += 1
        self._attempted_this_pass.clear()

        # Try to select a backtrack target
        target = self._choose_backtrack_target()
        if target is not None:
            removed = self._remove_placed(target)
            if removed is not None:
                removed_candidate, removed_record = removed
                self._record_backtrack(removed_candidate.entry_id)
                
                # Check if this entry has been backtracked too many times
                backtrack_count = self._entry_backtracks.get(removed_candidate.entry_id, 0)
                if backtrack_count >= self.MAX_BACKTRACKS_BEFORE_FALLBACK:
                    # Force fallback for this thrashing entry
                    entry = self.entries.get(removed_candidate.entry_id)
                    if entry is not None and entry.correct_answer:
                        logger.debug(
                            f"FORCING FALLBACK: entry_id={removed_candidate.entry_id} "
                            f"backtrack_count={backtrack_count}"
                        )
                        fallback_event = self._try_apply_fallback_and_create_event(newly_verified, entry)
                        if fallback_event is not None:
                            return fallback_event
                
                # Normal backtrack - return backtrack event
                score_at_placement = removed_record.score_at_placement
                logger.debug(
                    f"BACKTRACK: entry_id={removed_candidate.entry_id} answer={removed_candidate.answer} "
                    f"confidence={removed_record.confidence_at_placement:.2f} "
                    f"score={score_at_placement:.2f} "
                    f"pattern={removed_record.pattern_at_placement} widening_level={removed_candidate.widening_level}"
                )
                return self._finalize_event(
                    {
                        "event": "backtrack",
                        "candidate": {
                            "entry_id": removed_candidate.entry_id,
                            "answer": removed_candidate.answer,
                            "widening_level": removed_candidate.widening_level,
                            "confidence": removed_record.confidence_at_placement,
                            "score": score_at_placement,
                            "pattern": removed_record.pattern_at_placement,
                        },
                    },
                    newly_verified,
                )

        # No backtrack target available - try to apply any fallback
        fallback_event = self._try_apply_fallback_and_create_event(newly_verified)
        if fallback_event is not None:
            return fallback_event

        # No backtrack and no fallback possible - puzzle has failed
        return self._finalize_event({"event": "failed"}, newly_verified)

    def _choose_backtrack_target(self) -> str | None:
        """Select a backtrack target by finding the entry that appears most frequently
        as a crossing to unfilled entries.
        
        Algorithm:
        1. For each unplaced entry, identify all placed crossing entries
        2. Award each placed crossing entry a point
        3. Select the placed entry with the most points
        4. Break ties by lowest confidence score
        """
        candidates = [rec for rec in self._placed.values() if not rec.is_fallback]
        if not candidates:
            return None
        
        # Count points for each placed entry
        points: dict[str, int] = {rec.entry_id: 0 for rec in candidates}
        
        # Loop through unplaced entries
        for entry_id, entry in self.entries.items():
            # Skip if this entry is already placed
            if "." not in entry.pattern:
                continue
            
            # Find all placed entries that cross this unplaced entry
            for cell in entry.cells:
                # cell.sources contains the entry IDs that pass through this cell
                if cell.sources:
                    for crossing_id in cell.sources:
                        if crossing_id != entry_id and crossing_id in points:
                            # This crossing entry is placed, give it a point
                            points[crossing_id] += 1
        
        # Find the entry with the most points
        max_points = max(points.values())
        tied_entries = [rec for rec in candidates if points[rec.entry_id] == max_points]
        
        if not tied_entries:
            return None
        
        # Break ties by lowest confidence score
        selected = min(tied_entries, key=lambda rec: rec.confidence_at_placement)
        logger.debug(
            f"BACKTRACK TARGET SELECTED: entry_id={selected.entry_id} "
            f"points={max_points} confidence={selected.confidence_at_placement:.2f}"
        )
        return selected.entry_id

    def _record_placement(
        self,
        entry_id: str,
        answer: str,
        width_used: int,
        pattern_at_placement: str,
        confidence: float,
        score: float,
        is_fallback: bool,
    ) -> None:
        self._placed[entry_id] = PlacedRecord(
            entry_id=entry_id,
            answer=answer,
            width_used=width_used,
            pattern_at_placement=pattern_at_placement,
            confidence_at_placement=confidence,
            score_at_placement=score,
            is_fallback=is_fallback,
        )

    def _record_backtrack(self, entry_id: str) -> None:
        self._entry_backtracks[entry_id] = self._entry_backtracks.get(entry_id, 0) + 1

    def _remove_placed(self, entry_id: str) -> tuple[Candidate, PlacedRecord] | None:
        rec = self._placed.get(entry_id)
        if rec is None:
            return None

        # Collect crossing entries before removing
        crossing_entries = self.get_crossing_entry_ids(entry_id)
        logger.debug(f"BACKTRACK: {len(crossing_entries)} crossing entries detected for {entry_id}")

        # Remove the answer from the grid and from the Solver's list of placed entries
        candidate = Candidate(entry_id=rec.entry_id, answer=rec.answer, widening_level=rec.width_used)
        self.grid.remove_candidate(candidate)
        self._placed.pop(entry_id, None)

        # Apply a penalty to this answer so it is less likely (but not impossible!) to use again
        key = (entry_id, rec.pattern_at_placement, rec.width_used)
        self._penalties.setdefault(key, {})[rec.answer] = self._penalties.get(key, {}).get(rec.answer, 0.0) + 20.0

        # Reset the entry's verified flag
        self.entries[entry_id].verified = False

        # For each crossing entry not explicitly placed, unverify and regenerate candidates
        for crossing_id in crossing_entries:
            if crossing_id in self._placed:
                logger.debug(f"BACKTRACK: Crossing entry {crossing_id} was explicitly placed, not affected by backtrack.")
                continue
            crossing_entry = self.entries.get(crossing_id)
            assert crossing_entry is not None, "Invalid crossing entry ID " + crossing_id
            crossing_entry.verified = False
            crossing_attempt = self._attempts[crossing_id]
            crossing_attempt.generated_pattern = crossing_entry.pattern
            crossing_attempt.candidates = self._get_candidates(crossing_id, crossing_entry.pattern, crossing_attempt.current_width)
            crossing_attempt.next_index = 0
            logger.debug(f"BACKTRACK: Unverified and regenerated candidates for crossing entry {crossing_id} (pattern now '{crossing_entry.pattern}')")


        attempt = self._attempts[entry_id]
        attempt.current_width = 0
        attempt.generated_pattern = ""
        attempt.candidates = None
        attempt.next_index = 0

        return candidate, rec

    def _apply_fallback_with_conflict_removal(self, entry: Entry | None = None) -> tuple[PlacedRecord, list[str]] | None:
        if entry is None:
            entry = self._select_fallback_entry()
        if entry is None:
            return None

        eid = entry.entry_id
        answer = entry.correct_answer
        if len(answer) != entry.length:
            return None

        # Remove conflicting non-fallback placements until this fallback fits.
        removed_entries: list[str] = []
        while True:
            conflicting: set[str] = set()
            for cell, ch in zip(entry.cells, answer):
                if cell.letter is not None and cell.letter != ch:
                    conflicting.update(cell.sources)
            conflicting.discard(eid)

            to_remove = [c for c in conflicting if c in self._placed and not self._placed[c].is_fallback]
            if not to_remove:
                break
            for c in to_remove:
                if self._remove_placed(c) is not None:
                    removed_entries.append(c)

        candidate = Candidate(eid, answer, widening_level=0, is_fallback=True)
        if not self.grid.place_candidate(candidate):
            return None

        entry.used_fallback = True
        entry.verified = True
        self._record_placement(
            entry_id=eid,
            answer=answer,
            width_used=0,
            pattern_at_placement=entry.pattern,
            confidence=1.0,
            score=1.0,
            is_fallback=True,
        )
        return self._placed[eid], removed_entries

    def _select_fallback_entry(self, *, threshold_only: bool = False) -> Entry | None:
        entries: list[Entry] = [e for e in self.entries.values() if "." in e.pattern and e.correct_answer]
        if threshold_only:
            entries = [
                e for e in entries
                if self._entry_backtracks.get(e.entry_id, 0) >= self.MAX_BACKTRACKS_BEFORE_FALLBACK
            ]
        if not entries:
            return None
        if threshold_only:
            return max(
                entries,
                key=lambda e: (
                    self._entry_backtracks.get(e.entry_id, 0),
                    e.pattern.count("."),
                ),
            )
        return max(entries, key=lambda e: e.pattern.count("."))

    def _all_filled(self) -> bool:
        return all("." not in e.pattern for e in self.entries.values())

    def _can_place(self, entry: Entry, answer: str) -> bool:
        for cell, ch in zip(entry.cells, answer):
            if cell.letter is not None and cell.letter != ch:
                return False
        return True

    def _get_candidates(self, entry_id: str, pattern: str, widening_level: int) -> list[ScoredCandidate]:
        cached = self.cache.get(entry_id, pattern, widening_level)
        if cached is None:
            entry = self.entries[entry_id]
            candidates = LLM.generate_candidates(entry, widening_level)
            self.cache.put(entry_id, pattern, widening_level, candidates)
            return candidates
        return cached

    def _predict_crossing_entries(self, entry_id: str, answer: str) -> dict[str, str]:
        """Compute hypothetical patterns for crossing entries if we placed this answer.
        
        Returns a dict mapping entry_id -> hypothetical_pattern for entries that would
        become complete after placement.
        """
        entry = self.entries[entry_id]
        crossing_patterns: dict[str, str] = {}
        seen_crossings: set[str] = set()
        
        # For each cell in the entry, find crossing entries
        for _, (cell, ch) in enumerate(zip(entry.cells, answer)):
            for crossing_id in self.get_crossing_entry_ids(entry_id):
                if crossing_id in seen_crossings:
                    continue
                crossing_entry = self.entries[crossing_id]
                if cell not in crossing_entry.cells:
                    continue
                seen_crossings.add(crossing_id)
                # Skip if already explicitly placed
                if crossing_id in self._placed:
                    logger.debug(
                        f"VERIFY SKIP: {crossing_id} already placed (crossing {entry_id})"
                    )
                    continue
                # Compute what the pattern would be after placing
                hypothetical_pattern = list(crossing_entry.pattern)
                crossing_cell_idx = crossing_entry.cells.index(cell)
                hypothetical_pattern[crossing_cell_idx] = ch
                pattern_str = "".join(hypothetical_pattern)
                # Only include if it would be complete
                if "." not in pattern_str:
                    crossing_patterns[crossing_id] = pattern_str
                    logger.debug(
                        f"VERIFY CANDIDATE: {crossing_id} would be complete: '{pattern_str}' "
                        f"(crossing {entry_id}='{answer}' at cell {cell.row},{cell.col})"
                    )
                else:
                    logger.debug(
                        f"VERIFY SKIP: {crossing_id} would be incomplete: '{pattern_str}' "
                        f"(crossing {entry_id}='{answer}')"
                    )
        return crossing_patterns

    def verify_answers(self, answers: dict[str, str]) -> tuple[list[str], list[str]]:
        """Verify answers.
        Returns (newly_verified, failed) lists of entry IDs.
        """
        if not answers:
            return [], []
            
        verified_entry_ids: list[str] = []
        failed_entry_ids: list[str] = []
        
        for entry_id, answer in answers.items():
            entry = self.entries.get(entry_id)
            if entry is None:
                continue
            
            # Skip if already verified
            if entry.verified:
                logger.debug(f"VERIFY SKIP: {entry_id} already verified")
                continue
            
            if LLM.verify_answer(entry, answer):
                # Don't set verified flag yet - we haven't actually placed anything
                verified_entry_ids.append(entry_id)
                logger.debug(f"VERIFY ✓ SUCCESS: {entry_id}='{answer}'")
            else:
                failed_entry_ids.append(entry_id)
                logger.debug(f"VERIFY ✗ FAILED: {entry_id}='{answer}'")
        
        if verified_entry_ids or failed_entry_ids:
            logger.debug(
                f"VERIFY SUMMARY: {len(verified_entry_ids)} passed, {len(failed_entry_ids)} failed "
                f"(passed: {verified_entry_ids}, failed: {failed_entry_ids})"
            )
        
        return verified_entry_ids, failed_entry_ids
    
    def _reject_candidate(self, entry_id: str, answer: str) -> None:
        """Reject a candidate by applying a penalty, preventing it from being selected again."""
        attempt = self._attempts[entry_id]
        key = (entry_id, attempt.generated_pattern, attempt.current_width)
        penalties = self._penalties.setdefault(key, {})
        penalties[answer] = penalties.get(answer, 0.0) + 50.0  # Heavy penalty for verification failure
        logger.debug(f"REJECTED: entry={entry_id} answer={answer} (verification would fail)")

    def _all_entries_verified(self) -> bool:
        return all(e.verified for e in self.entries.values())

    def _finalize_event(self, event: dict[str, Any], newly_verified: list[str]) -> dict[str, Any]:
        if newly_verified:
            event["verified"] = newly_verified
        if event.get("event") == "verified" and not newly_verified:
            event["event"] = "failed"
        self.record_event(event)
        return event


    def get_crossing_entry_ids(self, entry_id: str) -> set[str]:
        """Return a set of entry IDs that cross the given entry (share at least one cell)."""
        entry = self.entries[entry_id]
        crossing_ids: set[str] = set()
        for other_id, other_entry in self.entries.items():
            if other_id == entry_id:
                continue
            if any(cell in other_entry.cells for cell in entry.cells):
                crossing_ids.add(other_id)
        return crossing_ids


    def get_recording(self) -> dict[str, Any] | None:
        """Get the recorded solve session as a dict. Returns None if recording is disabled."""
        if self._recording is None:
            return None
        
        return {
            "puzzle": self.grid.puzzle_id if hasattr(self.grid, 'puzzle_id') else "unknown",
            "width": self.grid.width,
            "height": self.grid.height,
            "events": self._recording,
        }
    
    def save_recording(self, filepath: str) -> bool:
        """Save the recording to a JSON file. Returns True if successful."""
        import json
        
        if self._recording is None:
            logger.warning("Recording is disabled; no data to save")
            return False
        
        try:
            recording = self.get_recording()
            with open(filepath, "w") as f:
                json.dump(recording, f, indent=2)
            logger.info(f"Recording saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save recording: {e}")
            return False
    
    @staticmethod
    def load_recording(filepath: str) -> dict[str, Any] | None:
        """Load a recording from a JSON file. Returns the recording dict or None on error."""
        import json
        
        try:
            with open(filepath, "r") as f:
                recording = json.load(f)
            logger.info(f"Recording loaded from {filepath}")
            return recording
        except Exception as e:
            logger.error(f"Failed to load recording: {e}")
            return None
    
    @staticmethod
    def get_recording_event_at_step(recording: dict[str, Any], step_num: int) -> dict[str, Any] | None:
        """Get the event at the specified step number (0-indexed) from a recording.
        
        Returns the event dict or None if step is out of range.
        """
        events = recording.get("events", [])
        if 0 <= step_num < len(events):
            return events[step_num]
        return None
    
    @staticmethod
    def get_recording_summary(recording: dict[str, Any]) -> str:
        """Get a brief summary of a recording."""
        puzzle = recording.get("puzzle", "unknown")
        width = recording.get("width", "?")
        height = recording.get("height", "?")
        event_count = len(recording.get("events", []))
        
        # Count event types
        events = recording.get("events", [])
        event_types: dict[str, int] = {}
        for event in events:
            event_type = event.get("event", "unknown")
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        return (
            f"Puzzle: {puzzle} ({width}x{height})\n"
            f"Total events: {event_count}\n"
            f"Event types: {event_types}"
        )

