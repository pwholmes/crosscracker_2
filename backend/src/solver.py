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
    placement_count: int
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

    # Maximum times a candidate can be placed before forcing fallback
    MAX_PLACEMENT_ATTEMPTS: int = 3

    def __init__(self, grid: Grid, *, defer_candidate_init: bool = False, record: bool = False):
        self.grid = grid
        self.entries = grid.entries
        self.cache = CandidateCache()
        self._attempts: dict[str, AttemptState] = {eid: AttemptState() for eid in self.entries}
        self._placed: dict[str, PlacedRecord] = {}
        self._placed_order: list[str] = []
        self._penalties: dict[tuple[str, str, int], dict[str, float]] = {}
        self._placement_counts: dict[tuple[str, str], int] = {}
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
        newly_verified, verification_failed = self._mark_verified_entries()
        if verification_failed:
            return self._handle_verification_failure(verification_failed, newly_verified, recently_placed=None)

        if self._all_filled():
            assert self._all_entries_verified(), (
                "All entries should be verified after _mark_verified_entries when the grid is filled."
            )
            return self._finalize_event({"event": "solved"}, newly_verified)

        while True:
            selection = self._select_best_unfilled_entry()
            if selection is None:
                # End of pass.
                return self._handle_stall(newly_verified)

            entry_id, selection_score = selection

            self._attempted_this_pass.add(entry_id)
            placed = self._try_fill_entry(entry_id, selection_score)
            if placed is None:
                continue

            # After explicitly placing an entry, we treat that entry as trusted/verified.
            # Verification is reserved for entries that become complete indirectly
            # (i.e., via crossings) to avoid redundant LLM calls and log spam.
            newly_verified, verification_failed = self._mark_verified_entries(
                newly_verified,
                exclude_entry_ids={placed.entry_id},
            )
            if verification_failed:
                return self._handle_verification_failure(verification_failed, newly_verified, recently_placed=placed.entry_id)

            self._attempted_this_pass.clear()
            self._stall_passes = 0

            rec = self._placed.get(placed.entry_id)
            confidence = rec.confidence_at_placement if rec is not None else None
            pattern_at_placement = rec.pattern_at_placement if rec is not None else None

            score_at_placement = rec.score_at_placement if rec is not None else None
            logger.debug(
                f"PLACED entry={entry_id} answer='{placed.answer}' "
                f"confidence={f'{confidence:.2f}' if confidence is not None else 'N/A'} "
                f"score={f'{score_at_placement:.2f}' if score_at_placement is not None else 'N/A'} "
                f"pattern={pattern_at_placement} widening_level={placed.widening_level}"
            )
            return self._finalize_event(
                {
                    "event": "placed",
                    "candidate": {
                        "entry_id": placed.entry_id,
                        "answer": placed.answer,
                        "widening_level": placed.widening_level,
                        "confidence": confidence,
                        "score": score_at_placement,
                        "pattern": pattern_at_placement,
                    },
                },
                newly_verified,
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

            cand = self._peek_best_fit_candidate(eid)
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

    def _peek_best_fit_candidate(self, entry_id: str) -> ScoredCandidate | None:
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
        
        if best_cand is None and attempt.candidates:
            logger.debug(f"No placeable candidate found for {entry_id}: "
                        f"{len(attempt.candidates)} candidates, pattern={pattern}, "
                        f"entries returned {len([c for c in attempt.candidates if len(c.answer) == entry.length])} "
                        f"with correct length")
        return best_cand

    def _try_fill_entry(self, entry_id: str, selection_score: float) -> Candidate | None:
        entry = self.entries[entry_id]
        attempt = self._attempts[entry_id]

        while attempt.current_width <= LLM.MAX_WIDENING:
            pattern = entry.pattern
            if attempt.candidates is None or attempt.generated_pattern != pattern:
                attempt.generated_pattern = pattern
                attempt.candidates = self._get_candidates(entry_id, pattern, attempt.current_width)
                attempt.next_index = 0

            key = (entry_id, attempt.generated_pattern, attempt.current_width)
            penalties = self._penalties.setdefault(key, {})

            while attempt.next_index < len(attempt.candidates):
                cand = attempt.candidates[attempt.next_index]
                attempt.next_index += 1
                if len(cand.answer) != entry.length:
                    penalties[cand.answer] = penalties.get(cand.answer, 0.0) + 10.0
                    continue

                candidate = Candidate(entry_id=entry_id, answer=cand.answer, widening_level=attempt.current_width)
                
                # Check if this candidate has been placed too many times (threshold to prevent thrashing)
                placement_count = self._placement_counts.get((entry_id, cand.answer), 0)
                if placement_count >= self.MAX_PLACEMENT_ATTEMPTS:
                    penalties[cand.answer] = penalties.get(cand.answer, 0.0) + 50.0
                    continue
                
                if not self.grid.place_candidate(candidate):
                    penalties[cand.answer] = penalties.get(cand.answer, 0.0) + 10.0
                    continue

                # Explicit placements are treated as trusted/verified.
                self.entries[entry_id].verified = True

                self._record_placement(
                    entry_id=entry_id,
                    answer=cand.answer,
                    width_used=attempt.current_width,
                    pattern_at_placement=pattern,
                    confidence=cand.confidence,
                    score=selection_score,
                    is_fallback=False,
                )
                return candidate

            # No candidates fit at this width: widen and regenerate using current pattern.
            attempt.current_width += 1
            attempt.candidates = None
            attempt.next_index = 0

        return None

    def _handle_stall(self, newly_verified: list[str]) -> dict[str, Any]:
        self._stall_passes += 1
        self._attempted_this_pass.clear()

        target = self._choose_backtrack_target()
        if target is not None:
            removed = self._remove_placed(target)
            if removed is not None:
                removed_candidate, removed_record = removed
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

        fallback = self._apply_fallback_with_conflict_removal()
        if fallback is not None:
            rec, conflicts_removed = fallback
            return self._finalize_event(
                {
                    "event": "placed_fallback",
                    "candidate": {
                        "entry_id": rec.entry_id,
                        "answer": rec.answer,
                        "widening_level": 0,
                        "confidence": rec.confidence_at_placement,
                        "pattern": rec.pattern_at_placement,
                    },
                    "conflicts_removed": conflicts_removed,
                },
                newly_verified,
            )

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
        key = (entry_id, answer)
        self._placement_counts[key] = self._placement_counts.get(key, 0) + 1
        placement_count = self._placement_counts[key]
        
        self._placed[entry_id] = PlacedRecord(
            entry_id=entry_id,
            answer=answer,
            width_used=width_used,
            pattern_at_placement=pattern_at_placement,
            confidence_at_placement=confidence,
            score_at_placement=score,
            placement_count=placement_count,
            is_fallback=is_fallback,
        )
        if entry_id in self._placed_order:
            self._placed_order.remove(entry_id)
        self._placed_order.append(entry_id)

    def _remove_placed(self, entry_id: str) -> tuple[Candidate, PlacedRecord] | None:
        rec = self._placed.get(entry_id)
        if rec is None:
            return None

        # Collect crossing entries before removing
        entry = self.entries[entry_id]
        crossing_entries: set[str] = set()
        for cell in entry.cells:
            crossing_entries.update(cell.sources)
        crossing_entries.discard(entry_id)

        candidate = Candidate(entry_id=rec.entry_id, answer=rec.answer, widening_level=rec.width_used)
        self.grid.remove_candidate(candidate)

        key = (entry_id, rec.pattern_at_placement, rec.width_used)
        self._penalties.setdefault(key, {})[rec.answer] = self._penalties.get(key, {}).get(rec.answer, 0.0) + 20.0

        self._placed.pop(entry_id, None)
        if entry_id in self._placed_order:
            self._placed_order.remove(entry_id)

        # Reset verified flag when backtracking
        self.entries[entry_id].verified = False

        # Unverify crossing entries that are now incomplete and were not explicitly placed
        for crossing_id in crossing_entries:
            crossing_entry = self.entries.get(crossing_id)
            if crossing_entry is None:
                continue
            # Only unverify if: (1) now incomplete AND (2) never explicitly placed
            if "." in crossing_entry.pattern and crossing_id not in self._placed:
                crossing_entry.verified = False
                logger.debug(f"BACKTRACK: Unverified crossing entry {crossing_id} (now incomplete)")

        attempt = self._attempts[entry_id]
        attempt.current_width = 0
        attempt.generated_pattern = ""
        attempt.candidates = None
        attempt.next_index = 0

        return candidate, rec

    def _apply_fallback_with_conflict_removal(self) -> tuple[PlacedRecord, list[str]] | None:
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

    def _select_fallback_entry(self) -> Entry | None:
        entries: list[Entry] = [e for e in self.entries.values() if "." in e.pattern]
        if not entries:
            return None
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

    def _mark_verified_entries(
        self,
        existing: list[str] | None = None,
        *,
        exclude_entry_ids: set[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        newly_verified: list[str] = existing or []
        failed: list[str] = []
        exclude = exclude_entry_ids or set()
        for entry in self.entries.values():
            if entry.entry_id in exclude:
                continue
            pattern = entry.pattern
            if "." in pattern:
                if entry.verified:
                    entry.verified = False
                continue
            if entry.verified:
                continue
            logger.debug(f"VERIFY CHECK: entry={entry.entry_id} pattern='{pattern}' length={entry.length}")
            # Defensive assertion: pattern should never have dots at this point
            assert "." not in pattern, (
                f"BUG: Attempting to verify entry {entry.entry_id} with incomplete pattern: '{pattern}'"
            )
            if LLM.verify_answer(entry, pattern):
                entry.verified = True
                newly_verified.append(entry.entry_id)
                logger.debug(f"VERIFY SUCCESS: entry={entry.entry_id} pattern='{pattern}'")
            else:
                failed.append(entry.entry_id)
                logger.debug(f"VERIFY FAILED: entry={entry.entry_id} pattern='{pattern}'")
        return newly_verified, failed

    def _all_entries_verified(self) -> bool:
        return all(e.verified for e in self.entries.values())

    def _finalize_event(self, event: dict[str, Any], newly_verified: list[str]) -> dict[str, Any]:
        if newly_verified:
            event["verified"] = newly_verified
        if event.get("event") == "verified" and not newly_verified:
            event["event"] = "failed"
        self.record_event(event)
        return event

    def _handle_verification_failure(self, failed: list[str], newly_verified: list[str], recently_placed: str | None = None) -> dict[str, Any]:
        # Log the verification failure
        logger.debug(f"VERIFICATION FAILED: entries={failed}")
        
        # Prefer removing a failing entry if it was placed and is not a fallback.
        target: str | None = None
        for eid in failed:
            if eid in self._placed and not self._placed[eid].is_fallback:
                target = eid
                break
        
        # If failing entries were auto-completed (not explicitly placed), 
        # choose among their placed crossings by lowest confidence.
        # Exclude the recently placed entry that triggered this verification.
        if target is None:
            crossing_candidates: dict[str, PlacedRecord] = {}
            for failed_id in failed:
                failed_entry = self.entries.get(failed_id)
                if failed_entry is None:
                    continue
                for cell in failed_entry.cells:
                    if not cell.sources:
                        continue
                    for crossing_id in cell.sources:
                        if crossing_id == failed_id or crossing_id == recently_placed:
                            continue
                        rec = self._placed.get(crossing_id)
                        if rec is not None and not rec.is_fallback:
                            crossing_candidates[rec.entry_id] = rec
            if crossing_candidates:
                target = min(
                    crossing_candidates.values(),
                    key=lambda rec: rec.confidence_at_placement,
                ).entry_id
                logger.debug(
                    f"Selected crossing with lowest confidence: {target} "
                    f"(confidence={crossing_candidates[target].confidence_at_placement:.2f})"
                )
        
        if target is None:
            target = self._choose_backtrack_target()
        if target is None:
            logger.debug(f"VERIFICATION FAILED - NO BACKTRACK TARGET: entries={failed}")
            return self._finalize_event({"event": "failed", "verification_failed": failed}, newly_verified)

        popped = self._remove_placed(target)
        newly_verified, _ = self._mark_verified_entries(newly_verified)
        if popped is None:
            return self._finalize_event({"event": "failed", "verification_failed": failed}, newly_verified)
        popped_candidate, popped_record = popped
        logger.debug(
            f"BACKTRACK (verification failed): entry_id={popped_candidate.entry_id} answer={popped_candidate.answer} "
            f"confidence={popped_record.confidence_at_placement:.2f} "
            f"pattern={popped_record.pattern_at_placement} widening_level={popped_candidate.widening_level} "
            f"failed={failed}"
        )
        return self._finalize_event(
            {
                "event": "backtrack",
                "candidate": {
                    "entry_id": popped_candidate.entry_id,
                    "answer": popped_candidate.answer,
                    "widening_level": popped_candidate.widening_level,
                    "confidence": popped_record.confidence_at_placement,
                    "pattern": popped_record.pattern_at_placement,
                },
                "verification_failed": failed,
            },
            newly_verified,
        )

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
