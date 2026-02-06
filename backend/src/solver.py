from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import asyncio
import logging
from .llm import LLM
from .model import Candidate, CandidateCache, Entry, Grid, ScoredCandidate

logger = logging.getLogger(__name__)


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

    def __init__(self, grid: Grid, *, defer_candidate_init: bool = False):
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
        # Initialize candidates for all entries at width 0 with empty pattern
        if not defer_candidate_init:
            self._initialize_candidates_at_width(0)

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
            return self._handle_verification_failure(verification_failed, newly_verified)

        if self._all_filled():
            if self._all_entries_verified():
                return self._finalize_event({"event": "solved"}, newly_verified)
            return self._finalize_event({"event": "failed"}, newly_verified)

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
                return self._handle_verification_failure(verification_failed, newly_verified)

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

        for cand in attempt.candidates[attempt.next_index :]:
            if len(cand.answer) != entry.length:
                continue
            if self._can_place(entry, cand.answer):
                return cand
        
        # Debug: log why no candidate was found
        if attempt.candidates:
            import logging
            logger = logging.getLogger("src.solver")
            logger.debug(f"No placeable candidate found for {entry_id}: "
                        f"{len(attempt.candidates)} candidates, pattern={pattern}, "
                        f"entries returned {len([c for c in attempt.candidates if len(c.answer) == entry.length])} "
                        f"with correct length")
        return None

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

        if self._all_filled() and self._all_entries_verified():
            return self._finalize_event({"event": "solved"}, newly_verified)

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
                # Log hints for the backtracked entry
                entry = self.entries.get(removed_candidate.entry_id)
                if entry and entry.hints:
                    for hint_clue, hint_answer in entry.hints:
                        logger.debug(f"  HINT: '{hint_clue}' -> '{hint_answer}'")
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
        candidates = [rec for rec in self._placed.values() if not rec.is_fallback]
        if not candidates:
            return None

        # Score each candidate by confidence, but penalize high crossing counts
        # and heavily penalize entries that intersect unfilled entries.
        # Lower score = more likely to backtrack.
        def backtrack_score(rec: PlacedRecord) -> float:
            entry = self.entries[rec.entry_id]
            crossing_count = self._count_crossing_letters(entry)
            unfilled_blocking_count = self._count_unfilled_blocking_entries(entry)
            
            # Formula: confidence with penalties
            # - crossing_count: makes entry safer (harder to backtrack) since it affects others
            # - unfilled_blocking_count: makes entry more likely to backtrack since it's blocking progress
            # Base score: confidence / (1 + crossing_penalty)
            base_score = rec.score_at_placement / (1.0 + crossing_count * 0.5)
            
            # Apply strong penalty for each unfilled entry this blocks
            # Divide by (1 + unfilled_blocking_count) to make blocking entries easier to backtrack
            penalty_score = base_score / (1.0 + unfilled_blocking_count * 1.5)
            
            return penalty_score

        candidates.sort(key=backtrack_score)
        return candidates[0].entry_id

    def _count_unfilled_blocking_entries(self, entry: Entry) -> int:
        """Count how many unfilled entries intersect this entry.
        
        These are potential "blocked" entries that might be unblocked by removing this one.
        """
        blocking_count = 0
        for cell in entry.cells:
            # Find all other entries that pass through this cell
            for other_eid in cell.sources or set():
                if other_eid == entry.entry_id:
                    continue
                other_entry = self.entries.get(other_eid)
                if other_entry and "." in other_entry.pattern:
                    # This other entry still has unfilled cells
                    blocking_count += 1
                    break  # Count each intersecting unfilled entry once
        return blocking_count

    def _count_crossing_letters(self, entry: Entry) -> int:
        """Count how many letters in this entry came from other placed entries."""
        count = 0
        for cell in entry.cells:
            # If this cell has sources other than the current entry, it's a crossing.
            if cell.sources and entry.entry_id not in cell.sources:
                count += 1
        return count

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

        candidate = Candidate(entry_id=rec.entry_id, answer=rec.answer, widening_level=rec.width_used)
        self.grid.remove_candidate(candidate)

        key = (entry_id, rec.pattern_at_placement, rec.width_used)
        self._penalties.setdefault(key, {})[rec.answer] = self._penalties.get(key, {}).get(rec.answer, 0.0) + 20.0

        self._placed.pop(entry_id, None)
        if entry_id in self._placed_order:
            self._placed_order.remove(entry_id)

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
            if LLM.verify_answer(entry, pattern):
                entry.verified = True
                newly_verified.append(entry.entry_id)
            else:
                failed.append(entry.entry_id)
        return newly_verified, failed

    def _all_entries_verified(self) -> bool:
        return all(e.verified for e in self.entries.values())

    def _finalize_event(self, event: dict[str, Any], newly_verified: list[str]) -> dict[str, Any]:
        if newly_verified:
            event["verified"] = newly_verified
        if event.get("event") == "verified" and not newly_verified:
            event["event"] = "failed"
        return event

    def _handle_verification_failure(self, failed: list[str], newly_verified: list[str]) -> dict[str, Any]:
        # Log the verification failure
        logger.debug(f"VERIFICATION FAILED: entries={failed}")
        
        # Prefer removing a failing entry if it was placed and is not a fallback.
        target: str | None = None
        for eid in failed:
            if eid in self._placed and not self._placed[eid].is_fallback:
                target = eid
                break
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
        # Log hints for the backtracked entry
        #entry = self.entries.get(popped_candidate.entry_id)
        #if entry and entry.hints:
        #    for hint_clue, hint_answer in entry.hints:
        #        logger.debug(f"  HINT: '{hint_clue}' -> '{hint_answer}'")
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
