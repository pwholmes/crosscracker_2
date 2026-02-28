from __future__ import annotations
from typing import Any, Callable
import asyncio
import logging

from llm import LLM
from model import Grid, Entry, Candidate, Placement
from heuristics import BasicStrategy

logger = logging.getLogger("src.solver")

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
        self._recording: list[dict[str, Any]] | None = [] if record else None
        # Initialize candidates for all entries at width 0 with empty pattern
        if not defer_candidate_init:
            self._initialize_candidates()

    def record_event(self, event: dict[str, Any]) -> None:
        """Record an event to the recording if recording is enabled.
        
        This is the single point through which all events flow for recording.
        Separated from broadcasting so the server can handle both in a unified way.
        """
        if self._recording is not None:
            self._recording.append(event.copy())


    def _initialize_candidates(self) -> None:
        for entry in self.grid.entries.values():
            # Call get_candidates() to trigger candidate generation for the entry's current pattern
            entry.get_candidates()

    async def async_initialize_with_progress(self, progress_callback: Callable[[int, int], Any]) -> None:
        """Async initialization with progress callback for UI feedback. """
        total = len(self.grid.entries)
        for idx, entry in enumerate(self.grid.entries.values(), 1):
            # Call get_candidates() to trigger candidate generation for the entry's current pattern
            entry.get_candidates()
            result = progress_callback(idx, total)
            if asyncio.iscoroutine(result):
                await result
            await asyncio.sleep(0)

    async def async_step_with_progress(self, progress_callback: Callable[[int, int], Any], broadcast_event_callback: Callable[[dict[str, Any]], Any] | None = None) -> dict[str, Any]:
        """Perform a step, broadcast placement event, then retrieve candidates for crossing entries with progress reporting."""
        # Perform the step and get the placed entry
        event = self.step()
        # Broadcast the placement event immediately if callback is provided
        if broadcast_event_callback is not None:
            result = broadcast_event_callback(event)
            if asyncio.iscoroutine(result):
                await result
        placed_entry_id = None
        candidate_info = event.get("candidate")
        if candidate_info:
            placed_entry_id = candidate_info.get("entry_id")
        # If no entry was placed, nothing to update
        if not placed_entry_id:
            return event
        crossing_ids = self.grid.get_crossing_entry_ids(placed_entry_id)
        affected_entries = [self.grid.entries[eid] for eid in crossing_ids if not self.grid.entries[eid].completed]
        total = len(affected_entries)
        for idx, entry in enumerate(affected_entries, 1):
            entry.get_candidates()
            result = progress_callback(idx, total)
            if asyncio.iscoroutine(result):
                await result
            await asyncio.sleep(0)
        return event

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

        # Loop until we are able to place an entry.  We might pick an Entry that has no
        # viable Candidate, or that would cause crossing entries to fail to verify,
        # in which case we have to continue looping.  If we can't pick ANY viable Entry,
        # we are "stalled", in which case we call the stall logic to backtrack.
        attempted_entries: set[tuple[str, str]] = set()
        while True:
            # Select the best unfilled entry
            selection = BasicStrategy.select_best_unfilled_entry(self.grid, attempted_entries)
            if selection is None:
                logger.debug(f"[STEP SELECT ENTRY]: No viable entry found, invoking stall logic")
                return self._handle_stall()
            entry, candidate, selection_score = selection
            logger.debug(f"[STEP SELECT]: Selected entry {entry.entry_id} with candidate {candidate.answer}, confidence {candidate.confidence:.1f}, selection score {selection_score:.1f}")

            # The selected Entry/Candidate might still be rejected if any crossing entry fails
            # verification. Mark this (entry_id, candidate.answer) as attempted so we don't pick it again.
            attempted_entries.add((entry.entry_id, candidate.answer))

            # Check if any crossing entries would be completed by the placement of this answer
            crossing_entries = self._predict_crossing_entry_patterns(entry.entry_id, candidate.answer)
            if crossing_entries:
                logger.debug(
                    f"[STEP VERIFY]: Checking crossing entries for {entry.entry_id}='{candidate.answer}': "
                    f"{list(crossing_entries.keys())}"
                )
            # Verify the crossing entries
            verified_entry_ids, failed_entry_ids = self.verify_answers(crossing_entries)
            if failed_entry_ids:
                # Reject this candidate and continue trying others
                candidate.penalty += 50
                logger.debug(f"[STEP VERIFY]: REJECTED {entry.entry_id}='{candidate.answer}', "
                             f"verification failed for crossing entries: {list(failed_entry_ids)}"
                )
                continue
            logger.debug(f"[STEP VERIFY]: PASSED entry {entry.entry_id}")

            # Verification passed - now actually place the entry
            placement = Placement(
                entry_id=entry.entry_id,
                answer=candidate.answer,
                pattern=entry.pattern,
                search_level=0,
                confidence=candidate.confidence,
                selection_score=selection_score
            )
            if not self.grid.place_candidate(candidate):
                logger.error(f"[STEP ERROR] Unable to place candidate {candidate.answer} for entry {entry.entry_id}")
                return self._finalize_event({"event": "fatal error"}, [])
            self.grid.entries[entry.entry_id].placement = placement

            # Log the placement and return an event for the UI.
            logger.debug(
                f"[STEP PLACE] entry={placement.entry_id} "
                f"answer='{placement.answer}' "
                f"pattern={placement.pattern} "
                f"search_level={placement.search_level} "
                f"confidence={placement.confidence:.1f} "
                f"selection score={placement.selection_score:.1f} "
            )

            return self._finalize_event(
                {
                    "event": "placed",
                    "candidate": {
                        "entry_id": placement.entry_id,
                        "answer": placement.answer,
                        "pattern": placement.pattern,
                        "search_level": placement.search_level,
                        "confidence": placement.confidence,
                        "selection score": placement.selection_score,
                    },
                },
                verified_entry_ids,
            )
        


    def _handle_stall(self) -> dict[str, Any]:
        """Handle a stall: no placements were made in this pass.
        Returns the backtrack event.
        
        Strategy:
        1. Try to backtrack a placed entry to open up new possibilities
        2. If that entry has been backtracked MAX_BACKTRACKS_BEFORE_FALLBACK times, 
           apply its fallback instead of continuing to thrash
        3. If no backtrack target exists, try to apply any available fallback
        4. If no fallback possible, puzzle has failed
        """
        # Try to select a backtrack target
        entry_id = BasicStrategy.select_best_backtrack_target(self.grid)
        if entry_id is not None:
            entry = self.grid.entries[entry_id]
            placement = entry.placement
            assert placement is not None, "Placement record for a placed entry cannot be None"
            if self._remove_placed(entry_id):
                # Check if this entry has been backtracked too many times
                entry.backtracks += 1
                if entry.backtracks >= self.MAX_BACKTRACKS_BEFORE_FALLBACK:
                    # Force fallback for this thrashing entry
                    entry = self.grid.entries.get(entry_id)
                    if entry is not None and entry.correct_answer:
                        logger.debug(
                            f"FORCING FALLBACK: entry_id={entry_id} "
                            f"backtrack_count={entry.backtracks}"
                        )
                        fallback_event = self._apply_fallback(entry)
                        if fallback_event is not None:
                            return fallback_event
                
                # Normal backtrack - return backtrack event
                logger.debug(
                    f"BACKTRACK: entry_id={placement.entry_id} "
                    f"answer={placement.answer} "
                    f"pattern={placement.pattern} "
                    f"search_level={placement.search_level}"
                    f"confidence={placement.confidence:.1f} "
                    f"selection score={placement.selection_score:.1f} "
                )

                return self._finalize_event(
                    {
                        "event": "backtrack",
                        "candidate": {
                            "entry_id": placement.entry_id,
                            "answer": placement.answer,
                            "pattern": placement.pattern,
                            "search_level": placement.search_level,
                            "confidence": placement.confidence,
                            "selection_score": placement.selection_score,
                        },
                    },
                    [],
                )

        # No backtrack target available - instead try to select an entry for fallback
        logger.debug(f"[STALLED]: No backtrack target available, selecting fallback...")
        entry = BasicStrategy.select_best_fallback_target(self.grid)
        if entry is not None:
            fallback_event = self._apply_fallback(entry)
            if fallback_event is not None:
                return fallback_event

        # No backtrack and no fallback possible - puzzle has failed
        return self._finalize_event({"event": "failed"}, [])


    def _apply_fallback(self, entry: Entry) -> dict[str, Any] | None:
        """Apply a fallback for a specific entry and create the event.
        
        Returns the placed_fallback event, which includes a list of any entries removed 
        due to conflict, or None if unsuccessful.
        """
        answer = entry.correct_answer
        assert len(answer) == entry.length, "Specified correct answer does not have the specified entry length"

        # Remove conflicting non-fallback placements
        # Get a list of crossing entries with conflicting cells
        removed_entry_ids: list[str] = []
        conflicting_entry_ids: set[str] = set()
        for cell, ch in zip(entry.cells, answer):
            if cell.letter is not None and cell.letter != ch:
                conflicting_entry_ids.update(cell.sources)
        conflicting_entry_ids.discard(entry.entry_id)

        # Loop through the conflicting entries
        for conflicting_entry_id in conflicting_entry_ids:
            # If the conflicting entry wasn't explicitly placed, skip it
            # (This shouldn't be possible, such entries couldn't be in this list, but whatever)
            conflicting_entry = self.grid.entries[conflicting_entry_id]
            if conflicting_entry.placement is None:
                continue
            # Also make sure the other entry isn't a fallback.  If it is and they conflict,
            # it's a fatal error because the puzzle definition is wrong.
            assert conflicting_entry.used_fallback is False, f"Two correct answers ({entry.entry_id} and {conflicting_entry_id}) have conflicting values at their intersecting cell"

            # Remove the entry from the Grid
            if self._remove_placed(conflicting_entry_id):
                removed_entry_ids.append(conflicting_entry_id)

        # Create a new "candidate" for the fallback answer.
        candidate = Candidate(entry.entry_id, answer, search_level=0, is_fallback=True)

        # Now the way is clear for the placement of the fallback entry.
        placement = Placement(
            entry_id=entry.entry_id,
            answer=answer,
            pattern=entry.pattern,
            search_level=0,
            confidence=100,
            selection_score=100
        )
        self.grid.place_candidate(candidate)
        entry.placement = placement                

        # Mark it as a fallback
        entry.used_fallback = True

        # Return a fallback event
        event: dict[str, Any] = {
            "event": "placed_fallback",
            "candidate": {
                "entry_id": placement.entry_id,
                "answer": placement.answer,
                "pattern": placement.pattern,
                "search_level": placement.search_level,
                "confidence": placement.confidence,
                "selection_score": placement.selection_score
            },
            "conflicts_removed": removed_entry_ids,
        }
        
        return self._finalize_event(event, [])


    def _remove_placed(self, entry_id: str) -> bool:
        entry = self.grid.entries.get(entry_id)
        if entry is None:
            return False

        # Collect crossing entries before removing
        crossing_entry_ids = self.grid.get_crossing_entry_ids(entry_id)
        logger.debug(f"BACKTRACK: {len(crossing_entry_ids)} crossing entries detected for {entry_id}")

        # Remove the answer from the grid and from the Solver's list of placed entries.
        candidate = Candidate(entry_id=entry.entry_id, answer=entry.pattern, search_level=entry.search_level)
        self.grid.remove_candidate(candidate)

        # Apply a penalty to this answer so it is less likely (but not impossible!) to use again.
        candidate.penalty += 20

        # Reset the entry's placement record
        self.grid.entries[entry_id].placement = None

        # For each crossing entry not explicitly placed, regenerate its candidates.
        # No need to "unverify" them, as entry.verified is a dynamically generated value.
        for crossing_entry_id in crossing_entry_ids:
            crossing_entry = self.grid.entries.get(crossing_entry_id)
            assert crossing_entry is not None, "Invalid crossing entry ID " + crossing_entry_id

            if crossing_entry.placement is not None:
                logger.debug(f"BACKTRACK: Crossing entry {crossing_entry_id} was explicitly placed, not affected by backtrack.")
                continue

        return True


    def _all_filled(self) -> bool:
        return all(e.completed for e in self.grid.entries.values())


    def _predict_crossing_entry_patterns(self, entry_id: str, answer: str) -> dict[str, str]:
        """Compute hypothetical patterns for crossing entries if we placed this answer.
        
        Returns a dict mapping entry_id -> hypothetical_pattern for entries that would
        become complete after placement.
        """
        entry = self.grid.entries[entry_id]
        crossing_patterns: dict[str, str] = {}

        # For each cell in the entry, find its potential crossing entry
        for _, (cell, ch) in enumerate(zip(entry.cells, answer)):
            # Find the crossing entry for this cell (not the current entry)
            crossing_entry = next(
                (e for e in self.grid.entries.values() if cell in e.cells and e is not entry),
                None
            )
            # No crossing entry for this cell (very rare, but possible)
            if crossing_entry is None:
                continue

            # Skip if this entry was explicitly placed
            if crossing_entry.placement is not None:
                logger.debug(f"VERIFY SKIP: {crossing_entry.entry_id} explicitly placed.")
                continue

            # Compute what the pattern would be after placing
            hypothetical_pattern = list(crossing_entry.pattern)
            crossing_cell_idx = crossing_entry.cells.index(cell)
            hypothetical_pattern[crossing_cell_idx] = ch
            pattern_str = "".join(hypothetical_pattern)

            # Only include if it would be complete
            if "." in pattern_str:
                logger.debug(
                    f"VERIFY SKIP: {crossing_entry.entry_id} would still be incomplete: '{pattern_str}' "
                    f"(crossing {entry_id}='{answer}')"
                )
                continue

            # This entry qualifies for verification.  Add it to the dict.
            crossing_patterns[crossing_entry.entry_id] = pattern_str
            logger.debug(
                f"VERIFY CANDIDATE: {crossing_entry.entry_id} would be: '{pattern_str}' "
                    f"(crossing {entry_id}='{answer}')"
            )

        return crossing_patterns


    def verify_answers(self, answers: dict[str, str]) -> tuple[list[str], list[str]]:
        """Verify answers.
        Returns tuple(list[verified_entry_ids], list[failed_entry_ids])
        """
        if not answers:
            return [], []
            
        verified_entry_ids: list[str] = []
        failed_entry_ids: list[str] = []
        
        for entry_id, answer in answers.items():
            entry = self.grid.entries.get(entry_id)
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
    

    def _all_entries_verified(self) -> bool:
        return all(e.verified for e in self.grid.entries.values())


    def _finalize_event(self, event: dict[str, Any], newly_verified: list[str]) -> dict[str, Any]:
        if newly_verified:
            event["verified"] = newly_verified
        if event.get("event") == "verified" and not newly_verified:
            event["event"] = "failed"
        self.record_event(event)
        return event


    def get_recording(self) -> dict[str, Any] | None:
        """Get the recorded solve session as a dict. Returns None if recording is disabled."""
        if self._recording is None:
            return None

        # Try to get puzzle_id and title from the grid and registry
        puzzle_id = getattr(self.grid, 'puzzle_id', None)
        puzzle_title = None
        if puzzle_id is not None:
            try:
                from .puzzles.registry import get_puzzle_spec
                spec = get_puzzle_spec(puzzle_id)
                if spec is not None:
                    puzzle_title = spec.title
            except Exception:
                pass
        return {
            "puzzle_id": puzzle_id or "unknown",
            "puzzle_title": puzzle_title or puzzle_id or "unknown",
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

