from __future__ import annotations
import asyncio
from threading import Lock
import os
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from pydantic import BaseModel
from typing import Any, Callable, cast
import uvicorn
import logging
import json
from pathlib import Path
from datetime import datetime
import uuid

# Load .env file before importing other source files.
# Any file that uses .env values at the global level should call load_dotenv() itself
# just to be sure the env values are loaded, but it doesn't hurt to call it more than once.
load_dotenv()

from model import Grid
from llm import LLM
from solver import Solver
from vector import populate_hints
import puzzles

# Initialize puzzle registry 
current_puzzle_id: str | None = None
grid: Grid | None = None
_hook: Callable[[Any, int, int], list[Any]] | None = None
solver: Solver | None = None
solver_lock = asyncio.Lock()

# Metrics
steps_executed: int = 0
fallbacks_used: int = 0
backtracks_used: int = 0
_metrics_lock = Lock()

# Recordings
BASE_DIR = Path(__file__).resolve().parents[2]
RECORDINGS_DIR = BASE_DIR / "backend" / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
PLAY_INTERVAL_SECONDS = 0

# Logger
logger = logging.getLogger("src.server")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logging.getLogger("urllib3").setLevel(logging.WARNING)


class ControlAction(BaseModel):
    action: str


app = FastAPI()
# Allow cross-origin requests from local files / dev UI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files will be mounted after the websocket and REST routes to avoid
# intercepting websocket connections (StaticFiles expects HTTP scope only).
# See end of file for the mount and favicon handler.


# Simple WebSocket manager
class WSManager:
    def __init__(self):
        self._conns: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._conns.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self._conns:
            self._conns.remove(ws)

    async def broadcast(self, message: dict[str, Any]) -> None:
        living: list[WebSocket] = []
        for ws in list(self._conns):
            try:
                await ws.send_json(message)
                living.append(ws)
            except Exception:
                # drop dead connection
                pass
        self._conns = living

manager = WSManager()

play_event = asyncio.Event()

# --- Reusable progress bar broadcast function ---
async def broadcast_progress(current: int, total: int, *, operation: str = "load"):
    await manager.broadcast({
        "type": "init_progress",
        "current": current,
        "total": total,
        "percentage": round((current / total) * 100) if total else 0,
        "operation": operation,
    })

# utility to serialize state
def serialize_state() -> dict[str, Any]:
    # If no puzzle is loaded, return minimal state
    if grid is None or solver is None:
        return {
            "type": "state",
            "metrics": {
                "puzzle_id": None,
                "steps": 0,
                "fallbacks": 0,
            },
            "grid": None,
        }
    
    # compute bounds
    max_r = 0
    max_c = 0
    used_cells: set[tuple[int, int]] = set()
    for e in grid.entries.values():
        for cell in e.cells:
            used_cells.add((cell.row, cell.col))
            max_r = max(max_r, cell.row)
            max_c = max(max_c, cell.col)
    rows = max_r + 1
    cols = max_c + 1

    letters: list[list[str | None]] = [[None for _ in range(cols)] for _ in range(rows)]
    blocks: list[list[bool]] = [[True for _ in range(cols)] for _ in range(rows)]
    cheated: list[list[bool]] = [[False for _ in range(cols)] for _ in range(rows)]
    for e in grid.entries.values():
        for cell in e.cells:
            letters[cell.row][cell.col] = cell.letter
            blocks[cell.row][cell.col] = False
            cheated[cell.row][cell.col] = bool(getattr(cell, "revealed_by_fallback", False))

    # build clue numbers for starting cells
    starts: dict[tuple[int, int], dict[str, Any]] = {}
    for eid, e in grid.entries.items():
        if not e.cells:
            continue
        start = e.cells[0]
        number = eid[:-1]
        direction = eid[-1].upper()
        key = (start.row, start.col)
        info = starts.get(key)
        if info is None:
            starts[key] = {"number": number, "dirs": {direction}}
        else:
            if info["number"] != number:
                raise ValueError(f"Conflicting clue numbers at cell ({start.row},{start.col}): {info['number']} vs {number}")
            if direction in info["dirs"]:
                raise ValueError(f"Duplicate clue start for {eid} at cell ({start.row},{start.col})")
            if len(info["dirs"]) >= 2:
                raise ValueError(f"More than two clues start at cell ({start.row},{start.col})")
            info["dirs"].add(direction)

    numbers: list[list[str | None]] = [[None for _ in range(cols)] for _ in range(rows)]
    for (r, c), info in starts.items():
        numbers[r][c] = info["number"]

    entries: dict[str, dict[str, Any]] = {
        eid: {
            "pattern": e.pattern,
            "clue": e.clue,
            "used_fallback": e.used_fallback,
            "verified": e.verified,
            "correct_answer": e.correct_answer,
        }
        for eid, e in grid.entries.items()
    }
    return {
        "type": "state",
        "metrics": {
            "puzzle_id": current_puzzle_id,
            "steps": steps_executed,
            "fallbacks": fallbacks_used,
            "backtracks": backtracks_used,
        },
        "grid": {
            "rows": rows,
            "cols": cols,
            "letters": letters,
            "blocks": blocks,
            "numbers": numbers,
            "cheated": cheated,
        },
        "entries": entries,
    }


def _step_and_update_metrics() -> dict[str, Any]:
    """Run one solver step and update authoritative metrics.
    
    Events are automatically recorded by the Solver's record_event() method,
    called via _finalize_event() during step execution.
    """
    global steps_executed, fallbacks_used, backtracks_used
    if solver is None:
        raise RuntimeError("Solver not initialized")
    ev = solver.step()
    with _metrics_lock:
        steps_executed += 1
        if ev.get("event") == "placed_fallback":
            fallbacks_used += 1
        elif ev.get("event") == "backtrack":
            backtracks_used += 1
    return ev


def _save_recording() -> None:
    """Save the current solver recording."""
    # This function is now only called when the user requests to save the recording.
    if solver is None or not hasattr(solver, 'get_recording'):
        return
    recording = solver.get_recording()
    if recording is None:
        return
    
    try:
        recording_id = str(uuid.uuid4())
        recording['id'] = recording_id
        recording['timestamp'] = datetime.now().isoformat()
        recording['event_count'] = len(recording.get('events', []))
        # Assign recording_title if missing
        if not recording.get('recording_title'):
            puzzle_title = recording.get('puzzle_title') or 'Unknown Puzzle'
            recording['recording_title'] = f"Recording for {puzzle_title} ({recording_id[:8]})"

        # Add grid state for replay
        if grid is not None:
            # Build initial grid state
            max_r = 0
            max_c = 0
            used_cells: set[tuple[int, int]] = set()
            for e in grid.entries.values():
                for cell in e.cells:
                    used_cells.add((cell.row, cell.col))
                    max_r = max(max_r, cell.row)
                    max_c = max(max_c, cell.col)

            rows = max_r + 1
            cols = max_c + 1

            blocks = [[True for _ in range(cols)] for _ in range(rows)]
            for e in grid.entries.values():
                for cell in e.cells:
                    blocks[cell.row][cell.col] = False

            recording['grid_state'] = {
                'rows': rows,
                'cols': cols,
                'blocks': blocks,
            }

        # Generate filename based on puzzle_id and ordinal
        puzzle_id = recording.get('puzzle_id', recording.get('puzzle', 'unknown'))
        safe_name = str(puzzle_id).replace(' ', '_').replace('/', '_').replace('\\', '_')

        # Find next available ordinal
        ordinal = 1
        while True:
            filename = f"{safe_name} - {ordinal}.json"
            filepath = RECORDINGS_DIR / filename
            if not filepath.exists():
                break
            ordinal += 1

        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(recording, f, indent=2)

        logger.info(f"Recording saved: {filename} (id: {recording_id}) for puzzle {puzzle_id}")
    except Exception as e:
        logger.error(f"Failed to save recording: {e}")


async def broadcast_step_events(ev: dict[str, Any]) -> None:
    """Broadcast step event and any verification events."""
    await manager.broadcast({"type": "event", "event": ev})
    if "verification_failed" in ev:
        await manager.broadcast({
            "type": "status",
            "message": f"Verification failed for: {', '.join(ev['verification_failed'])}"
        })
    if "verified" in ev:
        for entry_id in ev["verified"]:
            await manager.broadcast({"type": "event", "event": {"event": "candidate_verified", "entry_id": entry_id}})
    await manager.broadcast(serialize_state())


async def _emit_solver_step() -> dict[str, Any]:
    """Unified function for executing one solver step and broadcasting the result.
    
    This is the single point through which all solver steps flow, ensuring that:
    1. Events are recorded by the Solver (via _finalize_event → record_event)
    2. Events are broadcast to clients in the same operation
    3. The frontend always receives exactly what was recorded
    
    Returns the event dict. The caller should check the event type for terminal conditions.
    """
    if solver is None:
        raise RuntimeError("Solver not initialized")
    # Broadcast 'Solving...' status before step
    await manager.broadcast({
        "type": "status",
        "message": "Solving...",
        "state": "solving"
    })
    # Progress callback for step candidate retrieval
    async def progress_callback(current: int, total: int):
        await broadcast_progress(current, total, operation="step")
    # Placement event broadcast callback
    async def broadcast_event_callback(ev: dict[str, Any]):
        await broadcast_step_events(ev)
    # Execute step with progress bar, broadcasting placement event first
    if hasattr(solver, "async_step_with_progress"):
        ev = await solver.async_step_with_progress(progress_callback, broadcast_event_callback)
        # Send progress_done message to hide progress bar
        await manager.broadcast({"type": "progress_done", "operation": "step"})
    else:
        ev = await asyncio.to_thread(_step_and_update_metrics)
        await broadcast_step_events(ev)
    # If not solved/failed, broadcast 'Awaiting user input'
    if ev.get("event") not in ("solved", "failed"):
        await manager.broadcast({
            "type": "status",
            "message": "Ready",
            "state": "awaiting_input"
        })
    return ev

async def start_player() -> None:
    async def player_loop() -> None:
        while True:
            await play_event.wait()
            # Do one step (records and broadcasts automatically via _emit_solver_step)
            async with solver_lock:
                ev = await _emit_solver_step()
            if ev.get("event") in ("solved", "failed"):
                play_event.clear()
                # Recording is only saved if user confirms via frontend
            await asyncio.sleep(PLAY_INTERVAL_SECONDS)

    asyncio.create_task(player_loop())

# Register startup handler (preferred over deprecated decorator)
# Pylance sometimes reports the member type of add_event_handler as partially unknown;
# add an inline ignore to keep the Problems window clean while preserving runtime behavior.
app.add_event_handler("startup", start_player)  # type: ignore[reportUnknownMemberType]



@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        # send initial state
        await ws.send_json(serialize_state())
        while True:
            # keep connection alive; we don't expect client messages for now
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(ws)


@app.post("/play")
async def play() -> dict[str, str]:
    play_event.set()
    # Broadcast 'Solving...' status when play is triggered
    await manager.broadcast({
        "type": "status",
        "message": "Solving...",
        "state": "solving"
    })
    return {"status": "playing"}


@app.post("/pause")
async def pause() -> dict[str, str]:
    play_event.clear()
    # Broadcast 'Awaiting user input' status when paused
    await manager.broadcast({
        "type": "status",
        "message": "Ready",
        "state": "awaiting_input"
    })
    return {"status": "paused"}


@app.post("/step")
async def step_once() -> dict[str, Any]:
    if solver is None:
        return {"event": "error", "message": "No puzzle loaded"}
    async with solver_lock:
        ev = await _emit_solver_step()
    # Recording is only saved if user confirms via frontend
    return ev


@app.post("/reset")
async def reset() -> dict[str, str]:
    global solver, grid, current_puzzle_id, steps_executed, fallbacks_used, backtracks_used
    if current_puzzle_id is None:
        return {"status": "no puzzle loaded"}
    play_event.clear()
    async with solver_lock:
        # Reload the puzzle and its candidate-generation hook
        grid, hook = puzzles.load_puzzle(puzzle_id=current_puzzle_id)
        grid.puzzle_id = current_puzzle_id
        LLM.set_generate_candidates_hook(hook)

        # Create the solver but defer candidate init; we'll initialize with progress
        solver = Solver(grid, defer_candidate_init=True, record=True)

        # Use the reusable progress function
        async def progress_callback(current: int, total: int):
            await broadcast_progress(current, total, operation="reset")

        # Initialize candidates for all entries (async) so UI can show progress
        try:
            await solver.async_initialize_with_progress(progress_callback)
        except Exception:
            # If initialization fails, log but continue to reset solver state
            logger.exception("Candidate initialization failed during reset")

        with _metrics_lock:
            steps_executed = 0
            fallbacks_used = 0
            backtracks_used = 0
    await manager.broadcast(serialize_state())
    return {"status": "reset"}


@app.get("/state")
async def get_state():
    return serialize_state()


@app.post("/api/save_recording")
async def save_recording_decision(request: Request) -> dict[str, Any]:
    """Save the current solver recording if the user confirms saving."""
    _save_recording()
    logger.info("Recording saved by user request.")
    return {"success": True, "saved": True}


@app.get("/api/recordings")
def list_recordings() -> list[dict[str, Any]]:
    """List all saved recordings."""
    try:
        recordings: list[dict[str, Any]] = []
        for json_file in sorted(RECORDINGS_DIR.glob("*.json")):
            try:
                with open(json_file, 'r') as f:
                    rec = json.load(f)
                puzzle_id = rec.get('puzzle_id', 'unknown puzzle_id')
                puzzle_title = rec.get('puzzle_title', 'unknown puzzle_title')
                recording_id = rec.get('id', json_file.stem)
                recording_title = rec.get('recording_title')
                if not recording_title:
                    recording_title = f"{puzzle_title}-{recording_id[:8]}"
                recordings.append({
                    'recording_id': recording_id,
                    'recording_title': recording_title,
                    'puzzle_id': puzzle_id,
                    'puzzle_title': puzzle_title,
                    'width': rec.get('width', 0),
                    'height': rec.get('height', 0),
                    'event_count': rec.get('event_count', len(rec.get('events', []))),
                    'timestamp': rec.get('timestamp', ''),
                })
            except Exception as e:
                logger.error(f"Failed to load recording {json_file}: {e}")
        return recordings
    except Exception as e:
        logger.error(f"Failed to list recordings: {e}")
        return []


@app.get("/api/recordings/{recording_id}")
def get_recording(recording_id: str) -> dict[str, Any]:
    """Fetch a specific recording."""
    try:
        # Search through all recording files to find one with matching ID
        for json_file in RECORDINGS_DIR.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    rec = cast(dict[str, Any], json.load(f))
                # Check if this recording has the matching ID
                if rec.get('id') == recording_id:
                    logger.debug(f"Retrieved recording {recording_id}")
                    return rec
            except Exception:
                pass
        
        return {"error": "Recording not found"}
    except Exception as e:
        logger.error(f"Failed to get recording {recording_id}: {e}")
        return {"error": str(e)}


@app.get("/api/puzzle/{puzzle_id}")
def get_puzzle(puzzle_id: str) -> dict[str, Any]:
    """Get puzzle metadata including entries and their cell coordinates."""
    try:
        # Load the puzzle to get its entries
        try:
            grid, _ = puzzles.load_puzzle(puzzle_id=puzzle_id)
        except KeyError:
            msg = f"Puzzle ID '{puzzle_id}' is not registered. This recording or request refers to a puzzle that is not available."
            logger.warning(msg)
            return {"error": msg, "error_type": "puzzle_not_found", "puzzle_id": puzzle_id}

        # Build entry-to-cells mapping from the grid
        entries: dict[str, dict[str, Any]] = {}
        for entry_id, entry in grid.entries.items():
            # Extract cell coordinates from the entry's cells
            cells: list[list[int]] = []
            for cell in entry.cells:
                cells.append([int(cell.row), int(cell.col)])

            entries[entry_id] = {
                "clue": entry.clue,
                "answer": entry.correct_answer,
                "length": entry.length,
                "cells": cells,
            }

        return {
            "puzzle_id": puzzle_id,
            "width": grid.width,
            "height": grid.height,
            "entries": entries,
        }
    except Exception as e:
        logger.error(f"Failed to get puzzle {puzzle_id}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


@app.get("/puzzles")
def list_puzzles():
    return puzzles.list_puzzles()


@app.post("/puzzles/{puzzle_id}/load")
async def load_puzzle(puzzle_id: str):
    global solver, grid, current_puzzle_id, steps_executed, fallbacks_used, backtracks_used
    play_event.clear()
    current_puzzle_id = puzzle_id
    async with solver_lock:
        grid, hook = puzzles.load_puzzle(puzzle_id=puzzle_id)
        grid.puzzle_id = puzzle_id  # Store puzzle_id on the grid for recording
        LLM.set_generate_candidates_hook(hook)

        await manager.broadcast({
            "type": "status",
            "message": "Getting hints...",
            "state": "",
        })
        try:
            await asyncio.to_thread(populate_hints, list(grid.entries.values()))
            logger.debug("Hints loaded")
        except Exception as exc:
            logging.getLogger("src.server").warning(
                "Vector DB hint population failed; continuing without hints: %s",
                exc,
            )
        finally:
            await manager.broadcast({
                "type": "status",
                "message": "Initializing Puzzle...",
                "state": "initializing",
            })
        
        # Create solver with recording enabled
        solver = Solver(grid, defer_candidate_init=True, record=True)
        
        # Use the reusable progress function
        async def progress_callback(current: int, total: int):
            await broadcast_progress(current, total, operation="load")

        await solver.async_initialize_with_progress(progress_callback)
        
        with _metrics_lock:
            steps_executed = 0
            fallbacks_used = 0
            backtracks_used = 0
    
    await manager.broadcast(serialize_state())
    return {"status": "loaded", "puzzle": puzzle_id}


# Mount frontend static files after routes are defined so websocket routes are matched first
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    import logging
    logging.getLogger(__name__).warning("Frontend directory %s not found, not mounting static files", FRONTEND_DIR)


@app.get("/favicon.ico")
async def favicon():
    fav = FRONTEND_DIR / "favicon.ico"
    if fav.exists():
        return FileResponse(str(fav))
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="Not Found")


if __name__ == "__main__":
    import sys
    import os
    # Add backend directory to path for relative imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=False)
