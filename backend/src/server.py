from __future__ import annotations
import asyncio
from threading import Lock
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Any, Dict, List, Callable
import uvicorn
import logging

from .model import Grid
from .llm import LLM
from .solver import Solver
from .vector import populate_hints
from . import puzzles

# Initialize puzzle registry 
current_puzzle_id: str | None = None
grid: Grid | None = None
_hook: Callable[[Any, int, int], list[Any]] | None = None
solver: Solver | None = None
solver_lock = asyncio.Lock()

# Metrics
steps_executed: int = 0
fallbacks_used: int = 0
_metrics_lock = Lock()

# Logger
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
        self._conns: List[WebSocket] = []

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
play_interval_seconds = 0.8

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

    entries: Dict[str, Dict[str, Any]] = {
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
    """Run one solver step and update authoritative metrics."""
    global steps_executed, fallbacks_used
    if solver is None:
        raise RuntimeError("Solver not initialized")
    ev = solver.step()
    with _metrics_lock:
        steps_executed += 1
        if ev.get("event") == "placed_fallback":
            fallbacks_used += 1
    return ev


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

async def start_player() -> None:
    async def player_loop() -> None:
        while True:
            await play_event.wait()
            # Do one step
            async with solver_lock:
                ev = await asyncio.to_thread(_step_and_update_metrics)
            # After performing the step, broadcast the event and updated state
            await broadcast_step_events(ev)
            if ev.get("event") in ("solved", "failed"):
                play_event.clear()
            await asyncio.sleep(play_interval_seconds)

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
    return {"status": "playing"}


@app.post("/pause")
async def pause() -> dict[str, str]:
    play_event.clear()
    return {"status": "paused"}


@app.post("/step")
async def step_once() -> dict[str, Any]:
    if solver is None:
        return {"event": "error", "message": "No puzzle loaded"}
    async with solver_lock:
        ev = await asyncio.to_thread(_step_and_update_metrics)
    await broadcast_step_events(ev)
    return ev


@app.post("/solve")
async def solve_now() -> dict[str, Any]:
    if solver is None:
        return {"event": "error", "message": "No puzzle loaded"}
    # Run to completion, broadcasting each solver step.
    # (This keeps the UI log and step tally accurate.)
    async with solver_lock:
        while True:
            ev = await asyncio.to_thread(_step_and_update_metrics)
            await broadcast_step_events(ev)
            if ev.get("event") in ("solved", "failed"):
                return {"solved": ev.get("event") == "solved"}


@app.post("/reset")
async def reset() -> dict[str, str]:
    global solver, grid, current_puzzle_id, steps_executed, fallbacks_used
    if current_puzzle_id is None:
        return {"status": "no puzzle loaded"}
    play_event.clear()
    async with solver_lock:
        grid, hook = puzzles.load_puzzle(puzzle_id=current_puzzle_id)
        LLM.set_generate_candidates_hook(hook)
        solver = Solver(grid)
        with _metrics_lock:
            steps_executed = 0
            fallbacks_used = 0
    await manager.broadcast(serialize_state())
    return {"status": "reset"}


@app.get("/state")
async def get_state():
    return serialize_state()


@app.get("/puzzles")
def list_puzzles():
    return puzzles.list_puzzles()

@app.post("/puzzles/{puzzle_id}/load")
async def load_puzzle(puzzle_id: str):
    global solver, grid, current_puzzle_id, steps_executed, fallbacks_used
    play_event.clear()
    current_puzzle_id = puzzle_id
    async with solver_lock:
        grid, hook = puzzles.load_puzzle(puzzle_id=puzzle_id)
        LLM.set_generate_candidates_hook(hook)

        await manager.broadcast({
            "type": "status",
            "message": "Getting hints...",
            "state": "",
        })
        try:
            await asyncio.to_thread(populate_hints, list(grid.entries.values()))
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
        
        # Create solver (defer candidate init so we can stream progress)
        solver = Solver(grid, defer_candidate_init=True)
        
        # Initialize candidates with progress callback for UI feedback
        async def progress_callback(current: int, total: int):
            await manager.broadcast({
                "type": "init_progress",
                "current": current,
                "total": total,
                "percentage": round((current / total) * 100)
            })
        
        await solver.async_initialize_with_progress(progress_callback)
        
        with _metrics_lock:
            steps_executed = 0
            fallbacks_used = 0
    await manager.broadcast(serialize_state())
    return {"status": "loaded", "puzzle": puzzle_id}


# Mount frontend static files after routes are defined so websocket routes are matched first
BASE_DIR = Path(__file__).resolve().parents[2]
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
    uvicorn.run("backend.src.server:app", host="0.0.0.0", port=8000, reload=False)
