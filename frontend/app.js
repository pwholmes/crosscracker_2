const logEl = document.getElementById("log");
const gridContainer = document.getElementById("grid-container");
const entriesAcrossList = document.getElementById("entries-across");
const entriesDownList = document.getElementById("entries-down");
const statusMessage = document.getElementById("status-message");

const playBtn = document.getElementById("play");
const pauseBtn = document.getElementById("pause");
const stepBtn = document.getElementById("step");
const resetBtn = document.getElementById("reset");


const modeToggle = document.getElementById('mode-toggle');
const itemSelect = document.getElementById('item-select');
const itemSelectLabel = document.getElementById('item-select-label');


const stepCountEl = document.getElementById('step-count');
const fallbackCountEl = document.getElementById('fallback-count');
const backtrackCountEl = document.getElementById('backtrack-count');

const initProgressDiv = document.getElementById('init-progress');
const progressText = document.getElementById('progress-text');
const progressBarFill = document.getElementById('progress-bar-fill');
let initProgressStarted = false;

// Replay controls
const replaySpeedControl = document.getElementById('replay-speed-control');
const replaySpeedSlider = document.getElementById('replay-speed');
const speedLabel = document.getElementById('speed-label');
const currentStepSpan = document.getElementById('current-step');
const totalStepsSpan = document.getElementById('total-steps');

// Mode state
let currentMode = 'puzzles'; // 'puzzles' or 'recordings'
let isReplayMode = false;
let currentRecording = null;
let replayIndex = 0;
let isReplayPlaying = false;
let isPuzzlePlaying = false; // Track puzzle mode auto-play state
let replaySpeed = 1.0;
let replayTimer = null;

// Track grid state during replay
let replayGridState = null; // { rows, cols, letters, blocks }
let replayBaseGridState = null;
let replayEntries = null;
let replayBaseEntries = null;
let entryToCells = {}; // mapping from entry_id to list of [row, col] for replay
let replayFallbackEntries = new Set(); // track which entries were placed with fallback
let replayBacktrackCount = 0; // count total backtrack events during replay
let replayPlacedEntries = new Map(); // track currently placed entries: entry_id -> answer

function disableControls(disabled) {
  playBtn.disabled = disabled;
  pauseBtn.disabled = disabled;
  stepBtn.disabled = disabled;
  resetBtn.disabled = disabled;
}

function setPlayingState(playing) {
  if (playing) {
    playBtn.disabled = true;
    stepBtn.disabled = true;
    resetBtn.disabled = true;
    pauseBtn.disabled = false;
  } else {
    playBtn.disabled = false;
    stepBtn.disabled = false;
    resetBtn.disabled = false;
    pauseBtn.disabled = true;
  }
}

function setCompletedState() {
  // When puzzle is solved or failed, disable Play/Step/Pause but keep Reset enabled
  playBtn.disabled = true;
  stepBtn.disabled = true;
  pauseBtn.disabled = true;
  resetBtn.disabled = false;
}

function renderTallyFromState(metrics) {
  const steps = metrics?.steps ?? 0;
  const fallbacks = metrics?.fallbacks ?? 0;
  const backtracks = metrics?.backtracks ?? 0;
  if (stepCountEl) stepCountEl.textContent = String(steps);
  if (fallbackCountEl) fallbackCountEl.textContent = String(fallbacks);
  if (backtrackCountEl) backtrackCountEl.textContent = String(backtracks);
}

function setClueHeadingsVisible(visible) {
  const clueColumns = document.querySelectorAll('.clue-column h3');
  clueColumns.forEach(h3 => h3.style.display = visible ? 'block' : 'none');
}

function cloneJson(value) {
  return value ? JSON.parse(JSON.stringify(value)) : null;
}

function applyReplayEventToGrid(event, gridState, entryToCells, placedEntries) {
  if (!event || !gridState) return;
  const letters = gridState.letters || gridState.cells;
  if (!letters) return;
  
  const entryId = event.candidate?.entry_id;
  
  if (event.event === 'placed' || event.event === 'placed_fallback') {
    if (entryId && event.candidate?.answer) {
      const cells = entryToCells[entryId];
      if (cells) {
        const answer = String(event.candidate.answer);
        cells.forEach(([r, c], idx) => {
          if (idx < answer.length) {
            letters[r][c] = answer[idx];
          }
        });
        // Track this placement
        placedEntries.set(entryId, answer);
      }
    }
  } else if (event.event === 'backtrack') {
    if (entryId) {
      const cells = entryToCells[entryId];
      if (cells) {
        // Remove from placed entries
        placedEntries.delete(entryId);
        
        // Smart clearing: only clear cells that don't belong to other placed entries
        cells.forEach(([r, c]) => {
          // Check if this cell belongs to any other placed entry
          let belongsToOther = false;
          for (const [otherEntryId, otherAnswer] of placedEntries.entries()) {
            if (otherEntryId === entryId) continue;
            const otherCells = entryToCells[otherEntryId];
            if (otherCells) {
              const cellIndex = otherCells.findIndex(([or, oc]) => or === r && oc === c);
              if (cellIndex >= 0 && cellIndex < otherAnswer.length) {
                belongsToOther = true;
                break;
              }
            }
          }
          // Only clear if no other placed entry uses this cell
          if (!belongsToOther) {
            letters[r][c] = '';
          }
        });
      }
    }
  }
}

function computeReplayEntriesFromGrid(baseEntries, gridState, entryToCells, fallbackEntries) {
  const entries = cloneJson(baseEntries);
  if (!entries || !gridState) return entries;
  
  const letters = gridState.letters || gridState.cells;
  if (!letters) return entries;
  
  Object.entries(entryToCells).forEach(([entryId, cells]) => {
    if (!entries[entryId] || !cells) return;
    const entry = entries[entryId];
    const pattern = cells.map(([r, c]) => {
      const letter = letters?.[r]?.[c];
      return letter ? String(letter) : '.';
    }).join('');
    entry.pattern = pattern;
    entry.used_fallback = fallbackEntries.has(entryId);
  });
  
  return entries;
}

const logHeader = document.getElementById("log-header");
const logToggle = document.getElementById("log-toggle");
let logCollapsed = true;

logHeader.addEventListener('click', () => {
  logCollapsed = !logCollapsed;
  logEl.style.display = logCollapsed ? 'none' : 'block';
  logToggle.textContent = logCollapsed ? '▶' : '▼';
});

let gridState = null; // holds latest state
let ws = null;

const DEBUG_LOG_JSON = false;

function fmtNum(n, digits = 2) {
  if (typeof n !== 'number' || Number.isNaN(n)) return null;
  return n.toFixed(digits);
}

function fmtCandidate(cand) {
  if (!cand) return '';
  const eid = cand.entry_id ?? '?';
  const ans = cand.answer ?? '';
  const w = cand.widening_level;
  const conf = fmtNum(cand.confidence);
  const score = fmtNum(cand.score);
  const pat = cand.pattern;
  let parts = [`${eid}="${ans}"`];
  if (typeof w === 'number') parts.push(`w=${w}`);
  if (conf !== null) parts.push(`conf=${conf}`);
  if (score !== null) parts.push(`score=${score}`);
  if (typeof pat === 'string' && pat.length > 0) parts.push(`pat="${pat}"`);
  return parts.join(' ');
}

function formatEvent(ev) {
  if (!ev || typeof ev !== 'object') return String(ev);
  const type = ev.event;
  if (type === 'placed') return `placed ${fmtCandidate(ev.candidate)}`;
  if (type === 'backtrack') {
    const vf = Array.isArray(ev.verification_failed) ? ` vf=${ev.verification_failed.join(',')}` : '';
    return `backtrack ${fmtCandidate(ev.candidate)}${vf}`;
  }
  if (type === 'placed_fallback') {
    const removed = Array.isArray(ev.conflicts_removed) && ev.conflicts_removed.length
      ? ` removed=[${ev.conflicts_removed.join(',')}]`
      : '';
    return `fallback ${fmtCandidate(ev.candidate)}${removed}`;
  }
  if (type === 'candidate_verified') return `verified ${ev.entry_id}`;
  if (type === 'solved') return 'solved';
  if (type === 'failed') {
    const vf = Array.isArray(ev.verification_failed) ? ` vf=${ev.verification_failed.join(',')}` : '';
    return `failed${vf}`;
  }
  return JSON.stringify(ev);
}

function log(msg) {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

// ============ MODE SWITCHING FUNCTIONALITY ============

async function switchMode(mode) {
  currentMode = mode;


  if (mode === 'puzzles') {
    itemSelectLabel.textContent = 'Puzzle:';
    replaySpeedControl.style.display = 'none';
    isReplayMode = false;
    if (replayTimer) clearTimeout(replayTimer);
    isReplayPlaying = false;
    await refreshItemList();
  } else if (mode === 'recordings') {
    itemSelectLabel.textContent = 'Recording:';
    replaySpeedControl.style.display = 'none';
    await refreshItemList();
  }
  
  // Reset state
  itemSelect.value = '';
  gridContainer.innerHTML = '';
  entriesAcrossList.innerHTML = '';
  entriesDownList.innerHTML = '';
  document.getElementById('tally').style.display = 'none';
  statusMessage.textContent = mode === 'puzzles' ? 'Select a puzzle' : 'Select a recording';
  statusMessage.className = '';
  logEl.textContent = '';
  setClueHeadingsVisible(false);
  disableControls(true);
}

async function refreshItemList() {
  itemSelect.innerHTML = '';
  
  if (currentMode === 'puzzles') {
    // Add default option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = '-- select puzzle --';
    itemSelect.appendChild(defaultOption);
    
    // Load puzzles
    try {
      const res = await fetch(API_BASE + '/puzzles');
      if (!res.ok) throw new Error(res.statusText);
      const puzzles = await res.json();
      puzzles.forEach((p) => {
        const opt = document.createElement('option');
        opt.value = p.id;
        opt.textContent = p.title || p.id;
        itemSelect.appendChild(opt);
      });

    } catch (err) {
      log('[error] Could not load puzzles: ' + err);
    }
  } else if (currentMode === 'recordings') {
    // Add default option
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = '-- select recording --';
    itemSelect.appendChild(defaultOption);
    
    // Load recordings
    try {
      const res = await fetch(API_BASE + '/api/recordings');
      if (!res.ok) throw new Error(res.statusText);
      const recordings = await res.json();
      recordings.forEach((rec) => {
        const opt = document.createElement('option');
        opt.value = rec.recording_id
        opt.textContent = `${rec.puzzle_title} (${rec.width}x${rec.height}) - ${rec.event_count} events`;
        itemSelect.appendChild(opt);
      });
    } catch (err) {
      log('[error] Could not load recordings: ' + err);
    }
  }
}

// ============ CENTRALIZED MESSAGE PROCESSOR ============

function processMessage(data) {
  log("Message received: " + JSON.stringify(data))
  if (data.type === 'state') {
    gridState = data.grid;
    renderGrid(data.grid);
    renderEntries(data.entries);
    renderTallyFromState(data.metrics);
    // Sync dropdown to current puzzle (only in puzzle mode)
    const currentPuzzleId = data.metrics?.puzzle_id;
    if (currentMode === 'puzzles' && currentPuzzleId && itemSelect.value !== currentPuzzleId) {
      itemSelect.value = currentPuzzleId;
    }
    log('[state] updated');
    
    // Handle puzzle loaded vs not loaded
    if (!currentPuzzleId) {
      // No puzzle loaded - show "Select a puzzle" and hide sections
      statusMessage.textContent = 'Select a puzzle';
      statusMessage.className = '';
      document.getElementById('tally').style.display = 'none';
      // Find and hide the Across and Down headings
      setClueHeadingsVisible(false);
      //disableControls(true);
    } else {
      // Puzzle loaded - show sections and "Loaded" message
      // Set status to Loaded when state is received (unless already solving/solved/failed)
      if (!statusMessage.textContent.match(/Solving|Solved|Failed|Replaying/)) {
        statusMessage.textContent = 'Loaded';
        statusMessage.className = '';
      }
      // Show tally and headings
      document.getElementById('tally').style.display = 'block';
      setClueHeadingsVisible(true);
      // Enable controls now that puzzle is loaded (unless currently playing or completed)
      const isCompleted = statusMessage.textContent === 'Solved' || statusMessage.textContent === 'Failed';
      // Only reset controls when we're not actively playing/replaying and
      // when there is no in-progress initialization/step progress being reported.
      if (!isPuzzlePlaying && !isReplayPlaying && !isCompleted && !initProgressStarted) {
        setPlayingState(false);
      }
      // Hide progress bar when puzzle is fully loaded
      initProgressDiv.style.display = 'none';
    }
  } else if (data.type === 'status') {
    if (typeof data.message === 'string') {
      statusMessage.textContent = data.message;
    }
    if (data.state === 'loading_hints' || data.state === 'initializing') {
      statusMessage.className = 'loading';
    } else {
      statusMessage.className = '';
    }
  } else if (data.type === 'init_progress') {
    // Update progress bar
    const current = data.current || 0;
    const total = data.total || 1;
    const percentage = data.percentage || 0;
    if (!initProgressStarted) {
      // Set progress bar label based on operation
      if (data.operation === 'step') {
        statusMessage.textContent = 'Step progress...';
      } else {
        statusMessage.textContent = 'Initializing entries...';
      }
      statusMessage.className = '';
      initProgressStarted = true;
    }
    progressText.textContent = `${current} / ${total} entries`;
    progressBarFill.style.width = `${percentage}%`;
    initProgressDiv.style.display = 'flex';
  } else if (data.type === 'event') {
    const ev = data.event;
    log('[event] ' + formatEvent(ev));
    if (DEBUG_LOG_JSON) {
      log('[event.json] ' + JSON.stringify(ev));
    }
    // Update status based on event
    if (ev.event === 'solved') {
      statusMessage.textContent = 'Solved';
      statusMessage.className = 'status-solved';
      isPuzzlePlaying = false;
      setCompletedState();
      // Only prompt to save recording in puzzle mode
      if (!isReplayMode) {
        setTimeout(() => {
          if (window.confirm('Puzzle solved! Do you want to save this recording?')) {
            fetch('/api/save_recording', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
            });
          }
        }, 100);
      }
    } else if (ev.event === 'failed') {
      statusMessage.textContent = 'Failed';
      statusMessage.className = 'status-failed';
      isPuzzlePlaying = false;
      setCompletedState();
    } else if (ev.event === 'placed' || ev.event === 'placed_fallback' || ev.event === 'backtrack') {
      if (!isReplayMode) {
        statusMessage.textContent = 'Solving...';
      } else {
        statusMessage.textContent = 'Replaying...';
      }
      statusMessage.className = '';
    }
    // highlight placed/verified entries for a moment
    if (ev.event === 'placed' || ev.event === 'placed_fallback') {
      highlightEntry(ev.candidate.entry_id);
    }
    if (ev.event === 'candidate_verified') {
      highlightEntry(ev.entry_id);
    }
    if (Array.isArray(ev.verified)) {
      ev.verified.forEach((eid) => {
        highlightEntry(eid);
        log(`verified ${eid}`);
      });
    }
  } else if (data.type === 'progress_done') {
    log("ALL DONE")
    // Hide progress bar and reset progress state
    initProgressDiv.style.display = 'none';
    progressText.textContent = '0 / 0 entries';
    progressBarFill.style.width = '0%';
    initProgressStarted = false;
  }
}

// ============ REPLY MODE FUNCTIONALITY ============

function updateReplayProgress() {
  currentStepSpan.textContent = replayIndex;
  if (currentRecording) {
    totalStepsSpan.textContent = currentRecording.event_count || 0;
  }
}

function replayPlay() {
  if (!currentRecording || !isReplayMode) return;

  isReplayPlaying = true;
  setPlayingState(true);
  
  const playStep = () => {
    if (!isReplayPlaying || !currentRecording) return;
    
    const event = currentRecording.events?.[replayIndex];
    if (!event) {
      isReplayPlaying = false;
      return;
    }
    
    // Track fallback placements and backtracking
    const entryId = event.candidate?.entry_id;
    if (event.event === 'placed_fallback' && entryId) {
      replayFallbackEntries.add(entryId);
    } else if (event.event === 'backtrack' && entryId) {
      replayFallbackEntries.delete(entryId);
      replayBacktrackCount++;
    }
    
    applyReplayEventToGrid(event, replayGridState, entryToCells, replayPlacedEntries);
    replayEntries = computeReplayEntriesFromGrid(replayBaseEntries, replayGridState, entryToCells, replayFallbackEntries);
    
    // Count fallbacks from current entries
    const fallbacks = Object.values(replayEntries).filter(e => e.used_fallback).length;

    // Dispatch through the same message processor as WebSocket uses
    processMessage({ type: 'event', event });
    processMessage({
      type: 'state',
      grid: replayGridState,
      entries: replayEntries,
      metrics: {
        puzzle_id: currentRecording.puzzle_id,
        steps: replayIndex + 1,
        fallbacks: fallbacks,
        backtracks: replayBacktrackCount,
      },
    });
    
    replayIndex++;
    updateReplayProgress();
    
    if (replayIndex >= (currentRecording.event_count || 0)) {
      isReplayPlaying = false;
      // Check if the last event was solved/failed - if so, keep buttons disabled
      const lastEvent = currentRecording.events?.[replayIndex - 1];
      if (lastEvent?.event === 'solved' || lastEvent?.event === 'failed') {
        // setCompletedState() was already called by processMessage above
        // Don't call setPlayingState(false) as it would re-enable buttons
      } else {
        setPlayingState(false);
      }
      return;
    }
    
    const delay = (800 / replaySpeed);
    replayTimer = setTimeout(playStep, delay);
  };
  
  playStep();
}

function replayPause() {
  isReplayPlaying = false;
  if (replayTimer) clearTimeout(replayTimer);
  // Don't re-enable buttons if puzzle is already completed
  const isCompleted = statusMessage.textContent === 'Solved' || statusMessage.textContent === 'Failed';
  if (!isCompleted) {
    setPlayingState(false);
  }
}

function replayStep() {
  if (!currentRecording || !isReplayMode) return;
  
  // Pause if currently playing
  replayPause();
  
  const event = currentRecording.events?.[replayIndex];
  if (!event) return;
  
  // Track fallback placements and backtracking
  const entryId = event.candidate?.entry_id;
  if (event.event === 'placed_fallback' && entryId) {
    replayFallbackEntries.add(entryId);
  } else if (event.event === 'backtrack' && entryId) {
    replayFallbackEntries.delete(entryId);
    replayBacktrackCount++;
  }
  
  applyReplayEventToGrid(event, replayGridState, entryToCells, replayPlacedEntries);
  replayEntries = computeReplayEntriesFromGrid(replayBaseEntries, replayGridState, entryToCells, replayFallbackEntries);
  
  const fallbacks = Object.values(replayEntries).filter(e => e.used_fallback).length;
  
  processMessage({ type: 'event', event });
  processMessage({
    type: 'state',
    grid: replayGridState,
    entries: replayEntries,
    metrics: {
      puzzle_id: currentRecording.puzzle_id,
      steps: replayIndex + 1,
      fallbacks: fallbacks,
      backtracks: replayBacktrackCount,
    },
  });
  
  replayIndex++;
  updateReplayProgress();
  
  // Note: if event was solved/failed, setCompletedState() was already called by processMessage
  // and buttons are already in the correct disabled state
}

function replayReset() {
  replayPause();
  replayIndex = 0;
  updateReplayProgress();
  logEl.textContent = '';
  statusMessage.textContent = 'Recording loaded';
  statusMessage.className = '';
  replayGridState = cloneJson(replayBaseGridState);
  replayFallbackEntries.clear();
  replayPlacedEntries.clear();
  replayBacktrackCount = 0;
  replayEntries = computeReplayEntriesFromGrid(replayBaseEntries, replayGridState, entryToCells, replayFallbackEntries);
  if (replayGridState) {
    processMessage({
      type: 'state',
      grid: replayGridState,
      entries: replayEntries,
      metrics: {
        puzzle_id: currentRecording?.puzzle_id ?? null,
        steps: 0,
        fallbacks: 0,
        backtracks: 0,
      },
    });
  }
}

function buildEntryToCellsMap(puzzleData) {
  const map = {};
  if (!puzzleData || !puzzleData.entries) {
    return map;
  }
  
  // For each entry, build a list of [row, col] coordinates
  for (const [entryId, entry] of Object.entries(puzzleData.entries)) {
    if (entry.cells && Array.isArray(entry.cells)) {
      map[entryId] = entry.cells;
    }
  }
  
  return map;
}

async function loadSelectedRecording() {
  recordingId = itemSelect.value
  try {
    const res = await fetch(`/api/recordings/${recordingId}`);
    if (!res.ok) throw new Error(res.statusText);
    currentRecording = await res.json();
    
    isReplayMode = true;
    replayIndex = 0;
    updateReplayProgress();
    
    statusMessage.textContent = 'Recording loaded';
    statusMessage.className = '';
    
    
    // Load puzzle data to get entry-to-cells mapping for replay
    try {
      const puzzleRes = await fetch(`/api/puzzle/${currentRecording.puzzle_id}`);
      if (puzzleRes.ok) {
        const puzzleData = await puzzleRes.json();
        if (puzzleData.error_type === 'puzzle_not_found') {
          statusMessage.textContent = `Error: The puzzle for this recording (ID: ${currentRecording.puzzle_id}) is not available.`;
          statusMessage.className = 'error';
          log(`[error] ${puzzleData.error}`);
          entriesAcrossList.innerHTML = '';
          entriesDownList.innerHTML = '';
          setClueHeadingsVisible(false);
          return;
        }
        entryToCells = buildEntryToCellsMap(puzzleData);
        // Build base entries template from puzzle data
        const entriesForDisplay = {};
        for (const [entryId, entry] of Object.entries(puzzleData.entries)) {
          entriesForDisplay[entryId] = {
            pattern: '.'.repeat(entry.length),
            clue: entry.clue,
            correct_answer: entry.answer,
            used_fallback: false,
            verified: false,
          };
        }
        // Initialize replay base state
        replayBaseGridState = cloneJson(currentRecording.grid_state);
        replayGridState = cloneJson(currentRecording.grid_state);
        // Initialize empty letters matrix (not stored in recording)
        const rows = replayGridState.rows || currentRecording.height;
        const cols = replayGridState.cols || currentRecording.width;
        replayBaseGridState.letters = Array(rows).fill(null).map(() => Array(cols).fill(''));
        replayGridState.letters = Array(rows).fill(null).map(() => Array(cols).fill(''));
        // Compute clue numbers from puzzle data
        const numbers = Array(rows).fill(null).map(() => Array(cols).fill(null));
        for (const [entryId, cells] of Object.entries(entryToCells)) {
          if (cells && cells.length > 0) {
            const [r, c] = cells[0]; // First cell of the entry
            const clueNumber = entryId.slice(0, -1); // Remove direction suffix (A or D)
            numbers[r][c] = clueNumber;
          }
        }
        replayBaseGridState.numbers = numbers;
        replayGridState.numbers = numbers;
        replayBaseEntries = cloneJson(entriesForDisplay);
        replayFallbackEntries.clear();
        replayPlacedEntries.clear();
        replayBacktrackCount = 0;
        replayEntries = computeReplayEntriesFromGrid(replayBaseEntries, replayGridState, entryToCells, replayFallbackEntries);
        // Dispatch initial state through unified message path
        processMessage({
          type: 'state',
          grid: replayGridState,
          entries: replayEntries,
          metrics: {
            puzzle_id: currentRecording.puzzle_id,
            steps: 0,
            fallbacks: 0,
            backtracks: 0,
          },
        });
      }
    } catch (err) {
      log('[warning] Could not load puzzle for entry mapping: ' + err);
      // Clear entries if puzzle load fails
      entriesAcrossList.innerHTML = '';
      entriesDownList.innerHTML = '';
      setClueHeadingsVisible(false);
    }
    
    setPlayingState(false);
    replaySpeedControl.style.display = 'block';
    log(`[recording] Loaded: ${currentRecording.puzzle} (${currentRecording.event_count} events)`);

    // Remove the "-- select recording --" option once a recording is loaded
    const defaultOption = itemSelect.querySelector('option[value=""]');
    if (defaultOption) {
      defaultOption.remove();
    }
  } catch (err) {
    log('[error] Could not load recording: ' + err);
    statusMessage.textContent = 'Error loading recording';
    statusMessage.className = '';
  }
}

// Update speed label when slider changes
replaySpeedSlider.addEventListener('input', (e) => {
  replaySpeed = parseFloat(e.target.value);
  speedLabel.textContent = replaySpeed.toFixed(2) + 'x';
});

// Mode select handler
modeToggle.addEventListener('change', () => {
  switchMode(modeToggle.checked ? 'recordings' : 'puzzles');
});

function connect() {
  ws = new WebSocket('ws://localhost:8000/ws');

  ws.onopen = () => {
    log('[ws] connected');
  };

  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    processMessage(data);
  };

  ws.onclose = () => {
    log('[ws] disconnected, retrying in 1s...');
    setTimeout(connect, 1000);
  };
}

function createGridTable(rows, cols) {
  const table = document.createElement('table');
  table.id = 'grid';
  for (let r = 0; r < rows; r++) {
    const tr = document.createElement('tr');
    for (let c = 0; c < cols; c++) {
      const td = document.createElement('td');
      td.dataset.row = r;
      td.dataset.col = c;
      const number = document.createElement('span');
      number.className = 'clue-number';
      number.textContent = '';
      const letter = document.createElement('span');
      letter.className = 'cell-letter';
      letter.textContent = '';
      td.appendChild(number);
      td.appendChild(letter);
      tr.appendChild(td);
    }
    table.appendChild(tr);
  }
  return table;
}

function renderGrid(grid) {
  if (!grid) {
    gridContainer.innerHTML = '';
    entriesAcrossList.innerHTML = '';
    entriesDownList.innerHTML = '';
    return;
  }
  gridContainer.innerHTML = '';
  const table = createGridTable(grid.rows, grid.cols);
  const letters = grid.letters || grid.cells || [];
  const blocks = grid.blocks || null;
  const numbers = grid.numbers || null;
  const cheated = grid.cheated || null;
  // fill letters
  for (let r = 0; r < grid.rows; r++) {
    for (let c = 0; c < grid.cols; c++) {
      const letter = letters?.[r]?.[c] ?? null;
      const isBlock = blocks ? blocks?.[r]?.[c] : false;
      const number = numbers ? numbers?.[r]?.[c] : null;
      const isCheated = cheated ? cheated?.[r]?.[c] : false;
      const td = table.querySelector(`td[data-row="${r}"][data-col="${c}"]`);
      const numberEl = td.querySelector('.clue-number');
      const letterEl = td.querySelector('.cell-letter');
      td.classList.toggle('cell-block', isBlock);
      td.classList.toggle('cell-cheated', !isBlock && !!isCheated);
      numberEl.textContent = number || '';
      letterEl.textContent = letter || '';
    }
  }
  gridContainer.appendChild(table);
}

function renderEntries(entries) {
  entriesAcrossList.innerHTML = '';
  entriesDownList.innerHTML = '';
  
  // Handle null/undefined entries
  if (!entries || typeof entries !== 'object') {
    return;
  }
  
  const acrossEntries = [];
  const downEntries = [];
  
  for (const [eid, info] of Object.entries(entries)) {
    if (eid.endsWith('A')) {
      acrossEntries.push([eid, info]);
    } else if (eid.endsWith('D')) {
      downEntries.push([eid, info]);
    }
  }
  
  // Sort by numeric part
  const sortByNumber = (a, b) => {
    const numA = parseInt(a[0].slice(0, -1));
    const numB = parseInt(b[0].slice(0, -1));
    return numA - numB;
  };
  
  acrossEntries.sort(sortByNumber);
  downEntries.sort(sortByNumber);
  
  for (const [eid, info] of acrossEntries) {
    const li = document.createElement('li');
    li.id = `entry-${eid}`;
    const displayNum = eid.slice(0, -1); // strip A/D
    li.textContent = `${displayNum}: ${info.pattern} — ${info.clue}`;
    // Highlighting: fallback (orange) > incorrect (red) > correct (yellow)
    if (info.used_fallback) {
      li.classList.add('entry-incorrect');
    } else if (info.correct_answer) {
      if (info.pattern === info.correct_answer) {
        li.classList.add('entry-correct');
      } else if (!info.pattern.includes('.')) {
        li.classList.add('entry-fallback');
      }
    }
    entriesAcrossList.appendChild(li);
  }

  for (const [eid, info] of downEntries) {
    const li = document.createElement('li');
    li.id = `entry-${eid}`;
    const displayNum = eid.slice(0, -1); // strip A/D
    li.textContent = `${displayNum}: ${info.pattern} — ${info.clue}`;
    // Highlighting: fallback (orange) > incorrect (red) > correct (yellow)
    if (info.used_fallback) {
      li.classList.add('entry-incorrect');
    } else if (info.correct_answer) {
      if (info.pattern === info.correct_answer) {
        li.classList.add('entry-correct');
      } else if (!info.pattern.includes('.')) {
        li.classList.add('entry-fallback');
      }
    }
    entriesDownList.appendChild(li);
  }
}

function highlightEntry(eid) {
  const el = document.getElementById(`entry-${eid}`);
  if (!el) return;
  el.classList.add('entry-highlight');
  setTimeout(() => el.classList.remove('entry-highlight'), 900);
}

const API_BASE = 'http://localhost:8000';

async function fetchJson(path) {
  const res = await fetch(API_BASE + path);
  if (!res.ok) throw new Error(res.statusText);
  return res.json();
}

async function loadSelectedPuzzle() {
  const puzzleId = itemSelect.value;
  if (!puzzleId) return;
  
  // Clear old puzzle display
  gridContainer.innerHTML = '';
  
  // Hide clue sections
  const clueColumns = document.querySelectorAll('.clue-column h3');
  clueColumns.forEach(h3 => h3.style.display = 'none');
  const cluesList = document.querySelectorAll('.clue-column ul');
  cluesList.forEach(ul => ul.innerHTML = '');
  
  // Disable controls and show loading message
  statusMessage.textContent = 'Getting hints...';
  statusMessage.className = '';
  disableControls(true);
  logEl.textContent = '';
  
  // Reset progress bar (but don't show it yet - wait for first progress update)
  progressText.textContent = '0 / 0 entries';
  progressBarFill.style.width = '0%';
  initProgressDiv.style.display = 'none';
  initProgressStarted = false;
  
  await postAction(`/puzzles/${encodeURIComponent(puzzleId)}/load`);
  
  // Remove the "-- select puzzle --" option once a puzzle is loaded
  const defaultOption = itemSelect.querySelector('option[value=""]');
  if (defaultOption) {
    defaultOption.remove();
  }
}

async function postAction(path) {
  disableControls(true);
  try {
    const res = await fetch(API_BASE + path, {method: 'POST'});
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    log(`[http] ${path} -> ${JSON.stringify(data)}`);
    return data;
  } catch (err) {
    log('[http] error: ' + err);
  } finally {
    // Don't re-enable controls if we're actively playing or if puzzle is completed
    const isCompleted = statusMessage.textContent === 'Solved' || statusMessage.textContent === 'Failed';
    if (!isPuzzlePlaying && !isReplayPlaying && !isCompleted) {
      setPlayingState(false);
    }
  }
}

playBtn.addEventListener('click', () => {
  if (isReplayMode) {
    replayPlay();
  } else {
    isPuzzlePlaying = true;
    postAction('/play');
  }
  setPlayingState(true);
});

pauseBtn.addEventListener('click', () => {
  if (isReplayMode) {
    replayPause();
  } else {
    isPuzzlePlaying = false;
    postAction('/pause');
  }
  // Don't re-enable buttons if puzzle is already completed
  const isCompleted = statusMessage.textContent === 'Solved' || statusMessage.textContent === 'Failed';
  if (!isCompleted) {
    setPlayingState(false);
  }
});

stepBtn.addEventListener('click', () => {
  if (isReplayMode) {
    replayStep();
  } else {
    postAction('/step');
  }
});

resetBtn.addEventListener('click', () => {
  if (isReplayMode) {
    replayReset();
  } else {
    // Clear UI immediately for responsive feel
    renderGrid(null);
    renderEntries(null);
    statusMessage.textContent = 'Resetting...';
    statusMessage.className = '';
    logEl.textContent = '';
    renderTallyFromState({steps: 0, fallbacks: 0, backtracks: 0});
    
    // Then call server to reinitialize
    postAction('/reset');
  }
});

itemSelect.addEventListener('change', async () => {
  if (currentMode === 'puzzles') {
    await loadSelectedPuzzle();
  } else if (currentMode === 'recordings') {
    await loadSelectedRecording();
  }
});

connect();

refreshItemList();

renderTallyFromState({steps: 0, fallbacks: 0, backtracks: 0});

// Disable controls until something is loaded
disableControls(true);

