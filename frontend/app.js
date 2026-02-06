const logEl = document.getElementById("log");
const gridContainer = document.getElementById("grid-container");
const entriesAcrossList = document.getElementById("entries-across");
const entriesDownList = document.getElementById("entries-down");
const statusMessage = document.getElementById("status-message");

const playBtn = document.getElementById("play");
const pauseBtn = document.getElementById("pause");
const stepBtn = document.getElementById("step");
const solveBtn = document.getElementById("solve");
const resetBtn = document.getElementById("reset");

const modeToggle = document.getElementById('mode-toggle');
const itemSelect = document.getElementById('item-select');
const itemSelectLabel = document.getElementById('item-select-label');

const stepCountEl = document.getElementById('step-count');
const fallbackCountEl = document.getElementById('fallback-count');

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
let replaySpeed = 1.0;
let replayTimer = null;

function disableControls(disabled) {
  playBtn.disabled = disabled;
  pauseBtn.disabled = disabled;
  stepBtn.disabled = disabled;
  solveBtn.disabled = disabled;
  resetBtn.disabled = disabled;
}

function renderTallyFromState(metrics) {
  const steps = metrics?.steps ?? 0;
  const fallbacks = metrics?.fallbacks ?? 0;
  if (stepCountEl) stepCountEl.textContent = String(steps);
  if (fallbackCountEl) fallbackCountEl.textContent = String(fallbacks);
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
    solveBtn.style.display = 'inline-block';
    isReplayMode = false;
    if (replayTimer) clearTimeout(replayTimer);
    isReplayPlaying = false;
    await refreshItemList();
  } else if (mode === 'recordings') {
    itemSelectLabel.textContent = 'Recording:';
    replaySpeedControl.style.display = 'block';
    solveBtn.style.display = 'none';
    await refreshItemList();
  }
  
  // Reset state
  itemSelect.value = '';
  gridContainer.innerHTML = '';
  entriesAcrossList.innerHTML = '';
  entriesDownList.innerHTML = '';
  statusMessage.textContent = mode === 'puzzles' ? 'Select a puzzle' : 'Select a recording';
  statusMessage.className = '';
  logEl.textContent = '';
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
        opt.value = rec.id;
        opt.textContent = `${rec.puzzle_id} (${rec.width}x${rec.height}) - ${rec.event_count} events`;
        itemSelect.appendChild(opt);
      });
    } catch (err) {
      log('[error] Could not load recordings: ' + err);
    }
  }
}

// ============ REPLAY MODE FUNCTIONALITY ============

function updateReplayProgress() {
  currentStepSpan.textContent = replayIndex;
  if (currentRecording) {
    totalStepsSpan.textContent = currentRecording.event_count || 0;
  }
}

function applyReplayEvent(event) {
  if (!event) return;
  
  log('[event] ' + formatEvent(event));
  
  if (event.event === 'solved') {
    statusMessage.textContent = 'Solved';
    statusMessage.className = 'status-solved';
  } else if (event.event === 'failed') {
    statusMessage.textContent = 'Failed';
    statusMessage.className = 'status-failed';
  } else if (event.event === 'placed' || event.event === 'placed_fallback' || event.event === 'backtrack') {
    statusMessage.textContent = 'Replaying...';
    statusMessage.className = '';
  }
  
  if (event.event === 'placed' || event.event === 'placed_fallback') {
    if (event.candidate?.entry_id) {
      highlightEntry(event.candidate.entry_id);
    }
  }
  if (event.event === 'candidate_verified' && event.entry_id) {
    highlightEntry(event.entry_id);
  }
  if (Array.isArray(event.verified)) {
    event.verified.forEach((eid) => highlightEntry(eid));
  }
}

function replayStep() {
  if (!currentRecording || !isReplayMode) return;
  
  const event = currentRecording.events?.[replayIndex];
  if (!event) return;
  
  applyReplayEvent(event);
  replayIndex++;
  updateReplayProgress();
  
  if (replayIndex >= (currentRecording.event_count || 0)) {
    isReplayPlaying = false;
  }
}

function replayPlay() {
  if (!currentRecording || !isReplayMode) return;
  
  isReplayPlaying = true;
  
  const playStep = () => {
    if (!isReplayPlaying || !currentRecording) return;
    
    const event = currentRecording.events?.[replayIndex];
    if (!event) {
      isReplayPlaying = false;
      return;
    }
    
    applyReplayEvent(event);
    replayIndex++;
    updateReplayProgress();
    
    if (replayIndex >= (currentRecording.event_count || 0)) {
      isReplayPlaying = false;
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
}

function replayReset() {
  replayPause();
  replayIndex = 0;
  updateReplayProgress();
  logEl.textContent = '';
  statusMessage.textContent = 'Recording loaded';
  statusMessage.className = '';
  if (currentRecording?.grid_state) {
    renderGrid(currentRecording.grid_state);
  }
}

async function loadRecording(recordingId) {
  try {
    const res = await fetch(`/api/recordings/${recordingId}`);
    if (!res.ok) throw new Error(res.statusText);
    currentRecording = await res.json();
    
    isReplayMode = true;
    replayIndex = 0;
    updateReplayProgress();
    
    statusMessage.textContent = 'Recording loaded';
    statusMessage.className = '';
    
    if (currentRecording.grid_state) {
      renderGrid(currentRecording.grid_state);
    }
    
    // Clear entries (recordings don't have live entry state)
    entriesAcrossList.innerHTML = '';
    entriesDownList.innerHTML = '';
    
    disableControls(false);
    log(`[recording] Loaded: ${currentRecording.puzzle} (${currentRecording.event_count} events)`);
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
        const clueColumns = document.querySelectorAll('.clue-column h3');
        clueColumns.forEach(h3 => h3.style.display = 'none');
        disableControls(true);
      } else {
        // Puzzle loaded - show sections and "Loaded" message
        // Set status to Loaded when state is received (unless already solving/solved/failed)
        if (!statusMessage.textContent.match(/Solving|Solved|Failed/)) {
          statusMessage.textContent = 'Loaded';
          statusMessage.className = '';
        }
        // Show tally and headings
        document.getElementById('tally').style.display = 'block';
        const clueColumns = document.querySelectorAll('.clue-column h3');
        clueColumns.forEach(h3 => h3.style.display = 'block');
        // Enable controls now that puzzle is loaded
        disableControls(false);
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
        progressText.style.display = 'block';
        // Clear status message when progress starts
        statusMessage.textContent = 'Initializing entries...';
        statusMessage.className = '';
        initProgressStarted = true;
      }
      progressText.textContent = `${current} / ${total} entries`;
      progressBarFill.style.width = `${percentage}%`;
      initProgressDiv.style.display = 'block';
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
      } else if (ev.event === 'failed') {
        statusMessage.textContent = 'Failed';
        statusMessage.className = 'status-failed';
      } else if (ev.event === 'placed' || ev.event === 'placed_fallback' || ev.event === 'backtrack') {
        statusMessage.textContent = 'Solving...';
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
        ev.verified.forEach((eid) => highlightEntry(eid));
      }
    }
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
    if (info.used_fallback) li.classList.add('entry-highlight');
    if (info.verified) li.classList.add('entry-verified');
    entriesAcrossList.appendChild(li);
  }
  
  for (const [eid, info] of downEntries) {
    const li = document.createElement('li');
    li.id = `entry-${eid}`;
    const displayNum = eid.slice(0, -1); // strip A/D
    li.textContent = `${displayNum}: ${info.pattern} — ${info.clue}`;
    if (info.used_fallback) li.classList.add('entry-highlight');
    if (info.verified) li.classList.add('entry-verified');
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
    disableControls(false);
  }
}

playBtn.addEventListener('click', () => {
  if (isReplayMode) {
    replayPlay();
  } else {
    postAction('/play');
  }
});

pauseBtn.addEventListener('click', () => {
  if (isReplayMode) {
    replayPause();
  } else {
    postAction('/pause');
  }
});

stepBtn.addEventListener('click', () => {
  if (isReplayMode) {
    replayStep();
  } else {
    postAction('/step');
  }
});

solveBtn.addEventListener('click', () => postAction('/solve'));

resetBtn.addEventListener('click', () => {
  if (isReplayMode) {
    replayReset();
  } else {
    postAction('/reset');
    statusMessage.textContent = 'Loaded';
    statusMessage.className = '';
    logEl.textContent = '';
    renderTallyFromState({steps: 0, fallbacks: 0});
  }
});

itemSelect.addEventListener('change', async () => {
  const selectedValue = itemSelect.value;
  if (!selectedValue) return;
  
  if (currentMode === 'puzzles') {
    await loadSelectedPuzzle();
  } else if (currentMode === 'recordings') {
    await loadRecording(selectedValue);
  }
});

connect();

refreshItemList();

renderTallyFromState({steps: 0, fallbacks: 0});

// Disable controls until something is loaded
disableControls(true);
