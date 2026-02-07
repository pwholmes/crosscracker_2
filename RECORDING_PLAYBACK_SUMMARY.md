# Recording Playback Implementation Summary

## Overview
Added visual grid updates to the recording playback feature. When users select a recording from the dropdown and click play, the grid cells now update in real-time as events are replayed.

## Changes Made

### Frontend Changes (frontend/app.js)

#### 1. New Variables Added
```javascript
let entryToCells = {}; // mapping from entry_id to list of [row, col] for replay
```
- Tracks which grid cells correspond to each entry (across/down)
- Loaded from the puzzle metadata API when a recording is selected

#### 2. Enhanced `applyReplayEvent()` Function
**Location**: Lines 210-277

**Changes**:
- For `placed` and `placed_fallback` events: 
  - Updates grid cells with the placed letters
  - Uses `entryToCells` map to find cells for the entry_id
  - Writes letters from `event.candidate.answer` to the grid
  
- For `backtrack` events:
  - Clears the letters from grid cells
  - Removes visual indication of placement

**Implementation**:
```javascript
// Update grid cells for placed events
const cells = entryToCells[event.candidate.entry_id];
if (cells && event.candidate.answer) {
  const answer = event.candidate.answer;
  cells.forEach((cell, idx) => {
    if (idx < answer.length) {
      const [r, c] = cell;
      const td = document.querySelector(`#grid td[data-row="${r}"][data-col="${c}"]`);
      if (td) {
        const letterEl = td.querySelector('.cell-letter');
        if (letterEl) {
          letterEl.textContent = answer[idx];
        }
      }
    }
  });
}
```

#### 3. New `buildEntryToCellsMap()` Function
**Location**: Lines 344-356

**Purpose**: Converts puzzle metadata into a usable map of entry_id → cell coordinates

**Inputs**: 
- `puzzleData`: JSON from `/api/puzzle/{puzzle_id}` endpoint with entry metadata

**Returns**: Object mapping entry IDs to arrays of `[row, col]` pairs

#### 4. Enhanced `loadRecording()` Function
**Location**: Lines 358-397

**Changes**:
- Fetches puzzle metadata via `/api/puzzle/{currentRecording.puzzle}`
- Calls `buildEntryToCellsMap()` to prepare grid coordinate mapping
- Stores result in `entryToCells` global for use during replay

**Flow**:
```
User selects recording 
  ↓
loadRecording() fetches recording JSON
  ↓
Fetches puzzle metadata via /api/puzzle/{puzzle_id}
  ↓
buildEntryToCellsMap() creates entry → cells mapping
  ↓
Ready for replay with grid updates
```

### Backend Changes (backend/src/server.py)

#### 1. New Endpoint: GET /api/puzzle/{puzzle_id}
**Location**: Lines 414-445

**Purpose**: Provides puzzle metadata needed for replay grid updates

**Response Format**:
```json
{
  "puzzle_id": "simple 9x9",
  "width": 9,
  "height": 9,
  "entries": {
    "1A": {
      "clue": "Bar that connects rotating wheels",
      "answer": "AXLE",
      "length": 4,
      "cells": [[0,0], [0,1], [0,2], [0,3]]
    },
    "1D": {
      "clue": "A type of tree",
      "answer": "ASH",
      "length": 3,
      "cells": [[0,0], [1,0], [2,0]]
    }
    // ... more entries
  }
}
```

**Key Features**:
- Extracts cell coordinates from each entry's cells array
- Maps cells by their row/column positions
- Returns both clue and answer (used for verification only, replay uses events)

## How It Works - Recording Playback Flow

1. **User selects recording** from dropdown
2. **loadRecording()** is called:
   - Fetches recording JSON with events
   - Renders initial empty grid
   - Fetches puzzle metadata for entry-to-cells mapping
   - Stores mapping in `entryToCells`
3. **User clicks Play** → `replayPlay()` loops through events:
   - Calls `replayStep()` which calls `applyReplayEvent()` for each event
   - For `placed` events: updates grid cells with letters
   - For `backtrack` events: clears those grid cells
   - Visual animation shows solving progress in real-time
4. **Grid updates** happen instantly via DOM manipulation:
   - Direct cell update: `letterEl.textContent = answer[idx]`
   - No need to re-render entire grid
   - Smooth visual feedback of solver progress

## Event Types Handled

| Event Type | Action |
|-----------|--------|
| `placed` | Add letters to grid for the entry |
| `placed_fallback` | Add fallback letters (handled same as placed) |
| `backtrack` | Remove letters (clear cells for entry) |
| `solved` | Update status to "Solved" |
| `failed` | Update status to "Failed" |
| `candidate_verified` | Highlight entry (existing behavior) |

## Testing

The feature can be tested by:

1. Starting the server
2. Opening the UI in a browser
3. Triggering a puzzle solve (which creates a recording)
4. Switching to "Recordings" mode
5. Selecting the recording from the dropdown
6. Clicking "Play" to see the grid populate with letters as the solver's steps are replayed

## Dependencies & Assumptions

- **Frontend**: JavaScript DOM manipulation, existing `gridContainer` element with `#grid` ID
- **Backend**: Entry.cells array contains Cell objects with `.row` and `.col` properties
- **Data Format**: Recording JSON must include `puzzle` field (puzzle ID) to fetch metadata

## Future Improvements

- Add animation/transition effects for letter placement
- Show confidence levels during replay
- Add speed adjustment slider (already in UI, fully functional)
- Handle recording format changes if grid_state format evolves
