/** PCN Training Dashboard - blECSd terminal UI */

import {
  inputSystem,
  layoutSystem,
  renderSystem,
  outputSystem,
  cleanup,
  clearScreen,
  enableInput,
  enableKeys,
} from "blecsd";

import { createDirtyTracker } from "blecsd/core";
import { createDoubleBuffer, createScreenBuffer } from "blecsd/terminal";
import {
  setOutputStream,
  setOutputBuffer,
  setRenderBuffer,
  enterAlternateScreen,
  leaveAlternateScreen,
  hideCursor,
  showCursor,
} from "blecsd/systems";

import { createDataReader } from "./data-reader.js";
import {
  createDashboard,
  createDashboardState,
  updateDashboard,
  handleEpochEvent,
  handleEvalEvent,
  handleNewBookEvent,
  handleCheckpointEvent,
} from "./layout.js";
import type { TrainingEvent } from "./types.js";

// Get metrics file path from CLI args
const metricsPath = process.argv[2] || "data/output/metrics.jsonl";

// Terminal dimensions
const cols = process.stdout.columns || 120;
const rows = process.stdout.rows || 40;

// Initialize render pipeline buffers
setOutputStream(process.stdout);
const doubleBuffer = createDoubleBuffer(cols, rows);
setOutputBuffer(doubleBuffer);
const screenBuffer = createScreenBuffer(cols, rows);
const dirtyTracker = createDirtyTracker(cols, rows);
setRenderBuffer(dirtyTracker, screenBuffer);

// Initialize terminal
enterAlternateScreen();
hideCursor();
clearScreen();

// Create dashboard
const widgets = createDashboard();
const state = createDashboardState();

// Enable keyboard input for quit handling
enableInput(widgets.world, widgets.screen);
enableKeys(widgets.world, widgets.screen);

let needsRedraw = true;

// Handle training events
function onEvent(event: TrainingEvent): void {
  switch (event.type) {
    case "epoch":
      handleEpochEvent(state, event);
      break;
    case "eval":
      handleEvalEvent(state, event);
      break;
    case "new_book":
      handleNewBookEvent(state, event);
      break;
    case "checkpoint":
      handleCheckpointEvent(state, event);
      break;
  }
  needsRedraw = true;
}

// Start tailing the metrics file
const reader = createDataReader(metricsPath, onEvent);
reader.start();

// Render loop
function tick(): void {
  try {
    inputSystem(widgets.world);

    if (needsRedraw) {
      updateDashboard(widgets, state);
      needsRedraw = false;
    }

    layoutSystem(widgets.world);
    renderSystem(widgets.world);
    outputSystem(widgets.world);
  } catch (err) {
    // Swallow render errors to keep dashboard alive
    if (err instanceof Error) {
      state.logLines.push(`[Error] ${err.message}`);
      needsRedraw = true;
    }
  }
}

const renderInterval = setInterval(tick, 100); // 10 FPS

// Graceful shutdown
function shutdown(): void {
  clearInterval(renderInterval);
  reader.stop();
  try {
    showCursor();
    leaveAlternateScreen();
    cleanup();
  } catch {
    // Ignore cleanup errors
  }
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

// Also handle 'q' key to quit
process.stdin.on("data", (data: Buffer) => {
  const key = data.toString();
  if (key === "q" || key === "\x03") {
    shutdown();
  }
});
