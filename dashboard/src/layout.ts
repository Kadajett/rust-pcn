/** Dashboard layout: creates all widgets and provides update handles. */

import {
  createWorld,
  createScreenEntity,
  createBoxEntity,
  createTextEntity,
  setText,
} from "blecsd";

import {
  createLineChart,
  createBarChart,
  createSparkline,
} from "blecsd/widgets";

import type { EpochEvent, EvalEvent, NewBookEvent, CheckpointEvent } from "./types.js";

// Color constants (RGBA hex)
const GREEN = 0x00ff00ff;
const CYAN = 0x00ffffff;
const YELLOW = 0xffff00ff;
const RED = 0xff4444ff;
const MAGENTA = 0xff00ffff;
const WHITE = 0xffffffff;

export interface DashboardWidgets {
  world: ReturnType<typeof createWorld>;
  screen: number;
  energyChart: ReturnType<typeof createLineChart>;
  accuracyChart: ReturnType<typeof createLineChart>;
  layerSparklines: ReturnType<typeof createSparkline>;
  bookChart: ReturnType<typeof createBarChart>;
  infoText: number;
  logText: number;
}

export interface DashboardState {
  energyData: number[];
  accuracyData: number[];
  layerErrorData: number[];
  bookAccuracies: Map<string, number>;
  logLines: string[];
  currentEpoch: number;
  totalBooks: number;
  totalSamples: number;
  lastElapsed: number;
}

export function createDashboardState(): DashboardState {
  return {
    energyData: [],
    accuracyData: [],
    layerErrorData: [],
    bookAccuracies: new Map(),
    logLines: ["PCN Training Dashboard", "Waiting for metrics..."],
    currentEpoch: 0,
    totalBooks: 0,
    totalSamples: 0,
    lastElapsed: 0,
  };
}

export function createDashboard(): DashboardWidgets {
  const cols = process.stdout.columns || 120;
  const rows = process.stdout.rows || 40;

  const world = createWorld();
  const screen = createScreenEntity(world, { width: cols, height: rows });

  const halfW = Math.floor(cols / 2);
  const chartH = Math.floor(rows * 0.35);
  const sparkH = 4;
  const infoH = 8;
  const logH = rows - chartH - sparkH - infoH - 2;

  // Left column: Energy chart
  const energyChart = createLineChart(world, {
    x: 0,
    y: 0,
    width: halfW,
    height: chartH,
    showLegend: true,
    showGrid: true,
    yLabel: "Energy",
    xLabel: "Epoch",
  });

  // Right column: Accuracy chart
  const accuracyChart = createLineChart(world, {
    x: halfW,
    y: 0,
    width: halfW,
    height: chartH,
    showLegend: true,
    showGrid: true,
    yLabel: "Accuracy %",
    xLabel: "Epoch",
    minY: 0,
    maxY: 100,
  });

  // Layer error sparklines (below charts, left)
  const layerSparklines = createSparkline(world, {
    x: 0,
    y: chartH,
    width: halfW,
  });

  // Per-book accuracy bar chart (below charts, right)
  const bookChart = createBarChart(world, {
    x: halfW,
    y: chartH,
    width: halfW,
    height: sparkH + infoH,
    showValues: true,
    yLabel: "Accuracy %",
  });

  // Info panel (below sparklines, left)
  createBoxEntity(world, {
    x: 0,
    y: chartH + sparkH,
    width: halfW,
    height: infoH,
  });
  const infoText = createTextEntity(world, {
    x: 1,
    y: chartH + sparkH + 1,
    width: halfW - 2,
    height: infoH - 2,
  });
  setText(world, infoText, "Initializing...");

  // Log area (bottom)
  createBoxEntity(world, {
    x: 0,
    y: chartH + sparkH + infoH,
    width: cols,
    height: logH,
  });
  const logText = createTextEntity(world, {
    x: 1,
    y: chartH + sparkH + infoH + 1,
    width: cols - 2,
    height: logH - 2,
  });
  setText(world, logText, "Waiting for training events...");

  return {
    world,
    screen,
    energyChart,
    accuracyChart,
    layerSparklines,
    bookChart,
    infoText,
    logText,
  };
}

export function updateDashboard(
  widgets: DashboardWidgets,
  state: DashboardState,
): void {
  const { world, energyChart, accuracyChart, layerSparklines, bookChart, infoText, logText } = widgets;

  // Update energy chart
  if (state.energyData.length > 0) {
    energyChart.setSeries([
      { label: "Energy", data: state.energyData, color: GREEN },
    ]);
  }

  // Update accuracy chart
  if (state.accuracyData.length > 0) {
    accuracyChart.setSeries([
      { label: "Accuracy", data: state.accuracyData.map((a) => a * 100), color: CYAN },
    ]);
  }

  // Update layer error sparklines
  if (state.layerErrorData.length > 0) {
    layerSparklines.setData(state.layerErrorData);
  }

  // Update per-book bar chart
  if (state.bookAccuracies.size > 0) {
    const labels = Array.from(state.bookAccuracies.keys());
    const data = Array.from(state.bookAccuracies.values()).map((a) => a * 100);
    bookChart.setLabels(labels);
    bookChart.setSeries([{ label: "Accuracy %", data, color: YELLOW }]);
  }

  // Update info text
  const infoLines = [
    `Epoch: ${state.currentEpoch}`,
    `Books: ${state.totalBooks}  Samples: ${state.totalSamples}`,
    `Energy: ${state.energyData.at(-1)?.toFixed(4) ?? "---"}`,
    `Accuracy: ${((state.accuracyData.at(-1) ?? 0) * 100).toFixed(2)}%`,
    `Epoch time: ${state.lastElapsed.toFixed(1)}s`,
    `Books: ${Array.from(state.bookAccuracies.keys()).join(", ") || "none"}`,
  ];
  setText(world, infoText, infoLines.join("\n"));

  // Update log (show last N lines that fit)
  const maxLogLines = 20;
  const recentLogs = state.logLines.slice(-maxLogLines);
  setText(world, logText, recentLogs.join("\n"));
}

export function handleEpochEvent(state: DashboardState, event: EpochEvent): void {
  state.currentEpoch = event.epoch;
  state.energyData.push(event.avg_energy);
  state.accuracyData.push(event.accuracy);
  state.totalSamples = event.num_samples;
  state.totalBooks = event.num_books;
  state.lastElapsed = event.elapsed_secs;

  if (event.layer_errors.length > 0) {
    // Keep a rolling window of layer errors for the sparkline
    state.layerErrorData.push(...event.layer_errors);
    if (state.layerErrorData.length > 100) {
      state.layerErrorData = state.layerErrorData.slice(-100);
    }
  }

  state.logLines.push(
    `[Epoch ${event.epoch}] energy=${event.avg_energy.toFixed(4)} accuracy=${(event.accuracy * 100).toFixed(2)}% (${event.elapsed_secs.toFixed(1)}s)`,
  );
}

export function handleEvalEvent(state: DashboardState, event: EvalEvent): void {
  state.bookAccuracies.set(event.book, event.accuracy);

  const predictions = event.sample_predictions
    .slice(0, 2)
    .map((p) => `"${p.input}"->"${p.predicted}"(${p.correct ? "OK" : "expected:" + p.expected})`)
    .join(" ");

  state.logLines.push(
    `  [${event.book}] accuracy=${(event.accuracy * 100).toFixed(2)}% ${predictions}`,
  );
}

export function handleNewBookEvent(state: DashboardState, event: NewBookEvent): void {
  state.logLines.push(
    `  NEW BOOK: ${event.book} (${event.samples} samples, ${event.train_samples} train, ${event.eval_samples} eval)`,
  );
}

export function handleCheckpointEvent(state: DashboardState, event: CheckpointEvent): void {
  state.logLines.push(`  Checkpoint saved: ${event.path}`);
}
