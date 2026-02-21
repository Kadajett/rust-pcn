/** Tail a JSONL metrics file and emit parsed events. */

import * as fs from "node:fs";
import type { TrainingEvent } from "./types.js";

export type EventCallback = (event: TrainingEvent) => void;

export interface DataReader {
  start(): void;
  stop(): void;
}

export function createDataReader(filePath: string, onEvent: EventCallback): DataReader {
  let offset = 0;
  let watcher: fs.FSWatcher | null = null;
  let pollInterval: ReturnType<typeof setInterval> | null = null;
  let buffer = "";

  function readNewData(): void {
    let stat: fs.Stats;
    try {
      stat = fs.statSync(filePath);
    } catch {
      return; // File doesn't exist yet
    }

    if (stat.size <= offset) return;

    const fd = fs.openSync(filePath, "r");
    const readSize = stat.size - offset;
    const buf = Buffer.alloc(readSize);
    fs.readSync(fd, buf, 0, readSize, offset);
    fs.closeSync(fd);

    offset = stat.size;
    buffer += buf.toString("utf-8");

    // Process complete lines
    const lines = buffer.split("\n");
    // Keep the last (potentially incomplete) line in the buffer
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      try {
        const event = JSON.parse(trimmed) as TrainingEvent;
        onEvent(event);
      } catch {
        // Skip malformed lines
      }
    }
  }

  return {
    start() {
      // Read any existing content first
      readNewData();

      // Watch for changes
      try {
        watcher = fs.watch(filePath, () => readNewData());
      } catch {
        // File might not exist yet, fall through to polling
      }

      // Also poll periodically as a fallback (fs.watch can miss events)
      pollInterval = setInterval(readNewData, 500);
    },

    stop() {
      if (watcher) {
        watcher.close();
        watcher = null;
      }
      if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
      }
    },
  };
}
