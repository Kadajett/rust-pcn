#!/usr/bin/env python3
"""River Song run guard: archive every checkpoint generation and keep a crashed trainer from looping.

The trainer owns collapse handling (health gate, in-process rollback, `health.json`, exit code 78 when blocked).
This guard never restores checkpoints, never edits the unit file and never changes training flags. It only:

- Hard-links every `generation-*` directory the trainer writes into ARCHIVE_BASE/<root-name>/ (no extra disk
  until the trainer deletes its copy) and appends a row to index.jsonl. The `healthy` tag comes solely from
  `<root>/health.json` (`last_healthy.generation`). The newest KEEP archived generations are kept, plus the
  current `last_healthy` and every rollback target recorded in health.json.
- On trainer exit: status 78 or `health.json.blocked` -> publish one blocker and wait for a human;
  CUDA out of memory -> restart as-is after 30 s (at most 3 per hour, then blocker); any other failure ->
  restart after 60 s (at most 3 per 6 hours, then blocker); exit 0 -> log and wait.
Every action is appended to LOG; blockers are also published to the dashboard status card.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

UNIT = "river-song-training.service"
TELEMETRY = Path("/bulk-storage/connectome-merc/river-multimodal-runs/current")
ARCHIVE_BASE = Path("/bulk-storage/connectome-merc/river-checkpoint-archive")
LOG = Path("/bulk-storage/connectome-merc/river-gpu-trials/run-guard.jsonl")
PUBLISH = ["/home/kadajett/.pyenv/versions/3.12.12/bin/python3",
           "/home/kadajett/Dev/rust-pcn/scripts/push_river_status.py", "--message"]
KEEP = 8
POLL_SECONDS = 10
PARTIAL_MAX_AGE = 3600.0
BLOCKED_EXIT_STATUS = 78
OOM_MARKER = "CUDA_ERROR_OUT_OF_MEMORY"
OOM_DELAY, OOM_LIMIT, OOM_WINDOW = 30.0, 3, 3600.0
CRASH_DELAY, CRASH_LIMIT, CRASH_WINDOW = 60.0, 3, 6 * 3600.0

BLOCKED, OOM_RESTART, RESTART, FINISHED, WAIT = "blocked", "oom_restart", "restart", "finished", "wait"
GENERATION_NAME = re.compile(r"^generation-e(\d+)-b(\d+)-(\d+)$")


def now() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds")


def pacific() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%-I:%M %p %Z")


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------- pure decision logic (unit-tested)

def parse_output_root(execstart: str) -> Path:
    """Checkpoint root from `systemctl --user show -p ExecStart` output (argv is unquoted; --output has no spaces)."""
    match = re.search(r"--output\s+(\S+)", execstart)
    if not match:
        raise ValueError("no --output flag in ExecStart")
    return Path(match.group(1))


def parse_generation_name(name: str) -> tuple[int, int, int] | None:
    match = GENERATION_NAME.match(name)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3))) if match else None


def healthy_tag(health: dict, generation: str) -> bool | None:
    """True/False against health.json's last_healthy; None when the trainer has not recorded a healthy generation."""
    last = health.get("last_healthy") if isinstance(health, dict) else None
    if not isinstance(last, dict) or not last.get("generation"):
        return None
    return last["generation"] == generation


def protected_generations(health: dict) -> set[str]:
    names: set[str] = set()
    last = health.get("last_healthy")
    if isinstance(last, dict) and last.get("generation"):
        names.add(last["generation"])
    for rollback in health.get("rollbacks") or []:
        if isinstance(rollback, dict) and rollback.get("to_generation"):
            names.add(rollback["to_generation"])
    blocked = health.get("blocked")
    if isinstance(blocked, dict) and blocked.get("last_healthy_generation"):
        names.add(blocked["last_healthy_generation"])
    return names


def prune_plan(index_rows: list[dict], keep: int, protected: set[str]) -> list[str]:
    """Generations whose archive copy may be deleted: not among the newest `keep` rows and not protected."""
    newest = {row["generation"] for row in index_rows[-keep:]} if keep > 0 else set()
    return [row["generation"] for row in index_rows
            if row["generation"] not in newest and row["generation"] not in protected]


def stale_partials(entries: list[tuple[str, float]], now_epoch: float, max_age: float = PARTIAL_MAX_AGE) -> list[str]:
    """`.partial` staging dirs (name, mtime) older than max_age seconds."""
    return [name for name, mtime in entries if name.endswith(".partial") and now_epoch - mtime > max_age]


def journal_since_start(journal: str) -> str:
    """Tail of the journal from the last 'Started' line; the whole text when no start marker is present."""
    index = journal.rfind("Started ")
    if index == -1:
        return journal
    return journal[journal.rfind("\n", 0, index) + 1:]


def classify_exit(exec_main_status: int | None, health: dict, journal_tail: str) -> str:
    if (isinstance(health, dict) and health.get("blocked")) or exec_main_status == BLOCKED_EXIT_STATUS:
        return BLOCKED
    if OOM_MARKER in journal_since_start(journal_tail):
        return OOM_RESTART
    if exec_main_status == 0:
        return FINISHED
    if exec_main_status is None:
        return WAIT
    return RESTART


class RestartBudget:
    """At most `limit` events per sliding `window` seconds; clock injected for tests."""

    def __init__(self, limit: int, window: float, clock=time.monotonic) -> None:
        self.limit, self.window, self.clock = limit, window, clock
        self.events: list[float] = []

    def allow(self) -> bool:
        current = self.clock()
        self.events = [t for t in self.events if current - t < self.window]
        if len(self.events) >= self.limit:
            return False
        self.events.append(current)
        return True


# ---------------------------------------------------------------- systemd / filesystem plumbing

def systemctl_show(unit: str, *properties: str) -> dict[str, str]:
    args = ["systemctl", "--user", "show", unit]
    for prop in properties:
        args += ["-p", prop]
    out = subprocess.run(args, capture_output=True, text=True).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def checkpoint_root() -> Path:
    return parse_output_root(systemctl_show(UNIT, "ExecStart").get("ExecStart", ""))


def unit_status() -> dict:
    raw = systemctl_show(UNIT, "ExecMainStatus", "ExecMainCode", "Result", "ActiveState", "ExecMainExitTimestampMonotonic")
    status = {"active_state": raw.get("ActiveState", ""), "result": raw.get("Result", ""),
              "exec_main_code": raw.get("ExecMainCode", ""), "exec_main_status": None, "exit_key": None}
    try:
        status["exec_main_status"] = int(raw.get("ExecMainStatus", ""))
    except ValueError:
        pass
    stamp = raw.get("ExecMainExitTimestampMonotonic", "")
    status["exit_key"] = f"{stamp}:{raw.get('ExecMainStatus', '')}" if stamp not in ("", "0") else None
    return status


def journal_tail() -> str:
    return subprocess.run(["journalctl", "--user", "-u", UNIT, "-n", "200", "--no-pager", "-o", "cat"],
                          capture_output=True, text=True).stdout


class Guard:
    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.root = checkpoint_root()
        self.archive = ARCHIVE_BASE / self.root.name
        self.index = self.archive / "index.jsonl"
        if not dry_run:
            self.archive.mkdir(parents=True, exist_ok=True)
        self.health_snapshot: str | None = None
        self.handled_exit: str | None = None
        self.blocker_published_for: str | None = None
        self.oom_budget = RestartBudget(OOM_LIMIT, OOM_WINDOW)
        self.crash_budget = RestartBudget(CRASH_LIMIT, CRASH_WINDOW)

    # -- logging

    def log(self, record: dict, message: str | None = None) -> None:
        record = {"at": now(), **record}
        if message:
            record["message"] = message
        line = json.dumps(record)
        print(("DRY-RUN " if self.dry_run else "") + line, flush=True)
        if not self.dry_run:
            LOG.parent.mkdir(parents=True, exist_ok=True)
            with LOG.open("a") as stream:
                stream.write(line + "\n")

    def publish(self, message: str) -> None:
        if self.dry_run:
            print(f"DRY-RUN would publish: {message}", flush=True)
            return
        try:
            subprocess.run(PUBLISH + [message], timeout=120)
        except (OSError, subprocess.TimeoutExpired) as error:
            print(f"publish failed: {error}", flush=True)

    def systemctl(self, *args: str) -> None:
        if self.dry_run:
            print(f"DRY-RUN would run: systemctl --user {' '.join(args)}", flush=True)
            return
        subprocess.run(["systemctl", "--user", *args])

    # -- archive

    def rows(self) -> list[dict]:
        if not self.index.exists():
            return []
        return [json.loads(line) for line in self.index.read_text().splitlines() if line.strip()]

    def write_rows(self, rows: list[dict]) -> None:
        staged = self.index.with_suffix(".jsonl.tmp")
        staged.write_text("".join(json.dumps(row) + "\n" for row in rows))
        staged.replace(self.index)

    def health(self) -> dict:
        return read_json(self.root / "health.json")

    def source_generations(self) -> list[str]:
        names = [p.name for p in self.root.glob("generation-*") if p.is_dir() and parse_generation_name(p.name)]
        return sorted(names, key=parse_generation_name)

    def archive_new(self) -> None:
        archived = {row["generation"] for row in self.rows()}
        health = self.health()
        manifest = read_json(self.root / "experts.json")
        for generation in self.source_generations():
            if generation in archived or (self.archive / generation).exists():
                if self.dry_run:
                    print(f"DRY-RUN already archived {generation}", flush=True)
                continue
            parsed = parse_generation_name(generation)
            if self.dry_run:
                print(f"DRY-RUN would archive {generation} (healthy={healthy_tag(health, generation)})", flush=True)
                continue
            source = self.root / generation
            staged = self.archive / (generation + ".partial")
            shutil.rmtree(staged, ignore_errors=True)
            try:
                subprocess.run(["cp", "-al", str(source), str(staged)], check=True, capture_output=True)
                if manifest.get("generation") == generation:
                    (staged / "experts.json").write_text(json.dumps(manifest, indent=2) + "\n")
                staged.rename(self.archive / generation)
            except (subprocess.CalledProcessError, FileNotFoundError, OSError) as error:
                shutil.rmtree(staged, ignore_errors=True)
                self.log({"event": "archive_skipped", "generation": generation, "error": str(error)[:300]})
                continue
            row = {"generation": generation, "epoch": parsed[0], "batch": parsed[1], "archived_at": now(),
                   "healthy": healthy_tag(health, generation),
                   "last_healthy": (health.get("last_healthy") or {}).get("generation")}
            with self.index.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            self.log({"event": "archived", **row})
        self.prune(health)

    def prune(self, health: dict) -> None:
        rows = self.rows()
        for generation in prune_plan(rows, KEEP, protected_generations(health)):
            path = self.archive / generation
            if path.is_dir():
                if self.dry_run:
                    print(f"DRY-RUN would delete archived {generation}", flush=True)
                else:
                    shutil.rmtree(path, ignore_errors=True)
                    self.log({"event": "pruned", "generation": generation})
        if not self.archive.is_dir():
            return
        entries = [(p.name, p.stat().st_mtime) for p in self.archive.glob("*.partial") if p.is_dir()]
        for name in stale_partials(entries, time.time()):
            if self.dry_run:
                print(f"DRY-RUN would remove stale {name}", flush=True)
            else:
                shutil.rmtree(self.archive / name, ignore_errors=True)
                self.log({"event": "stale_partial_removed", "name": name})

    def sync_health(self) -> None:
        """Mirror health.json into the archive when it changes and retag the current last_healthy row."""
        source = self.root / "health.json"
        try:
            text = source.read_text()
        except (FileNotFoundError, OSError):
            return
        if text == self.health_snapshot:
            return
        self.health_snapshot = text
        if self.dry_run:
            print("DRY-RUN would copy health.json into the archive", flush=True)
            return
        shutil.copyfile(source, self.archive / "health.json")
        last = (read_json(source).get("last_healthy") or {}).get("generation")
        if last:
            rows = self.rows()
            changed = False
            for row in rows:
                if row["generation"] == last and row.get("healthy") is not True:
                    row["healthy"], changed = True, True
            if changed:
                self.write_rows(rows)

    # -- trainer exit handling

    def blocker(self, key: str, message: str, record: dict) -> None:
        if self.blocker_published_for == key:
            return
        self.blocker_published_for = key
        self.log({"event": "blocker", **record}, message)
        self.publish(message)

    def handle_exit(self, status: dict) -> None:
        health = self.health()
        action = classify_exit(status["exec_main_status"], health, journal_tail())
        key = status["exit_key"] or "no-exit"
        record = {"event": "trainer_exit", "action": action, "exec_main_status": status["exec_main_status"],
                  "result": status["result"], "active_state": status["active_state"]}
        if self.dry_run:
            print(f"DRY-RUN exit classification: {json.dumps(record)}", flush=True)
            return
        if action == BLOCKED:
            blocked = health.get("blocked") or {}
            reason = blocked.get("reason") or f"trainer exited with status {status['exec_main_status']}"
            last = blocked.get("last_healthy_generation") or (health.get("last_healthy") or {}).get("generation")
            self.blocker(f"blocked:{key}", f"BLOCKER {pacific()}: River Song trainer is blocked ({reason}). Last healthy "
                         f"generation: {last}. Run guard is waiting; needs a human (repair tool + systemctl --user "
                         f"start {UNIT}).", {**record, "reason": reason, "last_healthy": last})
            return
        if self.handled_exit == key:
            return
        self.handled_exit = key
        if action == FINISHED:
            self.log(record, f"Run guard {pacific()}: trainer exited 0; waiting.")
        elif action == OOM_RESTART:
            if self.oom_budget.allow():
                time.sleep(OOM_DELAY)
                self.systemctl("reset-failed", UNIT)
                self.systemctl("start", UNIT)
                self.log({**record, "restarted": True}, f"Run guard {pacific()}: GPU out of memory; restarted as-is.")
            else:
                self.blocker(f"oom:{key}", f"BLOCKER {pacific()}: River Song hit CUDA out of memory {OOM_LIMIT} times "
                             f"within an hour. Run guard stopped restarting; needs a human.", record)
        elif action == RESTART:
            if self.crash_budget.allow():
                time.sleep(CRASH_DELAY)
                self.systemctl("reset-failed", UNIT)
                self.systemctl("start", UNIT)
                self.log({**record, "restarted": True}, f"Run guard {pacific()}: trainer exited "
                         f"{status['exec_main_status']} ({status['result']}); restarted from its latest checkpoint.")
            else:
                self.blocker(f"crash:{key}", f"BLOCKER {pacific()}: River Song trainer failed {CRASH_LIMIT} times within "
                             f"{int(CRASH_WINDOW // 3600)} hours (last status {status['exec_main_status']}). Run guard "
                             f"stopped restarting; needs a human.", record)

    # -- main loop

    def poll(self) -> None:
        self.sync_health()
        self.archive_new()
        status = unit_status()
        if status["active_state"] in ("active", "activating", "deactivating", "reloading"):
            if self.blocker_published_for and not self.health().get("blocked"):
                self.blocker_published_for = None
            if self.dry_run:
                print(f"DRY-RUN unit is {status['active_state']}; classification if it exited now: "
                      f"{classify_exit(status['exec_main_status'], self.health(), journal_tail())} "
                      f"(ExecMainStatus={status['exec_main_status']}, Result={status['result']})", flush=True)
            return
        self.handle_exit(status)

    def run(self, once: bool = False) -> None:
        while True:
            try:
                self.poll()
            except Exception:  # noqa: BLE001 - the guard must never exit on its own
                self.log({"event": "guard_error", "traceback": traceback.format_exc()[-2000:]})
            if once:
                return
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print what would be done; no cp/systemctl/publish/log")
    parser.add_argument("--once", action="store_true", help="run a single poll iteration and exit")
    options = parser.parse_args()
    Guard(dry_run=options.dry_run).run(once=options.once)
