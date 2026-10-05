#!/usr/bin/env python3
"""River Song remote telemetry bridge: mirror a trainer's telemetry dir from a remote box over SSH, and forward
dashboard probes to it, so the local portal/helpers keep working unchanged against a local mirror directory.

Usage
    python3 scripts/river_remote_bridge.py [--host 4.71.129.4] [--user ubuntu] [--port 32317] \
        [--identity ~/.ssh/key | --password-file FILE] \
        --remote-telemetry /remote/telemetry/dir --local-mirror /local/mirror/dir \
        [--interval 3] [--probe-timeout 600] [--control-path /tmp/river-bridge-%C] [--control-persist 600] [--once]

    Defaults target the replacement B200 of Oct 5 2026 (ubuntu@4.71.129.4, port 32317; env RIVER_B200_HOST/USER/PORT
    override). The first instance (port 30204) was destroyed and must never be contacted; nothing on the new box
    survives from it. The new box is password-only until a key is installed:
      * recommended, once, interactively: `ssh-copy-id -p 32317 ubuntu@4.71.129.4`, then run with --identity
        (fully unattended reconnects);
      * or `--password-file FILE` (chmod 600) with `sshpass` installed (apt install sshpass): unattended;
      * or neither: the bridge opens the ssh master with a password prompt on its terminal and worker commands
        multiplex over it; the master outlives the bridge for --control-persist seconds (600), so restarts within
        that window need no password, but an SSH drop requires a human to retype it (the loop keeps retrying and
        prints the exact `ssh -M ...` command to open the master by hand when it has no terminal).
    --once runs exactly one mirror cycle (connect, mirror, forward probes) and exits 0 on success, 1 on a transport
    failure. Without --once the loop runs until SIGINT/SIGTERM, which closes the SSH ControlMaster (`ssh -O exit`).
    Logs go to stderr with Pacific (America/Los_Angeles) timestamps; only state changes and handoff events are logged.

Pointing the local tools at the mirror (no code changes needed; the mirror has the same file names as the remote dir):
    python3 portal/server.py --telemetry-dir <mirror> --token-file ...     (reads state.json, events.jsonl,
        samples.jsonl, audit-metrics.jsonl, manifest.json, promotion.json, capability-evidence.json; writes
        request.json / reads response.json for probes; keeps its own friday-status.sqlite3 in the dir)
    python3 scripts/live_prose_probe.py --telemetry <mirror> --url http://127.0.0.1:8799/api/test
        (reads <mirror>/state.json for the training status gate, appends its own rows to <mirror>/samples.jsonl;
        note those local appends are overwritten by the next mirror cycle, see "jsonl rule" below)
    python3 scripts/push_river_status.py --watch-training --url http://127.0.0.1:8799/api/friday-update
        (takes no telemetry flag: it reads /api/state from the portal, which reads the mirror; its systemd lookups
        describe the LOCAL river-song-training.service, so its service line is about the local box, not the remote)
    portal/auditor.py --telemetry-dir <mirror> ... writes audit.json / audit-metrics.jsonl into the dir it is given,
        so run the auditor against the mirror locally (nothing audit-related is expected from the remote).

Transport
    One SSH ControlMaster socket (`ssh -o ControlMaster=auto -o ControlPath=... -o ControlPersist=600 -N -f`) is
    opened once (key, sshpass, or an interactive password prompt; see Usage) and reused by every ssh/rsync call
    (rsync -e 'ssh -o ControlPath=... -o BatchMode=yes', which never prompts because it multiplexes over the master).
    Any ssh/rsync failure marks the connection dropped; the loop closes the master, waits 1, 2, 4, ... 30 s (capped),
    reconnects and retries. A successful cycle resets the backoff. The bridge never crashes on a drop.

Mirror rules (every ~3 s, one `find` listing + at most one rsync + one dd-over-ssh per growing jsonl file)
    * Whole-file rule (state.json, manifest.json, promotion.json, capability-evidence.json, audit.json, health-style
      JSON, anything that is not *.jsonl): fetched with rsync into <mirror>/.bridge-staging/ whenever the remote size
      or mtime changed, then moved into place with os.replace so a reader never sees a partial file.
    * jsonl rule (events.jsonl, samples.jsonl, audit-metrics.jsonl, events.before-*.jsonl ...): byte-offset tracking
      instead of `rsync --append-verify`. rsync --append skips a file whose destination is longer than the source,
      which is exactly what happens when the trainer restarts (it truncates events.jsonl at startup) and the file
      regrows slowly; the mirror would lag until the remote outgrew it and then need a full re-transfer. With
      offsets the bridge reads `dd skip=<local size - 256> count=...` over ssh, checks the 256-byte overlap against
      the local tail (detects truncate-and-regrow), appends the new bytes with O_APPEND + fsync, and on any
      shrink/mismatch atomically resets the local file to empty and re-streams from offset 0 in <= 32 MiB chunks per
      cycle. Local files are never deleted; remote files are never deleted except the probe handoff files below.
      Never uses rsync --delete or --inplace.
    * Skipped: request.json / request.processing.json / response.json / request.json.tmp (handoff, below),
      friday-status.sqlite3 (portal-local), dotfiles and *.tmp (writers' temporaries), subdirectories.

Probe handoff (portal/server.py enqueue_test <-> src/bin/train_universal.rs process_request)
    Portal: refuses with 409 while <dir>/request.json or <dir>/request.processing.json exists; unlinks response.json,
    writes request.json.tmp, renames to request.json, then polls response.json for a dict whose "id" matches for
    PROBE_TIMEOUT_SECONDS (600 s); on timeout unlinks request.json (if still there) and answers 504.
    Trainer (at batch boundaries): renames request.json -> request.processing.json, answers, writes response.json
    atomically, appends the probe to samples.jsonl, then removes request.processing.json. At startup it removes a
    leftover request.processing.json without answering it. A refusal/failure response is {"id", "ok": false,
    "answers": {}}, which the portal turns into HTTP 422.
    Bridge, each cycle:
      1. <mirror>/request.json present and nothing in flight -> remove any stale remote response.json, upload the
         bytes to <remote>/.request.json.bridge-tmp and `mv` them onto <remote>/request.json, then rename the local
         file to <mirror>/request.processing.json (the portal's busy check now reports the probe as in progress; its
         mtime is set to the upload time: the probe's deadline clock, never earlier than the portal's own, and it
         survives bridge restarts because in-flight state lives only on disk).
      2. In flight (<mirror>/request.processing.json present):
         - remote response.json exists: fetch it; if its id matches the in-flight id (or the request is unparseable)
           write <mirror>/response.json atomically (temp + fsync + replace + directory fsync), then remove the local
           processing marker, then remove the remote response.json. A non-matching id is stale: removed remotely.
         - remote request.processing.json exists: the trainer is answering; wait (never touch that remote file:
           the trainer's own remove_file would fail and abort the run).
         - remote request.json exists for longer than --probe-timeout: the trainer never picked it up; remove the
           remote request.json, drop the local marker and write a refusal response.json so a still-waiting portal
           answers 422 instead of hanging (the portal's own 600 s deadline has normally already passed).
         - none of the three remotely on two consecutive listings: the trainer restarted and discarded the probe;
           drop the local marker and write the refusal response.json.
      3. Nothing in flight and a remote response.json exists: stale, removed remotely and never mirrored.
    The only remote paths the bridge ever deletes are <remote>/request.json and <remote>/response.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PACIFIC = ZoneInfo("America/Los_Angeles")
REQUEST = "request.json"
PROCESSING = "request.processing.json"
RESPONSE = "response.json"
HANDOFF_FILES = {REQUEST, PROCESSING, RESPONSE, "request.json.tmp"}
REMOTE_DELETABLE = {REQUEST, RESPONSE}
LOCAL_ONLY_FILES = {"friday-status.sqlite3"}
APPEND_SUFFIX = ".jsonl"
STAGING_DIR = ".bridge-staging"
TEMP_SUFFIX = ".bridge-tmp"
OVERLAP_BYTES = 256
MAX_CHUNK_BYTES = 32 * 1024 * 1024
DEFAULT_PROBE_TIMEOUT = 600.0  # portal/server.py PROBE_TIMEOUT_SECONDS
BACKOFF_FIRST, BACKOFF_CAP = 1.0, 30.0
LOST_LISTINGS = 2  # consecutive listings without any handoff file before a probe counts as lost


def log(message: str) -> None:
    stamp = datetime.now(PACIFIC).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"{stamp} river-remote-bridge: {message}", file=sys.stderr, flush=True)


class TransportError(Exception):
    """Any ssh/rsync failure; the loop treats it as a dropped connection."""


@dataclass(frozen=True)
class RemoteFile:
    size: int
    stamp: str  # mtime as printed by find %T@, compared as text


class Transport:
    """Operations the bridge needs; SshTransport is the real one, tests subclass with a local directory."""

    def connect(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def listing(self) -> dict[str, RemoteFile]:
        raise NotImplementedError

    def fetch_whole(self, names: list[str], staging: Path) -> None:
        raise NotImplementedError

    def fetch_range(self, name: str, offset: int, length: int) -> bytes:
        raise NotImplementedError

    def put_file(self, name: str, data: bytes) -> None:
        raise NotImplementedError

    def remove(self, name: str) -> None:
        raise NotImplementedError


class SshTransport(Transport):
    def __init__(self, host: str, user: str | None, port: int, identity: Path | None, remote_dir: str,
                 control_path: str, password_file: Path | None = None, control_persist: int = 600,
                 command_timeout: float = 120.0, transfer_timeout: float = 900.0) -> None:
        self.destination = f"{user}@{host}" if user else host
        self.port = port
        self.identity = identity
        self.password_file = password_file
        self.control_persist = control_persist
        self.remote_dir = remote_dir.rstrip("/") or "/"
        self.control_path = control_path
        self.command_timeout = command_timeout
        self.transfer_timeout = transfer_timeout

    def ssh_options(self, batch: bool = True) -> list[str]:
        """Every worker command multiplexes over the master socket, so `BatchMode=yes` (no prompts) is right for
        them even on a password-only host; only the master open (`batch=False`) may prompt."""
        options = [
            "-o", f"ControlPath={self.control_path}", "-o", "ControlMaster=auto",
            "-o", f"ControlPersist={self.control_persist}",
            "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20", "-o", "Compression=yes",
            "-p", str(self.port),
        ]
        if batch:
            options += ["-o", "BatchMode=yes"]
        if self.identity is not None:
            options += ["-i", str(self.identity)]
        return options

    def run_ssh(self, remote_command: str, stdin: bytes | None = None, timeout: float | None = None) -> bytes:
        command = ["ssh", *self.ssh_options(), self.destination, remote_command]
        return self.run(command, stdin, timeout or self.command_timeout)

    def run_rsync(self, names: list[str], staging: Path) -> None:
        command = [
            "rsync", "--quiet", "--times", "--whole-file", "-s", "--from0", "--files-from=-",
            "--timeout=60", "-e", shlex.join(["ssh", *self.ssh_options()]),
            f"{self.destination}:{self.remote_dir}/", f"{staging}/",
        ]
        self.run(command, b"".join(name.encode() + b"\0" for name in names), self.transfer_timeout)

    @staticmethod
    def run(command: list[str], stdin: bytes | None, timeout: float) -> bytes:
        try:
            completed = subprocess.run(command, input=stdin, capture_output=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise TransportError(f"{command[0]}: {error}") from error
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()[-400:]
            raise TransportError(f"{command[0]} exited {completed.returncode}: {detail}")
        return completed.stdout

    def control(self, verb: str) -> subprocess.CompletedProcess:
        return subprocess.run(["ssh", *self.ssh_options(), "-O", verb, self.destination],
                              capture_output=True, timeout=30)

    def connect(self) -> None:
        """Open (or reuse) the ControlMaster. Key: unattended. Password file: unattended via sshpass.
        Neither: prompt on the terminal once (the master then outlives the bridge for --control-persist seconds);
        without a terminal the bridge cannot authenticate and says how to open the master by hand."""
        try:
            if self.control("check").returncode == 0:
                return
            master = ["ssh", *self.ssh_options(batch=False), "-N", "-f", self.destination]
            if self.password_file is not None:
                if shutil.which("sshpass") is None:
                    raise TransportError("--password-file needs `sshpass` (apt install sshpass); or install a key "
                                         "with ssh-copy-id and use --identity")
                self.run(["sshpass", "-f", str(self.password_file), *master], None, 60.0)
            elif self.identity is not None or not sys.stdin.isatty():
                try:
                    self.run(["ssh", *self.ssh_options(batch=True), "-N", "-f", self.destination], None, 60.0)
                except TransportError as error:
                    if self.identity is None:
                        raise TransportError(
                            f"{error}; no key and no terminal for a password prompt: open the master by hand with "
                            f"`ssh {shlex.join(self.ssh_options(batch=False))} -N -f {self.destination}` "
                            "(or pass --password-file with sshpass installed, or ssh-copy-id a key and use --identity)"
                        ) from error
                    raise
            else:
                log(f"opening ssh master to {self.destination} (password prompt on this terminal)")
                completed = subprocess.run(master, timeout=300)
                if completed.returncode != 0:
                    raise TransportError(f"ssh master exited {completed.returncode}")
        except (OSError, subprocess.TimeoutExpired) as error:
            raise TransportError(f"ssh master: {error}") from error

    def close(self) -> None:
        try:
            self.control("exit")
        except (OSError, subprocess.TimeoutExpired):
            pass

    def listing(self) -> dict[str, RemoteFile]:
        quoted = shlex.quote(self.remote_dir)
        output = self.run_ssh(
            f"if [ -d {quoted} ]; then cd {quoted} && find . -maxdepth 1 -type f -printf '%s\\t%T@\\t%P\\n'; fi"
        )
        return parse_listing(output.decode("utf-8", errors="replace"))

    def fetch_whole(self, names: list[str], staging: Path) -> None:
        self.run_rsync(names, staging)

    def fetch_range(self, name: str, offset: int, length: int) -> bytes:
        path = shlex.quote(f"{self.remote_dir}/{name}")
        return self.run_ssh(
            f"dd if={path} bs=1M iflag=skip_bytes,count_bytes skip={offset} count={length} status=none",
            timeout=self.transfer_timeout,
        )

    def put_file(self, name: str, data: bytes) -> None:
        directory = shlex.quote(self.remote_dir)
        temporary = shlex.quote(f"{self.remote_dir}/.{name}{TEMP_SUFFIX}")
        final = shlex.quote(f"{self.remote_dir}/{name}")
        self.run_ssh(f"mkdir -p {directory} && cat > {temporary} && mv -f {temporary} {final}", stdin=data)

    def remove(self, name: str) -> None:
        self.run_ssh(f"rm -f -- {shlex.quote(f'{self.remote_dir}/{name}')}")


def parse_listing(text: str) -> dict[str, RemoteFile]:
    files: dict[str, RemoteFile] = {}
    for line in text.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3 or not parts[0].isdigit():
            continue
        size, stamp, name = parts
        if name and "/" not in name:
            files[name] = RemoteFile(int(size), stamp)
    return files


def mirrored_name(name: str) -> bool:
    return not (name in HANDOFF_FILES or name in LOCAL_ONLY_FILES or name.startswith(".") or name.endswith(".tmp"))


def write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}{TEMP_SUFFIX}")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


class Bridge:
    def __init__(self, transport: Transport, mirror: Path, interval: float = 3.0,
                 probe_timeout: float = DEFAULT_PROBE_TIMEOUT, max_chunk: int = MAX_CHUNK_BYTES,
                 sleep=None, clock=time.time) -> None:
        self.transport = transport
        self.mirror = mirror
        self.interval = interval
        self.probe_timeout = probe_timeout
        self.max_chunk = max_chunk
        self.sleep = sleep or self.interruptible_sleep
        self.clock = clock
        self.stopped = False
        self.seen: dict[str, RemoteFile] = {}
        self.missing_listings = 0
        self.staging = mirror / STAGING_DIR

    # ---- loop -------------------------------------------------------------------------------------------------

    def stop(self, *_: object) -> None:
        self.stopped = True

    def interruptible_sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not self.stopped and (remaining := deadline - time.monotonic()) > 0:
            time.sleep(min(0.5, remaining))

    def run(self, once: bool = False) -> int:
        backoff = 0.0
        connected = False
        status = 0
        while not self.stopped:
            try:
                if not connected:
                    self.transport.connect()
                    connected = True
                    log("connected")
                self.run_once()
                if backoff:
                    log("recovered")
                backoff = 0.0
                status = 0
                if once:
                    break
                self.sleep(self.interval)
            except TransportError as error:
                connected = False
                status = 1
                backoff = min(BACKOFF_CAP, backoff * 2) if backoff else BACKOFF_FIRST
                log(f"transport failure: {error}; reconnecting in {backoff:.0f}s")
                self.transport.close()
                if once:
                    break
                self.sleep(backoff)
        self.transport.close()
        return status

    def run_once(self) -> None:
        self.mirror.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(exist_ok=True)
        listing = self.transport.listing()
        self.mirror_files(listing)
        self.forward_probe(listing)

    # ---- mirroring --------------------------------------------------------------------------------------------

    def mirror_files(self, listing: dict[str, RemoteFile]) -> None:
        whole = [name for name, remote in listing.items()
                 if mirrored_name(name) and not name.endswith(APPEND_SUFFIX) and self.seen.get(name) != remote]
        if whole:
            self.transport.fetch_whole(whole, self.staging)
            for name in whole:
                staged = self.staging / name
                if not staged.is_file():
                    continue  # vanished remotely between listing and fetch; retried next cycle
                os.replace(staged, self.mirror / name)
                self.seen[name] = listing[name]
            fsync_directory(self.mirror)
        for name, remote in listing.items():
            if mirrored_name(name) and name.endswith(APPEND_SUFFIX):
                self.sync_append(name, remote.size)

    def sync_append(self, name: str, remote_size: int, retry: bool = True) -> None:
        local = self.mirror / name
        local_size = local.stat().st_size if local.exists() else 0
        if remote_size == local_size:
            return
        if remote_size < local_size:
            log(f"{name}: remote shrank {local_size} -> {remote_size} bytes (trainer restart?); resetting mirror")
            write_atomic(local, b"")
            local_size = 0
        overlap = min(OVERLAP_BYTES, local_size)
        length = overlap + min(self.max_chunk, remote_size - local_size)
        data = self.transport.fetch_range(name, local_size - overlap, length)
        if overlap:
            with local.open("rb") as stream:
                stream.seek(local_size - overlap)
                tail = stream.read(overlap)
            if data[:overlap] != tail:
                if not retry:
                    raise TransportError(f"{name}: overlap mismatch persisted after reset")
                log(f"{name}: remote content diverged from the mirror tail; resetting mirror")
                write_atomic(local, b"")
                self.sync_append(name, remote_size, retry=False)
                return
        if len(data) > overlap:
            with local.open("ab") as stream:
                stream.write(data[overlap:])
                stream.flush()
                os.fsync(stream.fileno())

    # ---- probe handoff ----------------------------------------------------------------------------------------

    def remove_remote(self, name: str) -> None:
        if name not in REMOTE_DELETABLE:
            raise RuntimeError(f"refusing to delete remote file outside the handoff set: {name}")
        self.transport.remove(name)

    def forward_probe(self, listing: dict[str, RemoteFile]) -> None:
        local_request = self.mirror / REQUEST
        local_processing = self.mirror / PROCESSING
        if local_processing.exists():
            self.track_in_flight(listing, local_processing)
            return
        self.missing_listings = 0
        if local_request.exists():
            if RESPONSE in listing:
                self.remove_remote(RESPONSE)
            self.transport.put_file(REQUEST, local_request.read_bytes())
            os.replace(local_request, local_processing)
            started = self.clock()
            os.utime(local_processing, (started, started))  # probe start = upload time; survives bridge restarts
            fsync_directory(self.mirror)
            log(f"probe {self.request_id(local_processing)!r} uploaded")
        elif RESPONSE in listing:
            self.remove_remote(RESPONSE)
            log("stale remote response.json discarded")

    def track_in_flight(self, listing: dict[str, RemoteFile], local_processing: Path) -> None:
        request_id = self.request_id(local_processing)
        if RESPONSE in listing:
            data = self.transport.fetch_range(RESPONSE, 0, listing[RESPONSE].size)
            try:
                response = json.loads(data)
            except ValueError:
                response = None
            if request_id is not None and (not isinstance(response, dict) or response.get("id") != request_id):
                self.remove_remote(RESPONSE)
                log(f"stale remote response.json discarded while probe {request_id!r} is in flight")
            else:
                write_atomic(self.mirror / RESPONSE, data)
                local_processing.unlink(missing_ok=True)
                fsync_directory(self.mirror)
                self.remove_remote(RESPONSE)
                log(f"probe {request_id!r} answered")
            self.missing_listings = 0
            return
        if PROCESSING in listing:
            self.missing_listings = 0
            return
        if REQUEST in listing:
            self.missing_listings = 0
            age = self.clock() - local_processing.stat().st_mtime
            if age > self.probe_timeout:
                self.remove_remote(REQUEST)
                self.fail_in_flight(local_processing, request_id, f"not picked up within {age:.0f}s")
            return
        self.missing_listings += 1
        if self.missing_listings >= LOST_LISTINGS:
            self.fail_in_flight(local_processing, request_id, "discarded by the remote trainer (restart?)")

    def fail_in_flight(self, local_processing: Path, request_id: str | None, reason: str) -> None:
        refusal = {"id": request_id if request_id is not None else "", "ok": False, "answers": {}}
        write_atomic(self.mirror / RESPONSE, json.dumps(refusal, separators=(",", ":")).encode())
        local_processing.unlink(missing_ok=True)
        fsync_directory(self.mirror)
        self.missing_listings = 0
        log(f"probe {request_id!r} failed: {reason}; refusal response written")

    @staticmethod
    def request_id(path: Path) -> str | None:
        request = read_json(path)
        identifier = request.get("id") if request else None
        return identifier if isinstance(identifier, str) else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mirror a remote River Song telemetry dir and forward probes.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    # Replacement B200 (Oct 5 2026): ubuntu@4.71.129.4 port 32317, password-only until a key is installed.
    # The earlier instance (port 30204) was destroyed; never use it. Override with flags or RIVER_B200_* env.
    parser.add_argument("--host", default=os.environ.get("RIVER_B200_HOST", "4.71.129.4"))
    parser.add_argument("--user", default=os.environ.get("RIVER_B200_USER", "ubuntu"),
                        help="remote login (default: ubuntu)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("RIVER_B200_PORT", "32317")))
    parser.add_argument("--identity", type=Path, help="private key file (ssh -i); unattended reconnects")
    parser.add_argument("--password-file", type=Path,
                        help="file holding the ssh password, used with sshpass for unattended reconnects on a "
                             "password-only host (mode 0600; prefer ssh-copy-id + --identity)")
    parser.add_argument("--control-persist", type=int, default=600,
                        help="seconds an idle ssh master stays open after the bridge exits (lets a master opened "
                             "by hand or by a password prompt be reused across bridge restarts)")
    parser.add_argument("--remote-telemetry", required=True, help="trainer --telemetry-dir on the remote box")
    parser.add_argument("--local-mirror", type=Path, required=True, help="local dir the portal reads instead")
    parser.add_argument("--interval", type=float, default=3.0, help="seconds between mirror cycles")
    parser.add_argument("--probe-timeout", type=float, default=DEFAULT_PROBE_TIMEOUT,
                        help="seconds before an un-picked-up remote request.json is withdrawn (portal uses 600)")
    parser.add_argument("--control-path", default=f"/tmp/river-bridge-{os.getuid()}-%C",
                        help="ssh ControlPath (ssh expands %%C to a host/port/user hash)")
    parser.add_argument("--once", action="store_true", help="one mirror cycle, then exit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    transport = SshTransport(args.host, args.user, args.port, args.identity, args.remote_telemetry,
                             args.control_path, password_file=args.password_file,
                             control_persist=args.control_persist)
    bridge = Bridge(transport, args.local_mirror, interval=args.interval, probe_timeout=args.probe_timeout)
    signal.signal(signal.SIGINT, bridge.stop)
    signal.signal(signal.SIGTERM, bridge.stop)
    log(f"mirroring {transport.destination}:{transport.remote_dir} -> {args.local_mirror} every {args.interval:g}s")
    status = bridge.run(once=args.once)
    log("stopped" if bridge.stopped else f"exit {status}")
    return status


if __name__ == "__main__":
    sys.exit(main())
