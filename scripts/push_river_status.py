#!/usr/bin/env python3
"""Replace the single latest Friday progress message on the River Song portal."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlsplit, urlunsplit


def publish(message: str, url: str, token: str) -> str:
    if not message.strip() or len(message.encode("utf-8")) > 16_384:
        raise SystemExit("message must be non-empty UTF-8 text up to 16384 bytes")
    request = Request(
        url,
        data=json.dumps({"message": message}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-River-Token": token},
        method="POST",
    )
    try:
        with urlopen(request, timeout=15) as response:
            status = json.load(response)
    except HTTPError as error:
        detail = error.read(65_536).decode("utf-8", errors="replace")
        raise SystemExit(f"Status publication failed (HTTP {error.code}): {detail}") from error
    except URLError as error:
        raise SystemExit(f"Status publication failed: {error.reason}") from error
    return status["updated_at"]


def training_message(url: str, action: str, next_action: str) -> str:
    endpoint = urlsplit(url)
    state_url = urlunsplit((endpoint.scheme, endpoint.netloc, "/api/state", "", ""))
    with urlopen(Request(state_url, headers={"Cache-Control": "no-cache"}), timeout=15) as response:
        state = json.load(response)
    def unit_properties(unit: str) -> dict[str, str]:
        shown = subprocess.run(
            ["systemctl", "--user", "show", unit, "--property=ActiveState", "--property=SubState", "--property=MainPID"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        # systemctl prints properties in its own order, not request order.
        return dict(line.split("=", 1) for line in shown.stdout.splitlines())

    unit = "river-song-training.service"
    properties = unit_properties(unit)
    if properties["MainPID"] == "0":
        # The main service is stopped; report the GPU trial the dashboard mirrors instead, if one runs.
        trials = subprocess.run(
            ["systemctl", "--user", "list-units", "river-trial-*", "--state=active", "--no-legend", "--plain"],
            capture_output=True, text=True, timeout=15,
        ).stdout.split()
        trial_units = [word for word in trials if word.startswith("river-trial-") and word.endswith(".service")]
        if trial_units:
            unit = trial_units[0]
            properties = unit_properties(unit)
    pid = properties["MainPID"]
    service_state = f"{unit.removesuffix('.service')} {properties['ActiveState']}/{properties['SubState']}"
    # Observed executable identity, so the static --action text can never misstate the deployment.
    try:
        import hashlib
        with open(f"/proc/{pid}/exe", "rb") as executable:
            identity = f"PID {pid}, executable SHA-256 {hashlib.file_digest(executable, 'sha256').hexdigest()[:16]}"
    except OSError:
        identity = f"PID {pid}, executable unreadable"
    checkpoint_path = Path(state["checkpoint"]) / "experts.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    stamp = state.get("unix_millis")
    age = max(0, int(time.time() - stamp / 1000)) if isinstance(stamp, (int, float)) else None
    rate = state.get("samples_per_second")
    throughput = f"{rate:.2f} examples/second" if isinstance(rate, (int, float)) else "unavailable"
    promotion = state.get("promotion") or {}
    return (
        f"Live training service: {service_state} ({identity}). Latest telemetry: {state.get('status', 'unavailable')} "
        f"at batch {state.get('batch', 'unavailable')} ({age}s old).\n"
        f"Latest exact saved checkpoint: epoch {checkpoint['epoch']}, batch {checkpoint['cumulative_batches']}. "
        f"Observed rate: {throughput}.\n\n"
        f"Current action: {action}\n"
        f"Verified results: live service and committed expert manifest were read for this update. "
        f"Last held-out evaluation batch: {promotion.get('batch', 'unavailable')}; "
        f"promotion ready: {promotion.get('promotion_ready', 'unavailable')}. "
        "These are observations, not a claim of general intelligence or improved quality.\n"
        f"Next: {next_action}\n"
        "Automatic latest-only update; no trainer pause, GPU probe, model mutation, or message history."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    message_source = parser.add_mutually_exclusive_group(required=True)
    message_source.add_argument("--message")
    message_source.add_argument("--stdin", action="store_true")
    message_source.add_argument("--watch-training", action="store_true")
    parser.add_argument("--url", default="http://127.0.0.1:8799/api/friday-update")
    parser.add_argument("--token-file", type=Path, default=Path.home() / ".config/river-song/token")
    parser.add_argument("--interval", type=int, default=240)
    parser.add_argument("--action")
    parser.add_argument("--next", dest="next_action")
    args = parser.parse_args()
    token = args.token_file.read_text().strip()
    if args.watch_training:
        if not args.action or not args.next_action or not 30 <= args.interval <= 240:
            parser.error("--watch-training requires --action, --next, and an interval from 30 to 240 seconds")
        endpoint = urlsplit(args.url)
        if endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}:
            parser.error("--watch-training observes the local service and requires a loopback portal URL")
        while True:
            message = training_message(args.url, args.action, args.next_action)
            print(f"Published Friday update at {publish(message, args.url, token)}", flush=True)
            time.sleep(args.interval)
    message = args.message
    if args.stdin:
        try:
            message = sys.stdin.buffer.read(16_385).decode("utf-8")
        except UnicodeDecodeError:
            parser.error("message must be UTF-8 text")
    print(f"Published Friday update at {publish(message, args.url, token)}")


if __name__ == "__main__":
    main()
