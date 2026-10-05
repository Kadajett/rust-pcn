"""Unit tests for river_remote_bridge.py against a fake transport backed by a local 'remote' directory (no network)."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import river_remote_bridge as bridge  # noqa: E402

STATE_A = b'{"status":"training","batch":1}'
STATE_B = b'{"status":"training","batch":2,"longer":true}'
PROBE = {"id": "probe-1", "outputs": {"answer": {"type": "text", "max_bytes": 32}}}
ANSWER = {"id": "probe-1", "ok": True, "answers": {"answer": {"type": "text", "text": "hi", "output_scope": "inherited"}}}


class LocalTransport(bridge.Transport):
    """Fakes the ssh/rsync operations by reading and writing a local directory; records deletes and failures."""

    def __init__(self, remote: Path) -> None:
        self.remote = remote
        self.removed: list[str] = []
        self.fail_calls = 0
        self.connects = 0
        self.closes = 0
        self.during_fetch_whole = None  # optional callback run while a whole-file fetch is in progress

    def failing(self) -> None:
        if self.fail_calls:
            self.fail_calls -= 1
            raise bridge.TransportError("simulated drop")

    def connect(self) -> None:
        self.connects += 1
        self.failing()

    def close(self) -> None:
        self.closes += 1

    def listing(self) -> dict[str, bridge.RemoteFile]:
        self.failing()
        files = {}
        for path in self.remote.iterdir():
            if path.is_file():
                stat = path.stat()
                files[path.name] = bridge.RemoteFile(stat.st_size, f"{stat.st_mtime_ns}")
        return files

    def fetch_whole(self, names: list[str], staging: Path) -> None:
        self.failing()
        for name in names:
            if self.during_fetch_whole:
                self.during_fetch_whole(name)
            shutil.copyfile(self.remote / name, staging / name)

    def fetch_range(self, name: str, offset: int, length: int) -> bytes:
        self.failing()
        with (self.remote / name).open("rb") as stream:
            stream.seek(offset)
            return stream.read(length)

    def put_file(self, name: str, data: bytes) -> None:
        self.failing()
        temporary = self.remote / f".{name}.tmp"
        temporary.write_bytes(data)
        temporary.replace(self.remote / name)

    def remove(self, name: str) -> None:
        self.failing()
        self.removed.append(name)
        (self.remote / name).unlink(missing_ok=True)


class BridgeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="river-bridge-test-"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.remote = self.root / "remote"
        self.mirror = self.root / "mirror"
        self.remote.mkdir()
        self.transport = LocalTransport(self.remote)
        self.sleeps: list[float] = []
        self.now = 1_000_000.0
        self.bridge = bridge.Bridge(self.transport, self.mirror, interval=3.0, probe_timeout=600.0,
                                    sleep=self.sleeps.append, clock=lambda: self.now)
        bridge.log = lambda message: None

    def cycle(self) -> None:
        self.bridge.run_once()


class WholeFileMirrorTest(BridgeTestCase):
    def test_state_json_is_staged_then_replaced_atomically(self):
        (self.remote / "state.json").write_bytes(STATE_A)
        self.cycle()
        self.assertEqual((self.mirror / "state.json").read_bytes(), STATE_A)

        observed: list[bytes] = []

        def during_fetch(name: str) -> None:
            # While the new bytes are being staged, the published file is still the complete old one.
            observed.append((self.mirror / name).read_bytes())
            self.assertFalse((self.mirror / name).read_bytes() == STATE_B)

        self.transport.during_fetch_whole = during_fetch
        (self.remote / "state.json").write_bytes(STATE_B)
        self.cycle()
        self.assertEqual(observed, [STATE_A])
        self.assertEqual((self.mirror / "state.json").read_bytes(), STATE_B)
        self.assertEqual(list((self.mirror / bridge.STAGING_DIR).iterdir()), [])

    def test_unchanged_files_are_not_refetched_and_handoff_files_are_skipped(self):
        (self.remote / "manifest.json").write_bytes(b"{}")
        (self.remote / "request.processing.json").write_bytes(b"{}")
        (self.remote / "friday-status.sqlite3").write_bytes(b"x")
        (self.remote / ".state.json.tmp").write_bytes(b"partial")
        self.cycle()
        self.assertEqual(sorted(path.name for path in self.mirror.iterdir() if path.is_file()), ["manifest.json"])
        fetched: list[str] = []
        self.transport.during_fetch_whole = fetched.append
        self.cycle()
        self.assertEqual(fetched, [])


class AppendMirrorTest(BridgeTestCase):
    def test_jsonl_growth_across_cycles_matches_remote(self):
        events = self.remote / "events.jsonl"
        events.write_bytes(b'{"n":1}\n')
        self.cycle()
        with events.open("ab") as stream:
            stream.write(b'{"n":2}\n' * 400)
        self.cycle()
        self.assertEqual((self.mirror / "events.jsonl").read_bytes(), events.read_bytes())
        self.cycle()
        self.assertEqual((self.mirror / "events.jsonl").read_bytes(), events.read_bytes())

    def test_large_growth_is_chunked_and_catches_up(self):
        self.bridge.max_chunk = 1000
        samples = self.remote / "samples.jsonl"
        samples.write_bytes(b"x" * 2500 + b"\n")
        self.cycle()
        self.assertEqual((self.mirror / "samples.jsonl").stat().st_size, 1000)
        self.cycle()
        self.cycle()
        self.assertEqual((self.mirror / "samples.jsonl").read_bytes(), samples.read_bytes())

    def test_truncated_remote_resets_mirror(self):
        events = self.remote / "events.jsonl"
        events.write_bytes(b"old line\n" * 100)
        self.cycle()
        events.write_bytes(b"fresh\n")  # trainer restart truncates events.jsonl
        self.cycle()
        self.assertEqual((self.mirror / "events.jsonl").read_bytes(), b"fresh\n")

    def test_regrown_remote_with_different_content_resets_mirror(self):
        events = self.remote / "events.jsonl"
        events.write_bytes(b"a" * 500 + b"\n")
        self.cycle()
        events.write_bytes(b"b" * 900 + b"\n")  # truncated and regrown past the mirrored size
        self.cycle()
        self.assertEqual((self.mirror / "events.jsonl").read_bytes(), events.read_bytes())


class ProbeHandoffTest(BridgeTestCase):
    def write_local_request(self) -> bytes:
        self.mirror.mkdir(exist_ok=True)
        data = json.dumps(PROBE, separators=(",", ":")).encode()
        (self.mirror / "request.json").write_bytes(data)
        return data

    def test_request_is_uploaded_and_marked_processing_locally(self):
        data = self.write_local_request()
        self.cycle()
        self.assertEqual((self.remote / "request.json").read_bytes(), data)
        self.assertFalse((self.mirror / "request.json").exists())
        self.assertEqual((self.mirror / "request.processing.json").read_bytes(), data)
        self.assertEqual(self.transport.removed, [])

    def test_response_round_trip(self):
        self.write_local_request()
        self.cycle()
        (self.remote / "request.json").rename(self.remote / "request.processing.json")  # trainer picks it up
        self.cycle()
        self.assertTrue((self.mirror / "request.processing.json").exists())
        self.assertFalse((self.mirror / "response.json").exists())
        answer = json.dumps(ANSWER).encode()
        (self.remote / "response.json").write_bytes(answer)  # trainer answers ...
        (self.remote / "request.processing.json").unlink()  # ... and clears its marker
        self.cycle()
        self.assertEqual((self.mirror / "response.json").read_bytes(), answer)
        self.assertFalse((self.mirror / "request.processing.json").exists())
        self.assertFalse((self.remote / "response.json").exists())
        self.assertEqual(self.transport.removed, ["response.json"])
        self.cycle()  # the lingering local response.json is the portal's to clean up; nothing else happens
        self.assertEqual(self.transport.removed, ["response.json"])
        self.assertTrue((self.mirror / "response.json").exists())

    def test_stale_remote_response_is_discarded(self):
        (self.remote / "response.json").write_bytes(json.dumps(ANSWER).encode())
        self.cycle()
        self.assertFalse((self.mirror / "response.json").exists())
        self.assertFalse((self.remote / "response.json").exists())
        self.assertEqual(self.transport.removed, ["response.json"])

    def test_response_with_other_id_is_discarded_while_in_flight(self):
        self.write_local_request()
        self.cycle()
        (self.remote / "request.json").rename(self.remote / "request.processing.json")
        (self.remote / "response.json").write_bytes(json.dumps({**ANSWER, "id": "older"}).encode())
        self.cycle()
        self.assertTrue((self.mirror / "request.processing.json").exists())
        self.assertFalse((self.mirror / "response.json").exists())
        self.assertEqual(self.transport.removed, ["response.json"])

    def test_stale_remote_response_is_cleared_before_upload(self):
        (self.remote / "response.json").write_bytes(b'{"id":"older","ok":false,"answers":{}}')
        self.write_local_request()
        self.cycle()
        self.assertEqual(self.transport.removed, ["response.json"])
        self.assertTrue((self.remote / "request.json").exists())
        self.assertFalse((self.mirror / "response.json").exists())

    def test_unpicked_request_times_out_with_refusal(self):
        self.write_local_request()
        self.cycle()
        self.now += 599
        self.cycle()
        self.assertTrue((self.remote / "request.json").exists())
        self.now += 2
        self.cycle()
        self.assertFalse((self.remote / "request.json").exists())
        self.assertFalse((self.mirror / "request.processing.json").exists())
        self.assertEqual(json.loads((self.mirror / "response.json").read_bytes()),
                         {"id": "probe-1", "ok": False, "answers": {}})
        self.assertEqual(self.transport.removed, ["request.json"])

    def test_request_dropped_by_trainer_restart_fails_after_two_listings(self):
        self.write_local_request()
        self.cycle()
        (self.remote / "request.json").unlink()  # trainer restart discarded it without answering
        self.cycle()
        self.assertTrue((self.mirror / "request.processing.json").exists())
        self.cycle()
        self.assertFalse((self.mirror / "request.processing.json").exists())
        self.assertEqual(json.loads((self.mirror / "response.json").read_bytes()),
                         {"id": "probe-1", "ok": False, "answers": {}})
        self.assertEqual(self.transport.removed, [])

    def test_in_flight_state_survives_bridge_restart(self):
        self.write_local_request()
        self.cycle()
        (self.remote / "request.json").rename(self.remote / "request.processing.json")
        restarted = bridge.Bridge(self.transport, self.mirror, sleep=self.sleeps.append, clock=lambda: self.now)
        (self.remote / "response.json").write_bytes(json.dumps(ANSWER).encode())
        (self.remote / "request.processing.json").unlink()
        restarted.run_once()
        self.assertTrue((self.mirror / "response.json").exists())
        self.assertFalse((self.mirror / "request.processing.json").exists())

    def test_remote_delete_guard_rejects_non_handoff_names(self):
        with self.assertRaises(RuntimeError):
            self.bridge.remove_remote("state.json")
        self.assertEqual(self.transport.removed, [])


class LoopTest(BridgeTestCase):
    def test_backoff_grows_caps_and_resets_after_success(self):
        (self.remote / "state.json").write_bytes(STATE_A)
        self.transport.fail_calls = 7
        cycles = {"n": 0}
        original = self.bridge.run_once

        def counted() -> None:
            original()
            cycles["n"] += 1
            if cycles["n"] == 2:
                self.transport.fail_calls = 1
            if cycles["n"] == 3:
                self.bridge.stop()

        self.bridge.run_once = counted
        status = self.bridge.run()
        self.assertEqual(status, 0)
        self.assertEqual(self.sleeps, [1, 2, 4, 8, 16, 30, 30, 3.0, 3.0, 1, 3.0])
        self.assertEqual((self.mirror / "state.json").read_bytes(), STATE_A)
        self.assertGreaterEqual(self.transport.closes, 8)

    def test_once_runs_a_single_cycle(self):
        (self.remote / "state.json").write_bytes(STATE_A)
        self.assertEqual(self.bridge.run(once=True), 0)
        self.assertEqual(self.sleeps, [])
        self.assertEqual((self.mirror / "state.json").read_bytes(), STATE_A)
        self.assertEqual(self.transport.connects, 1)

    def test_once_reports_transport_failure(self):
        self.transport.fail_calls = 1
        self.assertEqual(self.bridge.run(once=True), 1)
        self.assertEqual(self.sleeps, [])


class ParserTest(unittest.TestCase):
    def test_required_flags_and_defaults(self):
        args = bridge.build_parser().parse_args(
            ["--host", "h", "--user", "u", "--remote-telemetry", "/r", "--local-mirror", "/m", "--once"])
        self.assertEqual((args.interval, args.probe_timeout, args.once), (3.0, 600.0, True))
        self.assertIn(str(os.getuid()), args.control_path)

    def test_listing_parser_ignores_subdirectory_entries_and_garbage(self):
        parsed = bridge.parse_listing("12\t1.5\tstate.json\nbad line\n3\t2.0\tsub/x.json\n")
        self.assertEqual(parsed, {"state.json": bridge.RemoteFile(12, "1.5")})


if __name__ == "__main__":
    unittest.main()
