import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from portal.server import PortalServer


class LatestStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.token = "isolated-test-status-token-not-a-real-credential"
        self.server = PortalServer(("127.0.0.1", 0), self.root, self.token)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/api/friday-update"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.directory.cleanup()

    def publish(self, message: str, token: str | None = None) -> dict:
        request = Request(
            self.url,
            data=json.dumps({"message": message}).encode(),
            headers={"Content-Type": "application/json", "X-River-Token": self.token if token is None else token},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            return json.load(response)

    def test_replaces_old_message_and_survives_server_restart(self) -> None:
        self.publish("Checkpoint committed; preparing the benchmark.")
        self.publish("Benchmark rejected; training resumed.")
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.server = PortalServer(("127.0.0.1", 0), self.root, self.token)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/api/friday-update"
        with urlopen(self.url, timeout=5) as response:
            self.assertEqual(json.load(response)["message"], "Benchmark rejected; training resumed.")
        with sqlite3.connect(self.root / "friday-status.sqlite3") as database:
            self.assertEqual(database.execute("SELECT id, message FROM latest_status").fetchall(), [(1, "Benchmark rejected; training resumed.")])

    def test_unauthorized_push_cannot_replace_latest_message(self) -> None:
        self.publish("Verified live training continues.")
        with self.assertRaises(HTTPError) as error:
            self.publish("False status", token="incorrect-token")
        self.assertEqual(error.exception.code, 401)
        with urlopen(self.url, timeout=5) as response:
            self.assertEqual(json.load(response)["message"], "Verified live training continues.")

    def test_public_samples_do_not_grant_status_write_access(self) -> None:
        sample = {"batch": 42, "expected": 0.7, "actual": 0.6}
        (self.root / "samples.jsonl").write_text(json.dumps(sample) + "\n")
        base_url = self.url.removesuffix("/api/friday-update")
        with urlopen(base_url + "/api/samples", timeout=5) as response:
            self.assertEqual(json.load(response)["samples"], [sample])
        request = Request(
            self.url,
            data=json.dumps({"message": "Unauthenticated replacement"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        self.publish("Training is active.")
        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=5)
        self.assertEqual(error.exception.code, 401)
        with urlopen(self.url, timeout=5) as response:
            self.assertEqual(json.load(response)["message"], "Training is active.")


if __name__ == "__main__":
    unittest.main()
