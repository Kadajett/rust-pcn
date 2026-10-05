#!/usr/bin/env python3
"""Serve live River training telemetry and a bounded in-process model probe queue."""

from __future__ import annotations

import argparse
import hmac
import sqlite3
import json
import math
import os
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import closing
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
TEST_LOCK = threading.Lock()
CAPABILITY_REPORT_MAX_BYTES = 1_048_576
PROBE_TIMEOUT_SECONDS = 600.0  # covers stage-boundary corpus loading, when probes are not served
# Requests arriving through the public Cloudflare tunnel pause training while the trainer answers them, so they get
# a small budget: few outputs, short answers, one request per interval. Local helpers are unaffected.
PUBLIC_MAX_OUTPUTS = 2
PUBLIC_MAX_BYTES = 256
PUBLIC_MIN_INTERVAL_SECONDS = 120.0
PUBLIC_LOCK = threading.Lock()
public_last_request = 0.0

def reject_non_finite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value} is not allowed")


class PortalServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], telemetry: Path, token: str):
        super().__init__(address, PortalHandler)
        self.telemetry = telemetry
        self.token = token
        self.status_path = telemetry / "friday-status.sqlite3"
        with closing(sqlite3.connect(self.status_path)) as database:
            database.execute(
                "CREATE TABLE IF NOT EXISTS latest_status ("
                "id INTEGER PRIMARY KEY CHECK (id = 1), "
                "message TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            database.commit()

    def latest_status(self) -> dict[str, str] | None:
        with closing(sqlite3.connect(self.status_path)) as database:
            row = database.execute(
                "SELECT message, updated_at FROM latest_status WHERE id = 1"
            ).fetchone()
        return None if row is None else {"message": row[0], "updated_at": row[1]}

    def replace_status(self, message: str) -> dict[str, str]:
        updated_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        with closing(sqlite3.connect(self.status_path)) as database:
            database.execute(
                "INSERT INTO latest_status (id, message, updated_at) VALUES (1, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "message = excluded.message, updated_at = excluded.updated_at",
                (message, updated_at),
            )
            database.commit()
        return {"message": message, "updated_at": updated_at}


class PortalHandler(BaseHTTPRequestHandler):
    server: PortalServer
    server_version = "RiverSongPortal/0.1"

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"river-portal {self.client_address[0]} {format_string % args}", flush=True)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        route = urlparse(self.path).path
        if route == "/api/capabilities":
            self.send_capabilities()
            return
        if route == "/api/friday-update":
            status = self.server.latest_status()
            if status is None:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "No Friday update has been published."})
            else:
                self.send_json(HTTPStatus.OK, status)
            return
        if route == "/api/state":
            self.send_state()
            return
        if route == "/api/events":
            self.send_events()
            return
        if route == "/api/audits":
            self.send_audits()
            return
        if route == "/api/samples":
            self.send_samples()
            return
        if route == "/api/stream":
            self.stream_state()
            return
        if route == "/healthz":
            self.send_json(HTTPStatus.OK, {"ok": True, "unix_millis": int(time.time() * 1000)})
            return
        static = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
            "/docs": ("docs/index.html", "text/html; charset=utf-8"),
            "/docs/": ("docs/index.html", "text/html; charset=utf-8"),
            "/docs/index.html": ("docs/index.html", "text/html; charset=utf-8"),
            "/docs/model": ("docs/model.html", "text/html; charset=utf-8"),
            "/docs/model.html": ("docs/model.html", "text/html; charset=utf-8"),
            "/docs/training": ("docs/training.html", "text/html; charset=utf-8"),
            "/docs/training.html": ("docs/training.html", "text/html; charset=utf-8"),
            "/docs/data": ("docs/data.html", "text/html; charset=utf-8"),
            "/docs/data.html": ("docs/data.html", "text/html; charset=utf-8"),
            "/docs/outputs": ("docs/outputs.html", "text/html; charset=utf-8"),
            "/docs/outputs.html": ("docs/outputs.html", "text/html; charset=utf-8"),
            "/docs/paper": ("docs/paper.html", "text/html; charset=utf-8"),
            "/docs/paper.html": ("docs/paper.html", "text/html; charset=utf-8"),
            "/docs/research": ("docs/research.html", "text/html; charset=utf-8"),
            "/docs/research.html": ("docs/research.html", "text/html; charset=utf-8"),
            "/docs/docs.css": ("docs/docs.css", "text/css; charset=utf-8"),
        }
        item = static.get(route)
        if item is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        name, content_type = item
        path = ROOT / name
        try:
            body = path.read_bytes()
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_capabilities(self) -> None:
        path = self.server.telemetry / "capability-evidence.json"
        try:
            with path.open("rb") as stream:
                encoded = stream.read(CAPABILITY_REPORT_MAX_BYTES + 1)
            if len(encoded) > CAPABILITY_REPORT_MAX_BYTES:
                raise ValueError("report exceeds the 1 MiB limit")
            report = json.loads(encoded.decode("utf-8"), parse_constant=reject_non_finite_json)
            if (
                not isinstance(report, dict)
                or report.get("schema") != "river-capability-evidence-v1"
                or not isinstance(report.get("cases"), list)
                or not isinstance(report.get("aggregate_by_family"), dict)
            ):
                raise ValueError("report does not match the capability evidence schema")
        except FileNotFoundError:
            self.send_json(
                HTTPStatus.NOT_FOUND,
                {"error": "No capability evidence report has been published."},
            )
            return
        except (ValueError, RecursionError):
            self.send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "Published capability evidence report is corrupt or exceeds the 1 MiB limit."},
            )
            return
        except OSError:
            self.send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "Published capability evidence report is unavailable."},
            )
            return
        self.send_json(HTTPStatus.OK, report)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        route = urlparse(self.path).path
        if route not in {"/api/test", "/api/friday-update"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if route == "/api/friday-update":
            supplied = self.headers.get("X-River-Token", "")
            if not self.server.token or not hmac.compare_digest(supplied, self.server.token):
                self.send_json(HTTPStatus.UNAUTHORIZED, {"error": "valid X-River-Token required"})
                return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65_536:
                raise ValueError("request must be between 1 and 65536 bytes")
            request = json.loads(
                self.rfile.read(length),
                parse_constant=reject_non_finite_json,
            )
            if route == "/api/friday-update":
                if not isinstance(request, dict) or set(request) != {"message"}:
                    raise ValueError("status request must contain exactly message")
                message = request["message"]
                if (
                    not isinstance(message, str)
                    or not message.strip()
                    or len(message.encode("utf-8")) > 16_384
                ):
                    raise ValueError("message must be non-empty UTF-8 text up to 16384 bytes")
                self.send_json(HTTPStatus.OK, self.server.replace_status(message))
                return
            normalized, legacy_client = self.validate_test_request(request)
            if self.headers.get("Cf-Connecting-Ip") or self.headers.get("Cf-Ray"):
                self.check_public_budget(normalized)
        except PermissionError as error:
            self.send_json(HTTPStatus.TOO_MANY_REQUESTS, {"error": str(error)})
            return
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            return
        if not TEST_LOCK.acquire(blocking=False):
            self.send_json(HTTPStatus.CONFLICT, {"error": "another model probe is active"})
            return
        try:
            self.enqueue_test(normalized, legacy_client)
        finally:
            TEST_LOCK.release()

    @staticmethod
    def check_public_budget(request: dict[str, object]) -> None:
        global public_last_request
        outputs = request["outputs"]
        if len(outputs) > PUBLIC_MAX_OUTPUTS or any(
            spec.get("max_bytes", 0) > PUBLIC_MAX_BYTES for spec in outputs.values()
        ):
            raise ValueError(
                f"public requests allow at most {PUBLIC_MAX_OUTPUTS} outputs of {PUBLIC_MAX_BYTES} bytes each"
            )
        with PUBLIC_LOCK:
            wait = public_last_request + PUBLIC_MIN_INTERVAL_SECONDS - time.monotonic()
            if wait > 0:
                raise PermissionError(f"public probes are limited to one per {PUBLIC_MIN_INTERVAL_SECONDS:.0f} s; retry in {wait:.0f} s")
            public_last_request = time.monotonic()

    def validate_test_request(self, request: object) -> tuple[dict[str, object], bool]:
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
        if "inputs" in request or "outputs" in request:
            if set(request) != {"id", "inputs", "outputs"}:
                raise ValueError("runtime v1 request must contain exactly id, inputs, and outputs")
            request_id = request["id"]
            if (
                not isinstance(request_id, str)
                or not request_id.strip()
                or len(request_id.encode("utf-8")) > 128
            ):
                raise ValueError("id must be non-empty UTF-8 text no longer than 128 bytes")
            if request["inputs"] is None:
                raise ValueError("inputs must be a non-null JSON value")
            outputs = request["outputs"]
            if not isinstance(outputs, dict) or not 1 <= len(outputs) <= 64:
                raise ValueError("outputs must be an object containing 1 to 64 named outputs")
            normalized_outputs: dict[str, object] = {}
            for name, specification in outputs.items():
                if (
                    not isinstance(name, str)
                    or not name.strip()
                    or name != name.strip()
                    or len(name.encode("utf-8")) > 128
                    or any(ord(character) < 32 for character in name)
                ):
                    raise ValueError("output names must be trimmed UTF-8 text from 1 to 128 bytes")
                normalized_outputs[name] = self.validate_output_specification(name, specification)
            return {
                "id": request_id,
                "inputs": request["inputs"],
                "outputs": normalized_outputs,
            }, False

        allowed = {"id", "prompt", "mode", "modality", "max_bytes", "schema"}
        if not set(request).issubset(allowed):
            raise ValueError("legacy request contains unsupported fields")
        prompt = request.get("prompt", "")
        mode = request.get("mode", "json")
        modality = request.get("modality", "prose")
        max_bytes = request.get("max_bytes", 128)
        schema = request.get("schema")
        if not isinstance(prompt, str) or len(prompt.encode("utf-8")) > 8_192:
            raise ValueError("prompt must be UTF-8 text no longer than 8192 bytes")
        if mode not in {"json", "text"}:
            raise ValueError("mode must be json or text")
        if modality not in {"prose", "code"}:
            raise ValueError("modality must be prose or code")
        self.validate_max_bytes(max_bytes)
        request_id = request.get("id", uuid.uuid4().hex)
        if (
            not isinstance(request_id, str)
            or not request_id.strip()
            or len(request_id.encode("utf-8")) > 128
        ):
            raise ValueError("legacy id must be non-empty UTF-8 text no longer than 128 bytes")
        if mode == "json":
            self.validate_json_schema(schema)
        output: dict[str, object] = {
            "type": "structured" if mode == "json" else "text",
            "instructions": prompt or "Answer using the supplied input.",
            "max_bytes": max_bytes,
        }
        if mode == "json":
            output["schema"] = schema
        return {
            "id": request_id,
            "inputs": {"prompt": prompt, "modality": modality},
            "outputs": {"output": output},
        }, True

    def validate_output_specification(self, name: str, specification: object) -> dict[str, object]:
        if not isinstance(specification, dict):
            raise ValueError(f'output "{name}" must be an object')
        output_type = specification.get("type")
        required = {
            "noul": {"type", "instructions", "criteria"},
            "choice": {"type", "instructions", "criteria"},
            "score": {"type", "instructions", "criteria"},
            "text": {"type", "instructions", "max_bytes"},
            "structured": {"type", "instructions", "schema", "max_bytes"},
        }.get(output_type)
        if required is None:
            raise ValueError(
                f'output "{name}" type must be noul, choice, score, text, or structured'
            )
        if set(specification) != required:
            fields = ", ".join(sorted(required))
            raise ValueError(f'output "{name}" must contain exactly {fields}')
        instructions = specification["instructions"]
        if (
            not isinstance(instructions, str)
            or not instructions.strip()
            or len(instructions.encode("utf-8")) > 8_192
        ):
            raise ValueError(f'output "{name}" instructions must be 1 to 8192 UTF-8 bytes')
        if output_type == "noul":
            criteria = specification["criteria"]
            if not isinstance(criteria, dict) or set(criteria) != {"true", "false"}:
                raise ValueError(f'output "{name}" criteria must contain exactly true and false')
            for criterion, description in criteria.items():
                if (
                    not isinstance(description, str)
                    or not description.strip()
                    or len(description.encode("utf-8")) > 4_096
                ):
                    raise ValueError(
                        f'output "{name}" {criterion} criterion must be 1 to 4096 UTF-8 bytes'
                    )
            if criteria["true"] == criteria["false"]:
                raise ValueError(f'output "{name}" true and false criteria must differ')
        elif output_type == "choice":
            criteria = specification["criteria"]
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 64:
                raise ValueError(f'output "{name}" Choice criteria must contain 2 to 64 options')
            for option, description in criteria.items():
                if (
                    not isinstance(option, str)
                    or not option.strip()
                    or len(option.encode("utf-8")) > 128
                    or not isinstance(description, str)
                    or not description.strip()
                    or len(description.encode("utf-8")) > 4_096
                ):
                    raise ValueError(f'output "{name}" Choice options and descriptions are invalid')
        elif output_type == "score":
            criteria = specification["criteria"]
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 64:
                raise ValueError(f'output "{name}" Score criteria must contain 2 to 64 levels')
            if any(
                not isinstance(description, str)
                or not description.strip()
                or len(description.encode("utf-8")) > 4_096
                for description in criteria
            ):
                raise ValueError(f'output "{name}" Score descriptions are invalid')
        else:
            self.validate_max_bytes(specification["max_bytes"], name)
            if output_type == "structured":
                self.validate_json_schema(specification["schema"])
        return dict(specification)

    @staticmethod
    def validate_max_bytes(value: object, name: str | None = None) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65_536:
            prefix = f'output "{name}" ' if name is not None else ""
            raise ValueError(f"{prefix}max_bytes must be an integer from 1 to 65536")

    def validate_json_schema(
        self,
        schema: object,
        path: str = "schema",
        depth: int = 0,
        nodes: list[int] | None = None,
    ) -> None:
        if nodes is None:
            nodes = [0]
        nodes[0] += 1
        if depth > 16 or nodes[0] > 512:
            raise ValueError("schema exceeds the maximum depth or node count")
        if not isinstance(schema, dict):
            raise ValueError(f"{path} must be an object")
        schema_type = schema.get("type")
        fields = {
            "object": ({"type", "properties", "required"}, {"type", "properties"}),
            "array": ({"type", "items", "min_items", "max_items"}, {"type", "items", "max_items"}),
            "string": ({"type", "enum", "max_length"}, {"type"}),
            "integer": ({"type", "minimum", "maximum"}, {"type", "minimum", "maximum"}),
            "number": ({"type", "minimum", "maximum"}, {"type", "minimum", "maximum"}),
            "boolean": ({"type"}, {"type"}),
            "null": ({"type"}, {"type"}),
        }.get(schema_type)
        if fields is None:
            raise ValueError(f"{path}.type must be object, array, string, integer, number, boolean, or null")
        allowed, required = fields
        if not required.issubset(schema) or not set(schema).issubset(allowed):
            raise ValueError(f"{path} contains missing or unsupported fields for type {schema_type}")
        if schema_type == "object":
            properties = schema["properties"]
            required_properties = schema.get("required", [])
            if not isinstance(properties, dict) or len(properties) > 128:
                raise ValueError(f"{path}.properties must be an object with at most 128 entries")
            if (
                not isinstance(required_properties, list)
                or any(not isinstance(item, str) for item in required_properties)
                or len(set(required_properties)) != len(required_properties)
                or not set(required_properties).issubset(properties)
            ):
                raise ValueError(f"{path}.required must contain unique property names present in properties")
            for property_name, child in properties.items():
                if not isinstance(property_name, str) or not property_name:
                    raise ValueError(f"{path}.properties names must be non-empty text")
                self.validate_json_schema(child, f"{path}.properties.{property_name}", depth + 1, nodes)
        elif schema_type == "array":
            minimum = schema.get("min_items", 0)
            maximum = schema["max_items"]
            if (
                isinstance(minimum, bool)
                or isinstance(maximum, bool)
                or not isinstance(minimum, int)
                or not isinstance(maximum, int)
                or not 0 <= minimum <= maximum <= 1_024
            ):
                raise ValueError(f"{path} array bounds must satisfy 0 <= min_items <= max_items <= 1024")
            self.validate_json_schema(schema["items"], f"{path}.items", depth + 1, nodes)
        elif schema_type == "string":
            maximum = schema.get("max_length", 128)
            enum = schema.get("enum", [])
            if (
                isinstance(maximum, bool)
                or not isinstance(maximum, int)
                or not 1 <= maximum <= 16_384
                or not isinstance(enum, list)
                or any(
                    not isinstance(item, str) or len(item.encode("utf-8")) > maximum
                    for item in enum
                )
            ):
                raise ValueError(f"{path} string max_length/enum is invalid")
        elif schema_type in {"integer", "number"}:
            minimum = schema["minimum"]
            maximum = schema["maximum"]
            numeric = (int,) if schema_type == "integer" else (int, float)
            valid_numbers = (
                not isinstance(minimum, bool)
                and not isinstance(maximum, bool)
                and isinstance(minimum, numeric)
                and isinstance(maximum, numeric)
            )
            if schema_type == "integer" and valid_numbers and (
                minimum < -(2**63) or maximum > 2**63 - 1
            ):
                raise ValueError(f"{path} integer bounds must fit signed 64-bit values")
            try:
                finite_numbers = valid_numbers and math.isfinite(minimum) and math.isfinite(maximum)
            except OverflowError:
                finite_numbers = False
            if not finite_numbers or minimum > maximum:
                raise ValueError(f"{path} numeric bounds must be finite, correctly typed, and ordered")

    @staticmethod
    def json_value_matches_schema(value: object, schema: dict[str, object]) -> bool:
        schema_type = schema["type"]
        if schema_type == "object":
            if not isinstance(value, dict):
                return False
            properties = schema["properties"]
            required = schema.get("required", [])
            return (
                set(value).issubset(properties)
                and set(required).issubset(value)
                and all(
                    PortalHandler.json_value_matches_schema(child, properties[key])
                    for key, child in value.items()
                )
            )
        if schema_type == "array":
            return (
                isinstance(value, list)
                and schema.get("min_items", 0) <= len(value) <= schema["max_items"]
                and all(
                    PortalHandler.json_value_matches_schema(item, schema["items"])
                    for item in value
                )
            )
        if schema_type == "string":
            if not isinstance(value, str) or len(value.encode("utf-8")) > schema.get("max_length", 128):
                return False
            enum = schema.get("enum", [])
            return not enum or value in enum
        if schema_type == "integer":
            return (
                not isinstance(value, bool)
                and isinstance(value, int)
                and schema["minimum"] <= value <= schema["maximum"]
            )
        if schema_type == "number":
            return (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
                and schema["minimum"] <= value <= schema["maximum"]
            )
        if schema_type == "boolean":
            return isinstance(value, bool)
        return schema_type == "null" and value is None

    def enqueue_test(self, request: dict[str, object], legacy_client: bool) -> None:
        telemetry = self.server.telemetry
        state = self.read_state()
        runtime_contract = state.get("runtime_contract")
        runtime_v1 = runtime_contract == "river-runtime-request-v1"
        if runtime_contract is not None and not runtime_v1:
            self.send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": f"unsupported trainer runtime contract: {runtime_contract}"},
            )
            return
        if (
            runtime_contract is None
            and state.get("schema") == "river-universal-trainer-state-v1"
        ):
            self.send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "universal output path is not enabled yet; no probe value was generated"},
            )
            return
        if state.get("status") not in {"training", "training_tasks", "training_noul", "checkpointing"}:
            if runtime_v1:
                self.send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"error": "runtime v1 trainer is not active; offline universal probing is unavailable"},
                )
                return
            self.run_checkpoint_probe(request, legacy_client)
            return
        telemetry.mkdir(parents=True, exist_ok=True)
        request_path = telemetry / "request.json"
        if request_path.exists() or (telemetry / "request.processing.json").exists():
            self.send_json(HTTPStatus.CONFLICT, {"error": "the trainer is processing another probe"})
            return
        try:
            wire_request = (
                request
                if runtime_v1
                else self.legacy_trainer_request(
                    request,
                    max_bytes_limit=512,
                    source="active v4 trainer",
                )
            )
        except ValueError as error:
            self.send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error)})
            return
        response_path = telemetry / "response.json"
        response_path.unlink(missing_ok=True)
        temporary = telemetry / "request.json.tmp"
        temporary.write_text(json.dumps(wire_request, separators=(",", ":")))
        temporary.replace(request_path)
        # 100-step settling makes each generated byte slower, and requests wait for a batch boundary.
        deadline = time.monotonic() + PROBE_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            try:
                response = json.loads(response_path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                response = None
            if isinstance(response, dict) and response.get("id") == request["id"]:
                try:
                    normalized = (
                        self.validate_runtime_response(request, response)
                        if runtime_v1
                        else self.normalize_legacy_response(request, response)
                    )
                except ValueError as error:
                    self.send_json(HTTPStatus.BAD_GATEWAY, {"error": str(error)})
                    return
                if legacy_client:
                    source = "runtime_v1_compatibility" if runtime_v1 else "v4_trainer"
                    normalized = self.legacy_client_response(normalized, source)
                status = HTTPStatus.OK if normalized.get("ok") else HTTPStatus.UNPROCESSABLE_ENTITY
                self.send_json(status, normalized)
                return
            time.sleep(0.25)
        if request_path.exists():
            request_path.unlink(missing_ok=True)
        self.send_json(HTTPStatus.GATEWAY_TIMEOUT, {"error": f"trainer did not answer within {PROBE_TIMEOUT_SECONDS:.0f} seconds"})

    def legacy_trainer_request(
        self,
        request: dict[str, object],
        *,
        max_bytes_limit: int,
        source: str,
    ) -> dict[str, object]:
        outputs = request["outputs"]
        if not isinstance(outputs, dict) or len(outputs) != 1:
            raise ValueError(f"{source} supports exactly one inherited output")
        _, specification = next(iter(outputs.items()))
        if (
            not isinstance(specification, dict)
            or specification.get("type") in {"noul", "choice", "score"}
        ):
            raise ValueError(f"{source} cannot produce request-conditioned typed judgments")
        inputs = request["inputs"]
        if isinstance(inputs, dict) and isinstance(inputs.get("prompt"), str):
            prompt = inputs["prompt"]
            modality = inputs.get("modality", "prose")
        else:
            prompt = json.dumps(inputs, ensure_ascii=False, separators=(",", ":"))
            modality = "prose"
        if modality not in {"prose", "code"}:
            modality = "prose"
        if specification["max_bytes"] > max_bytes_limit:
            raise ValueError(f"{source} supports max_bytes at most {max_bytes_limit}")
        instructions = specification["instructions"]
        if instructions != prompt:
            prompt = f"{prompt}\n\nOutput request: {instructions}"
        output_type = specification["type"]
        return {
            "id": request["id"],
            "prompt": prompt,
            "mode": "json" if output_type == "structured" else "text",
            "modality": modality,
            "max_bytes": specification["max_bytes"],
            "schema": specification.get("schema") if output_type == "structured" else None,
        }

    def normalize_legacy_response(
        self, request: dict[str, object], response: dict[str, object]
    ) -> dict[str, object]:
        if response.get("ok") is not True:
            return {"id": request["id"], "ok": False, "answers": {}}
        outputs = request["outputs"]
        if not isinstance(outputs, dict) or len(outputs) != 1 or "output" not in response:
            raise ValueError("legacy v4 trainer returned a malformed response")
        name, specification = next(iter(outputs.items()))
        if not isinstance(specification, dict):
            raise ValueError("legacy v4 output specification was lost")
        if specification["type"] == "text" and not isinstance(response["output"], str):
            raise ValueError("legacy v4 trainer returned a non-text value for a text request")
        if specification["type"] == "text":
            answer = {
                "type": "text",
                "text": response["output"],
                "output_scope": "inherited",
            }
        else:
            answer = {
                "type": "structured",
                "value": response["output"],
                "output_scope": "inherited",
            }
        return {"id": request["id"], "ok": True, "answers": {name: answer}}

    def validate_runtime_response(
        self, request: dict[str, object], response: dict[str, object]
    ) -> dict[str, object]:
        if set(response) != {"id", "ok", "answers"} or not isinstance(response.get("ok"), bool):
            raise ValueError("runtime v1 trainer returned a malformed response envelope")
        answers = response.get("answers")
        outputs = request["outputs"]
        if not isinstance(answers, dict) or not isinstance(outputs, dict):
            raise ValueError("runtime v1 trainer answers must be a name-keyed object")
        if response["ok"] is False:
            if answers:
                raise ValueError("failed runtime v1 response must not claim typed answers")
            return response
        if set(answers) != set(outputs):
            raise ValueError("runtime v1 trainer did not return exactly the requested named answers")
        for name, answer in answers.items():
            specification = outputs[name]
            if not isinstance(answer, dict) or not isinstance(specification, dict):
                raise ValueError(f'runtime v1 answer "{name}" must be an object')
            output_type = specification["type"]
            expected_fields = {
                "noul": {"type", "noul", "output_scope"},
                "choice": {
                    "type", "choice", "probabilities", "confidence", "output_scope"
                },
                "score": {
                    "type", "score", "legend", "probabilities", "confidence", "output_scope"
                },
                "text": {"type", "text", "output_scope"},
                "structured": {"type", "value", "output_scope"},
            }[output_type]
            if set(answer) != expected_fields or answer.get("type") != output_type:
                raise ValueError(f'runtime v1 answer "{name}" does not preserve its requested type')
            expected_scope = (
                "request_conditioned"
                if output_type in {"noul", "choice", "score"}
                else "inherited"
            )
            if answer.get("output_scope") != expected_scope:
                raise ValueError(f'runtime v1 answer "{name}" has an invalid output scope')
            if output_type == "noul":
                value = answer.get("noul")
                valid_noul = not isinstance(value, bool) and isinstance(value, (int, float))
                try:
                    valid_noul = valid_noul and math.isfinite(value) and 0 < value < 1
                except OverflowError:
                    valid_noul = False
                if not valid_noul:
                    raise ValueError(f'runtime v1 answer "{name}" has an invalid Noul value')
            elif output_type in {"choice", "score"}:
                probabilities = answer.get("probabilities")
                expected_keys = (
                    set(specification["criteria"])
                    if output_type == "choice"
                    else {str(index) for index in range(len(specification["criteria"]))}
                )
                if not isinstance(probabilities, dict) or set(probabilities) != expected_keys:
                    raise ValueError(f'runtime v1 answer "{name}" has invalid probabilities')
                values = list(probabilities.values())
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                    for value in values
                ) or not math.isclose(sum(values), 1.0, abs_tol=1e-4):
                    raise ValueError(f'runtime v1 answer "{name}" has invalid probabilities')
                confidence = answer.get("confidence")
                maximum_probability = max(values)
                if (
                    isinstance(confidence, bool)
                    or not isinstance(confidence, (int, float))
                    or not math.isfinite(confidence)
                    or not 0 <= confidence <= 1
                    or not math.isclose(confidence, maximum_probability, abs_tol=1e-5)
                ):
                    raise ValueError(f'runtime v1 answer "{name}" has invalid confidence')
                if output_type == "choice":
                    expected_choice = max(
                        expected_keys,
                        key=lambda candidate: (probabilities[candidate], candidate),
                    )
                    if answer.get("choice") != expected_choice:
                        raise ValueError(f'runtime v1 answer "{name}" has an invalid choice')
                else:
                    expected_legend = {
                        str(index): criterion
                        for index, criterion in enumerate(specification["criteria"])
                    }
                    score = answer.get("score")
                    expected_score = sum(
                        int(level) * probabilities[level] for level in expected_keys
                    )
                    if (
                        answer.get("legend") != expected_legend
                        or isinstance(score, bool)
                        or not isinstance(score, (int, float))
                        or not math.isfinite(score)
                        or not 0 <= score <= len(expected_legend) - 1
                        or not math.isclose(score, expected_score, abs_tol=1e-4)
                    ):
                        raise ValueError(f'runtime v1 answer "{name}" has an invalid score')
            elif output_type == "text":
                text = answer.get("text")
                if not isinstance(text, str) or len(text.encode("utf-8")) > specification["max_bytes"]:
                    raise ValueError(f'runtime v1 answer "{name}" has an invalid text value')
            elif output_type == "structured":
                try:
                    encoded = json.dumps(
                        answer["value"],
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    if len(encoded) > specification["max_bytes"]:
                        raise ValueError("structured value exceeds max_bytes")
                    if not self.json_value_matches_schema(
                        answer["value"], specification["schema"]
                    ):
                        raise ValueError("structured value does not satisfy schema")
                except (TypeError, ValueError):
                    raise ValueError(
                        f'runtime v1 answer "{name}" has an invalid structured value'
                    ) from None
        return response

    @staticmethod
    def legacy_client_response(
        response: dict[str, object], source: str = "v4_trainer"
    ) -> dict[str, object]:
        if response.get("ok") is not True:
            return {
                "id": response.get("id"),
                "ok": False,
                "error": "legacy v4 probe failed",
                "legacy": True,
                "output_scope": "inherited",
                "source": source,
            }
        answers = response["answers"]
        if not isinstance(answers, dict) or len(answers) != 1:
            raise ValueError("legacy client response requires exactly one answer")
        answer = next(iter(answers.values()))
        if not isinstance(answer, dict):
            raise ValueError("legacy client answer is malformed")
        output = answer.get("text") if answer.get("type") == "text" else answer.get("value")
        return {
            "id": response["id"],
            "ok": True,
            "output": output,
            "legacy": True,
            "output_scope": "inherited",
            "source": source,
        }

    def run_checkpoint_probe(
        self, request: dict[str, object], legacy_client: bool
    ) -> None:
        state = self.read_state()
        try:
            legacy_request = self.legacy_trainer_request(
                request,
                max_bytes_limit=128,
                source="offline v4 checkpoint",
            )
        except ValueError as error:
            self.send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error)})
            return
        checkpoint = state.get("checkpoint")
        if not isinstance(checkpoint, str):
            self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "no checkpoint is available"})
            return
        executable = ROOT.parent / "target" / "release" / "river-pcn-generate"
        command = [
            str(executable),
            "--checkpoint",
            checkpoint,
            "--prompt",
            str(legacy_request["prompt"]),
            "--relax-steps",
            "1",
            "--max-bytes",
            str(legacy_request["max_bytes"]),
        ]
        schema_path: Path | None = None
        try:
            if legacy_request["mode"] == "text":
                command.append("--text")
            else:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                ) as schema_file:
                    json.dump(legacy_request["schema"], schema_file, separators=(",", ":"))
                    schema_path = Path(schema_file.name)
                command.extend(["--schema", str(schema_path)])
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=80,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or "checkpoint generation failed"
                self.send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": detail})
                return
            output: object
            if legacy_request["mode"] == "json":
                output = json.loads(completed.stdout)
            else:
                output = completed.stdout.removesuffix("\n")
            normalized = self.normalize_legacy_response(
                request,
                {"id": request["id"], "ok": True, "output": output},
            )
            if legacy_client:
                normalized = self.legacy_client_response(normalized, "v4_checkpoint")
            self.send_json(HTTPStatus.OK, normalized)
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            self.send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)})
        finally:
            if schema_path is not None:
                schema_path.unlink(missing_ok=True)

    def read_state(self) -> dict[str, object]:
        path = self.server.telemetry / "state.json"
        try:
            state = json.loads(path.read_text())
            if not isinstance(state, dict):
                raise ValueError("state is not an object")
            try:
                with (self.server.telemetry / "events.jsonl").open("rb") as stream:
                    size = stream.seek(0, 2)
                    stream.seek(max(0, size - 64 * 1024))
                    if stream.tell():
                        stream.readline()
                    event_lines = stream.readlines()
            except FileNotFoundError:
                event_lines = []
            for line in reversed(event_lines):
                try:
                    latest_event = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(latest_event, dict) or latest_event.get("run") != state.get("run"):
                    continue
                for key, value in latest_event.items():
                    state.setdefault(key, value)
                break
            try:
                manifest = json.loads((self.server.telemetry / "manifest.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                manifest = {}
            if isinstance(manifest, dict):
                for key, value in manifest.items():
                    state.setdefault(key, value)
            try:
                promotion = json.loads((self.server.telemetry / "promotion.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                promotion = None
            if isinstance(promotion, dict):
                state["promotion"] = promotion
            state["server_unix_millis"] = int(time.time() * 1000)
            try:
                registry = json.loads((ROOT.parent / "datasets" / "training-registry.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                registry = {}
            if isinstance(registry, dict) and isinstance(registry.get("datasets"), list):
                state["dataset_registry"] = [
                    {
                        "id": dataset.get("id"),
                        "kind": dataset.get("kind"),
                        "status": dataset.get("status"),
                        "source": dataset.get("source"),
                        "examples": sum(
                            value
                            for value in (dataset.get("splits") or {}).values()
                            if isinstance(value, int)
                        ),
                    }
                    for dataset in registry["datasets"]
                    if isinstance(dataset, dict)
                ]
            return state
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
            return {
                "status": "waiting",
                "detail": str(error),
                "server_unix_millis": int(time.time() * 1000),
            }

    def send_state(self) -> None:
        self.send_json(HTTPStatus.OK, self.read_state())

    def send_events(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        try:
            limit = min(1_000, max(1, int(query.get("limit", ["240"])[0])))
        except ValueError:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "limit must be an integer"})
            return
        path = self.server.telemetry / "events.jsonl"
        try:
            with path.open("rb") as stream:
                size = stream.seek(0, 2)
                stream.seek(max(0, size - 4 * 1024 * 1024))
                if stream.tell():
                    stream.readline()
                lines = stream.readlines()[-limit:]
        except FileNotFoundError:
            lines = []
        events = []
        for line in lines:
            try:
                event = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(event, dict):
                events.append(event)
        self.send_json(
            HTTPStatus.OK,
            {"events": events, "server_unix_millis": int(time.time() * 1000)},
        )

    def send_samples(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        try:
            limit = min(100, max(1, int(query.get("limit", ["20"])[0])))
        except ValueError:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "limit must be an integer"})
            return
        # samples.jsonl may be a bridge mirror of a remote trainer (rewritten from the remote copy), so local helpers
        # append their rows to samples.local.jsonl instead; both are merged here in time order.
        samples = []
        for name in ("samples.jsonl", "samples.local.jsonl"):
            path = self.server.telemetry / name
            try:
                with path.open("rb") as stream:
                    size = stream.seek(0, 2)
                    stream.seek(max(0, size - 1024 * 1024))
                    if stream.tell():
                        stream.readline()
                    lines = stream.readlines()[-limit:]
            except FileNotFoundError:
                continue
            for line in lines:
                try:
                    sample = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(sample, dict):
                    samples.append(sample)
        samples.sort(key=lambda sample: (sample.get("unix_millis") or sample.get("evaluated_at_unix_ms") or 0))
        samples = samples[-limit:]
        self.send_json(
            HTTPStatus.OK,
            {"samples": samples, "server_unix_millis": int(time.time() * 1000)},
        )

    def send_audits(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        try:
            limit = min(500, max(1, int(query.get("limit", ["120"])[0])))
        except ValueError:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "limit must be an integer"})
            return
        path = self.server.telemetry / "audit-metrics.jsonl"
        try:
            with path.open("rb") as stream:
                size = stream.seek(0, 2)
                stream.seek(max(0, size - 2 * 1024 * 1024))
                if stream.tell():
                    stream.readline()
                lines = stream.readlines()[-limit:]
        except FileNotFoundError:
            lines = []
        audits = []
        for line in lines:
            try:
                audit = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(audit, dict):
                audits.append(audit)
        self.send_json(
            HTTPStatus.OK,
            {"audits": audits, "server_unix_millis": int(time.time() * 1000)},
        )

    def stream_state(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.security_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        previous = None
        deadline = time.monotonic() + 3_600
        try:
            while time.monotonic() < deadline:
                state = self.read_state()
                encoded = json.dumps(state, separators=(",", ":"))
                if encoded != previous:
                    self.wfile.write(f"event: state\ndata: {encoded}\n\n".encode())
                    previous = encoded
                else:
                    self.wfile.write(b": heartbeat\n\n")
                self.wfile.flush()
                time.sleep(1.0)
        except (BrokenPipeError, ConnectionResetError):
            return

    def security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; connect-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'",
        )

    def send_json(self, status: HTTPStatus, value: object) -> None:
        body = json.dumps(value, separators=(",", ":")).encode()
        try:
            self.send_response(status)
            self.security_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--telemetry-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    token = args.token_file.read_text().strip()
    if len(token) < 32:
        raise SystemExit("portal token must contain at least 32 characters")
    server = PortalServer((args.host, args.port), args.telemetry_dir.resolve(), token)
    print(f"River Song portal: http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
