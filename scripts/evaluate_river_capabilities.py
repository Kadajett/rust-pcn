#!/usr/bin/env python3
"""Evaluate six fixed River fixture families through the public, no-key runtime API.

This is semantic evidence only for these fixtures, not a general benchmark or a
claim that next-byte accuracy, valid JSON, or a live checkpoint measures quality.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


HTTP_TIMEOUT = 620
STAGE_BOUNDARY_WAIT_SECONDS = 1800
ACTIVE_TRAINER_STATUSES = {"training", "training_tasks", "training_noul", "checkpointing"}  # portal/server.py
FIXTURE_VERSION = "river-capabilities-facts-v1"
FAMILIES = ("prose", "code", "structured", "choice", "score", "noul")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def fixtures() -> list[dict]:
    cases = []

    def add(name, family, inputs, specification, expected, contrast=None):
        cases.append({
            "id": name, "family": family, "contrast_with": contrast,
            "request": {"id": f"{FIXTURE_VERSION}:{name}", "inputs": inputs,
                        "outputs": {"answer": specification}},
            "expected": expected,
        })

    for country, capital in (("France", "Paris"), ("Germany", "Berlin")):
        add(f"prose-{country.lower()}", "prose", {"country": country},
            {"type": "text", "instructions":
             "Name the capital of the supplied country. Return only its name, optionally with a final period.",
             "max_bytes": 32},
            {"fact": capital, "allowed_answers": [capital.lower()],
             "normalization": "casefold, collapse whitespace, remove one final period"},
            "prose-france" if country == "Germany" else None)

    for operation in ("plus", "minus"):
        add(f"code-{operation}", "code", {"modality": "code", "x": 7, "amount": 3,
            "operation": operation},
            {"type": "text", "instructions":
             "Return only a Python function f(x) that adds amount to x for plus, or subtracts amount for minus. "
             "Use literal numbers, one return statement, and no calls or imports. No markdown.", "max_bytes": 48},
            {"function_cases": [{"input": x, "result": x + 3 if operation == "plus" else x - 3}
                                for x in (-2, 0, 7)]},
            "code-plus" if operation == "minus" else None)

    schema = {"type": "object", "properties": {
        "count": {"type": "integer", "minimum": 0, "maximum": 9},
        "status": {"type": "string", "enum": ["ready", "blocked"], "max_length": 7}},
        "required": ["count", "status"]}
    for name, count, status in (("ready", 3, "ready"), ("blocked", 1, "blocked")):
        add(f"structured-{name}", "structured", {"items": list(range(count)), "status": status},
            {"type": "structured", "instructions":
             "Return count equal to the number of items and status exactly as supplied.",
             "schema": schema, "max_bytes": 48}, {"value": {"count": count, "status": status}},
            "structured-ready" if name == "blocked" else None)

    choice_criteria = {"A": "The integer is even.", "B": "The integer is odd."}
    for name, number, criteria, option, contrast in (
        ("even", 2, choice_criteria, "A", None),
        ("odd", 3, choice_criteria, "B", "choice-even"),
        ("reversed", 2, {"A": choice_criteria["B"], "B": choice_criteria["A"]}, "B", "choice-even"),
    ):
        add(f"choice-{name}", "choice", {"integer": number},
            {"type": "choice", "instructions": "Choose the criterion true of the supplied integer.",
             "criteria": criteria},
            {"option": option, "probabilities": {key: float(key == option) for key in criteria}}, contrast)

    legend = ["Exactly zero items.", "Exactly one item.", "Exactly two items."]
    for name, items, criteria, position, contrast in (
        ("zero", [], legend, 0, None),
        ("two", ["a", "b"], legend, 2, "score-zero"),
        ("reversed", [], list(reversed(legend)), 2, "score-zero"),
    ):
        add(f"score-{name}", "score", {"items": items},
            {"type": "score", "instructions": "Score the number of supplied items using the ordered criteria.",
             "criteria": criteria},
            {"position": position, "legend": {str(i): text for i, text in enumerate(criteria)},
             "probabilities": {str(i): float(i == position) for i in range(3)}, "mae_tolerance": 0.25}, contrast)

    noul_criteria = {"true": "A uniformly selected token from the bag is red.",
                     "false": "A uniformly selected token from the bag is blue."}
    for name, bag, criteria, target, contrast in (
        ("quarter", ["red", "blue", "blue", "blue"], noul_criteria, 0.25, None),
        ("three-quarters", ["red", "red", "red", "blue"], noul_criteria, 0.75, "noul-quarter"),
        ("reversed", ["red", "blue", "blue", "blue"],
         {"true": noul_criteria["false"], "false": noul_criteria["true"]}, 0.75, "noul-quarter"),
    ):
        add(f"noul-{name}", "noul", {"bag": bag},
            {"type": "noul", "instructions":
             "Return the probability of the true criterion when one token is sampled uniformly from the bag.",
             "criteria": criteria}, {"probability": target, "mae_tolerance": 0.1}, contrast)
    return cases


def fetch(url: str, payload: dict | None = None) -> dict:
    started = timestamp()
    clock = time.monotonic()
    record = {"url": url, "started_at": started, "http_status": None, "body": None, "errors": []}
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
    request = Request(url, data=data, headers={"Accept": "application/json", **(
        {"Content-Type": "application/json"} if data is not None else {})},
        method="GET" if data is None else "POST")
    try:
        with urlopen(request, timeout=HTTP_TIMEOUT) as response:
            record["http_status"] = response.status
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("response exceeded 2000000 bytes")
        record["body"] = {"raw": raw.decode("utf-8", errors="replace")}
        record["body"] = json.loads(raw, parse_constant=reject_constant)
        if not isinstance(record["body"], dict):
            raise ValueError("response must be a JSON object")
    except HTTPError as error:
        record["http_status"] = error.code
        try:
            detail = error.read(65_536).decode("utf-8", errors="replace")
        except (OSError, HTTPException) as read_error:
            detail = f"could not read error body: {type(read_error).__name__}: {read_error}"
        record["errors"].append(f"HTTP {error.code}: {detail}")
        try:
            record["body"] = json.loads(detail, parse_constant=reject_constant)
        except (ValueError, json.JSONDecodeError):
            record["body"] = {"raw": detail}
    except (URLError, TimeoutError, OSError, HTTPException, ValueError, RecursionError) as error:
        record["errors"].append(f"{type(error).__name__}: {error}")
    record["finished_at"] = timestamp()
    record["latency_seconds"] = round(time.monotonic() - clock, 6)
    return record


def finite_number(value) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def arithmetic_source(source: str) -> tuple[str, bool]:
    """Whitelist the entire AST before spawning; original model bytes are not run."""
    if not isinstance(source, str) or len(source.encode("utf-8")) > 64:
        raise ValueError("code must be UTF-8 text of at most 64 bytes")
    tree = ast.parse(source.strip(), mode="exec")
    if len(list(ast.walk(tree))) > 64 or len(tree.body) != 1:
        raise ValueError("code must contain one small arithmetic expression or function")
    root = tree.body[0]
    function = isinstance(root, ast.FunctionDef)
    if function:
        args = root.args
        if (root.name != "f" or root.decorator_list or root.returns is not None
                or root.type_comment is not None or getattr(root, "type_params", [])
                or args.posonlyargs or len(args.args) != 1 or args.args[0].arg != "x"
                or args.args[0].annotation is not None or args.args[0].type_comment is not None
                or args.vararg or args.kwarg or args.kwonlyargs or args.defaults or args.kw_defaults
                or len(root.body) != 1 or not isinstance(root.body[0], ast.Return)):
            raise ValueError("only undecorated f(x) with one arithmetic return is allowed")
        expression = root.body[0].value
    elif isinstance(root, ast.Expr):
        expression = root.value
    else:
        raise ValueError("only an expression or f(x) is allowed")

    def check(node):
        if isinstance(node, ast.Constant) and finite_number(node.value) and abs(node.value) <= 1_000_000:
            return
        if function and isinstance(node, ast.Name) and node.id == "x" and isinstance(node.ctx, ast.Load):
            return
        if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod):
            check(node.left)
            check(node.right)
            return
        if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            check(node.operand)
            return
        raise ValueError("disallowed arithmetic AST: " + type(node).__name__)

    check(expression)
    return ast.unparse(tree), function


# This trusted wrapper executes only canonical, parent-whitelisted arithmetic.
# No shell, model-supplied calls, imports, attributes, loops, or ambient builtins.
CODE_CHILD = """
import json, math, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (1, 1))
resource.setrlimit(resource.RLIMIT_AS, (67108864, 67108864))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
data = json.load(sys.stdin)
namespace = {"__builtins__": {}}
try:
    if data["function"]:
        exec(compile(data["source"], "<arithmetic>", "exec"), namespace)
        result = namespace["f"](data["x"])
    else:
        result = eval(compile(data["source"], "<arithmetic>", "eval"), namespace)
    if type(result) not in (int, float) or not math.isfinite(result):
        raise ValueError("non-finite numeric result")
    print(json.dumps({"result": result}, allow_nan=False))
except Exception as error:
    print(json.dumps({"error": type(error).__name__ + ": " + str(error)}))
"""


def execute_arithmetic(source: str, x: int) -> dict:
    canonical, function = arithmetic_source(source)
    completed = subprocess.run([sys.executable, "-I", "-S", "-c", CODE_CHILD],
        input=json.dumps({"source": canonical, "function": function, "x": x}),
        text=True, capture_output=True, timeout=3, env={}, cwd="/")
    if completed.returncode:
        raise ValueError(f"restricted arithmetic subprocess exited {completed.returncode}: {completed.stderr[:512]}")
    result = json.loads(completed.stdout, parse_constant=reject_constant)
    if "error" in result:
        raise ValueError(result["error"])
    return {"canonical_source": canonical, "result": result["result"]}


def distribution(answer: dict, expected: dict) -> dict:
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(expected):
        raise ValueError("distribution keys do not match requested criteria")
    if (any(not finite_number(p) or not 0 <= p <= 1 for p in probabilities.values())
            or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-4)):
        raise ValueError("invalid finite normalized probability distribution")
    confidence = answer.get("confidence")
    if not finite_number(confidence) or not math.isclose(confidence, max(probabilities.values()), abs_tol=1e-5):
        raise ValueError("confidence is not the maximum option probability")
    return probabilities


def grade(case: dict, answer: dict) -> dict:
    family, expected = case["family"], case["expected"]
    specification = case["request"]["outputs"]["answer"]
    if not isinstance(answer, dict) or answer.get("type") != specification["type"]:
        raise ValueError("missing answer or wrong answer type")
    expected_scope = "request_conditioned" if family in ("choice", "score", "noul") else "inherited"
    if answer.get("output_scope") != expected_scope:
        raise ValueError("answer output_scope does not match the runtime contract")
    if family == "prose":
        text = answer.get("text")
        if not isinstance(text, str) or len(text.encode("utf-8")) > specification["max_bytes"]:
            raise ValueError("prose is not text within the requested byte limit")
        normalized = re.sub(r"\s+", " ", text.strip().casefold())
        if normalized.endswith("."):
            normalized = normalized[:-1]
        passed = normalized in expected["allowed_answers"]
        return {"pass": passed, "normalized_answer": normalized,
                "diff": {"expected_fact": expected["fact"], "actual_complete_answer": text,
                         "allowed_answer_match": passed}}
    if family == "code":
        text = answer.get("text")
        if not isinstance(text, str) or len(text.encode("utf-8")) > specification["max_bytes"]:
            raise ValueError("code is not text within the requested byte limit")
        _, function = arithmetic_source(text)
        if not function:
            raise ValueError("code fixture requires f(x), not a single arithmetic expression")
        executions = []
        for check in expected["function_cases"]:
            execution = execute_arithmetic(text, check["input"])
            executions.append({"input": check["input"], "expected": check["result"],
                               **execution, "result_delta": execution["result"] - check["result"]})
        return {"pass": all(item["result_delta"] == 0 for item in executions),
                "executions": executions,
                "diff": {"result_deltas": [item["result_delta"] for item in executions]}}
    if family == "structured":
        value = answer.get("value")
        json.dumps(value, allow_nan=False)
        schema_valid = (isinstance(value, dict) and set(value) == {"count", "status"}
            and type(value["count"]) is int and 0 <= value["count"] <= 9
            and value["status"] in ("ready", "blocked"))
        semantic_equal = schema_valid and value == expected["value"]
        return {"pass": semantic_equal, "valid_json": True, "schema_valid": schema_valid,
                "semantic_values_equal": semantic_equal,
                "diff": {key: {"expected": wanted, "actual": value.get(key) if isinstance(value, dict) else None}
                         for key, wanted in expected["value"].items()}}
    if family in ("choice", "score"):
        probabilities = distribution(answer, expected["probabilities"])
        brier = sum((probabilities[key] - target) ** 2 for key, target in expected["probabilities"].items())
        delta = {key: probabilities[key] - target for key, target in expected["probabilities"].items()}
        common = {"brier": brier, "confidence": answer["confidence"], "distribution_valid": True}
        if family == "choice":
            selected = max(probabilities, key=lambda key: (probabilities[key], key))
            if answer.get("choice") != selected:
                raise ValueError("choice does not agree with distribution argmax")
            return {**common, "pass": selected == expected["option"],
                    "target_probability": probabilities[expected["option"]],
                    "diff": {"expected_option": expected["option"], "actual_option": selected,
                             "probability_delta": delta}}
        position = sum(int(key) * p for key, p in probabilities.items())
        legend_matches = answer.get("legend") == expected["legend"]
        if not finite_number(answer.get("score")) or not math.isclose(answer["score"], position, abs_tol=1e-4):
            raise ValueError("score does not agree with distribution expected position")
        mae = abs(position - expected["position"])
        return {**common, "pass": legend_matches and mae <= expected["mae_tolerance"],
                "ordered_legend_matches": legend_matches, "expected_position": position, "mae": mae,
                "diff": {"position_delta": position - expected["position"], "probability_delta": delta,
                         "expected_legend": expected["legend"], "actual_legend": answer.get("legend")}}
    probability = answer.get("noul")
    if not finite_number(probability) or not 0 <= probability <= 1:
        raise ValueError("Noul must be finite and between zero and one inclusive")
    delta = probability - expected["probability"]
    return {"pass": abs(delta) <= expected["mae_tolerance"], "mae": abs(delta),
            "brier": delta ** 2, "diff": {"probability_delta": delta}}


def state_identity(record: dict) -> dict:
    state = record.get("body")
    if not isinstance(state, dict):
        return {"checkpoint": None, "batch": None, "epoch": None, "status": None}
    return {key: state.get(key) for key in ("checkpoint", "batch", "epoch", "status", "run", "active_expert")}


def contrasts(results: list[dict]) -> list[dict]:
    indexed = {case["id"]: case for case in results}
    report = []
    for case in results:
        baseline_id = case["contrast_with"]
        if baseline_id is None:
            continue
        baseline = indexed[baseline_id]
        first, second = baseline["actual"], case["actual"]
        available = isinstance(first, dict) and isinstance(second, dict)
        changed = None
        if available:
            keys = {"prose": ("text",), "code": ("text",), "structured": ("value",),
                    "choice": ("choice", "probabilities"), "score": ("score", "probabilities"),
                    "noul": ("noul",)}[case["family"]]
            changed = any(first.get(key) != second.get(key) for key in keys)
        report.append({"baseline": baseline_id, "case": case["id"], "family": case["family"],
                       "expected": "different semantic answer or target distribution",
                       "actual_changed": changed, "insensitivity_observed": changed is False,
                       "both_semantically_pass": baseline["pass"] and case["pass"],
                       "limitation": "Different outputs alone do not prove criterion sensitivity or correctness."})
    return report


def aggregate(results: list[dict]) -> dict:
    report = {}
    for family in FAMILIES:
        rows = [row for row in results if row["family"] == family]
        graded = [row for row in rows if row.get("actual") is not None]
        item = {"cases": len(rows), "graded_cases": len(graded),
                "unavailable_cases": len(rows) - len(graded),
                "passed": sum(row["pass"] for row in graded),
                "failed": sum(not row["pass"] for row in graded),
                "cases_with_errors": sum(bool(row["errors"]) for row in rows),
                "total_latency_seconds": sum(row["http"]["latency_seconds"] for row in rows)}
        for metric in ("brier", "mae", "confidence", "target_probability"):
            values = [row["grading"][metric] for row in graded if metric in row["grading"]]
            if values:
                item[f"mean_{metric}"] = sum(values) / len(values)
                item[f"{metric}_graded_cases"] = len(values)
        report[family] = item
    return report


def publish_report(directory: Path, encoded: str) -> None:
    """Atomically replace only the latest public report, never an artifact."""
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory,
            prefix=".capability-evidence.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(directory / "capability-evidence.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8799/api/test", help="public runtime test endpoint")
    parser.add_argument("--state-url", help="state endpoint; defaults to /api/state at the test endpoint origin")
    parser.add_argument("--output", type=Path, help="also create a JSON result artifact; refuses to overwrite")
    parser.add_argument("--publish-dir", type=Path,
                        help="atomically replace capability-evidence.json in this telemetry directory")
    args = parser.parse_args()
    origin = urlsplit(args.url)
    if origin.scheme not in ("http", "https") or not origin.netloc or origin.username or origin.password:
        parser.error("--url must be an HTTP(S) URL without credentials")
    state_url = args.state_url or urlunsplit((origin.scheme, origin.netloc, "/api/state", "", ""))
    state_parts = urlsplit(state_url)
    if state_parts.scheme not in ("http", "https") or not state_parts.netloc or state_parts.username or state_parts.password:
        parser.error("--state-url must be an HTTP(S) URL without credentials")
    if args.output is not None and args.output.exists():
        parser.error("--output already exists; choose a new artifact path")

    # The portal rejects probes between stages (data loading, checkpoint boundary); wait for active training
    # so a scheduled run does not publish fifteen "trainer not active" failures over the previous report.
    deadline = time.monotonic() + STAGE_BOUNDARY_WAIT_SECONDS
    while True:
        before = fetch(state_url)
        status = before["body"].get("status") if isinstance(before["body"], dict) else None
        if status in ACTIVE_TRAINER_STATUSES or time.monotonic() >= deadline:
            break
        time.sleep(10)
    started = timestamp()
    results = []
    for case in fixtures():
        http = fetch(args.url, case["request"])
        body = http["body"]
        answer = None
        errors = list(http["errors"])
        if not errors:
            if not isinstance(body, dict) or body.get("id") != case["request"]["id"] or body.get("ok") is not True:
                errors.append("runtime did not return an ok response with the requested id")
            elif not isinstance(body.get("answers"), dict) or set(body["answers"]) != {"answer"}:
                errors.append("runtime response did not contain exactly the requested named answer")
            else:
                answer = body["answers"]["answer"]
        grading = {}
        if not errors:
            try:
                grading = grade(case, answer)
            except (ValueError, TypeError, SyntaxError, OverflowError, OSError, subprocess.TimeoutExpired) as error:
                errors.append(f"grading rejected output: {type(error).__name__}: {error}")
                grading = {"pass": False, "rejected_output": True,
                           "diff": {"semantic_rejection": str(error)}}
        results.append({**case, "actual": answer, "diff": grading.get("diff", {"ungraded": True}),
                        "pass": bool(grading.get("pass", False)), "errors": errors,
                        "grading": grading, "http": http})
    after = fetch(state_url)
    first, last = state_identity(before), state_identity(after)
    for record, identity in ((before, first), (after, last)):
        if not record["errors"] and (not isinstance(identity.get("checkpoint"), str)
                                     or type(identity.get("batch")) is not int):
            record["errors"].append("state endpoint did not supply an exact checkpoint path and integer batch")
    identities_available = (not before["errors"] and not after["errors"]
                            and all(identity.get("checkpoint") is not None and identity.get("batch") is not None
                                    for identity in (first, last)))
    changed = first != last if identities_available else None
    live = any(identity.get("status") in ("training", "training_tasks", "training_noul", "checkpointing")
               for identity in (first, last))
    report = {
        "schema": "river-capability-evidence-v1", "fixture_version": FIXTURE_VERSION,
        "started_at": started, "finished_at": timestamp(), "test_url": args.url,
        "scope": "Semantic grading applies only to these fifteen fixed fact-based fixtures in six families; not general capability or next-byte quality.",
        "limits": {"http_timeout_seconds": HTTP_TIMEOUT, "portal_timeout_seconds": 600,
                   "automatic_retries": 0, "concurrent_requests": 1, "outputs_per_request": 1,
                   "text_max_bytes": 48,
                   "typed_candidate_pairs": {"noul": 1, "choice": 2, "score": 3},
                   "code_subprocess_timeout_seconds": 3, "code_subprocess_cpu_seconds": 1,
                   "code_subprocess_address_space_bytes": 67_108_864,
                   "code_ast": "whitelisted undecorated f(x) with one arithmetic return; bounded numeric literals and + - * / // % only; exact outputs on x=-2,0,7 required; top-level constant arithmetic does not pass",
                   "brier_convention": "Choice/Score sum squared category errors; Noul squared error against the continuous soft target",
                   "pass_rules": "prose finite complete answers; code exact result; structured exact values; Choice correct argmax; Score MAE <=0.25 plus ordered legend; Noul MAE <=0.1"},
        "checkpoint_state": {"before": before, "after": after, "before_identity": first, "after_identity": last,
                             "identity_changed": changed, "nonfrozen_live_state": live or changed is True,
                             "frozen_verified": False,
                             "warning": "State snapshots do not prove immutable weights. Live requests may use different batches or weights, even if the checkpoint path is unchanged."},
        "cases": results, "contrasts": contrasts(results), "aggregate_by_family": aggregate(results),
        "errors": [{"phase": phase, "errors": record["errors"]}
                   for phase, record in (("state_before", before), ("state_after", after)) if record["errors"]],
    }
    encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    print(encoded, end="")
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    if args.publish_dir is not None:
        publish_report(args.publish_dir, encoded)
    # Semantic failures and unavailable evidence remain visible to automation.
    raise SystemExit(0 if all(case["pass"] for case in results) and not report["errors"] else 1)


if __name__ == "__main__":
    main()
