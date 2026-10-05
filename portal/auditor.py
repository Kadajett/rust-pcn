#!/usr/bin/env python3
"""Periodically grade River training samples with TypeSafe Jev and retain audit data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_URL = "https://api.typesafe.ai/v1/systemone"
NOUL_MAE_TOLERANCE = 0.1
NOUL_SATURATION_EPSILON = 0.0001
NOUL_SOFT_TARGET_MARGIN = 0.05
SCORE_LEVELS = [
    "No demonstrated capability; outputs are unrelated or degenerate",
    "Weak; occasional plausible output dominated by errors",
    "Mixed; recognizable behavior with frequent errors",
    "Useful emerging behavior; mostly plausible with bounded errors",
    "Strong; outputs consistently fit the supplied contexts and targets",
]
QUESTIONS = {
    "context_fit": {
        "type": "score",
        "instructions": "Rate how well Song's named answers fit the supplied request, output instructions, criteria, and available supervision. For supervised Noul answers, compare the returned value with the target.",
        "criteria": SCORE_LEVELS,
    },
    "language_quality": {
        "type": "score",
        "instructions": "Rate the response quality demonstrated by the output types present. Judge Noul answers against their criteria and targets, structured answers for valid useful values, and text or code for coherence. Do not penalize a batch merely because it contains no text output.",
        "criteria": SCORE_LEVELS,
    },
    "behavioral_diversity": {
        "type": "score",
        "instructions": "Rate whether the answers vary appropriately with requests and supervised targets rather than collapsing to one response across unrelated examples.",
        "criteria": SCORE_LEVELS,
    },
    "learned_signal": {
        "type": "noul",
        "instructions": "Do these samples provide evidence of learned request-sensitive behavior beyond random or single-output collapse?",
        "criteria": {
            "true": "Answers respond to the requested output contract and agree with multiple supervised targets",
            "false": "Answers are unrelated to requests or dominated by one value regardless of target",
        },
    },
    "strongest_area": {
        "type": "choice",
        "instructions": "Which capability is strongest in this audit batch?",
        "criteria": {
            "typed_decision": "Noul, Choice, or Score answers follow their criteria or rubric",
            "structured": "Schema-constrained structured answers are valid and useful",
            "text_code": "Natural-language or code answers fit the request",
            "multimodal": "Answers respond to image or structured-state evidence",
            "no_clear_strength": "No capability is clearly stronger",
        },
    },
    "dominant_failure": {
        "type": "choice",
        "instructions": "Which failure mode most limits this audit batch?",
        "criteria": {
            "repeated_output": "The same answer dominates requests with different targets",
            "target_mismatch": "Answers vary but disagree with available supervision",
            "weak_language": "Text or code is locally incoherent",
            "schema_error": "Structured output violates its requested shape",
            "insufficient_evidence": "The batch is too small or incomplete to identify a failure",
        },
    },
}


def read_env_key(path: Path | None) -> str | None:
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key
    if path is None:
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for raw in lines:
        line = raw.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        if not line.startswith("TYPESAFE_API_KEY="):
            continue
        value = line.split("=", 1)[1].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return value or None
    return None


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def read_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    try:
        with path.open("rb") as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - 1024 * 1024))
            if stream.tell():
                stream.readline()
            lines = stream.readlines()[-limit:]
    except FileNotFoundError:
        return []
    records = []
    for line in lines:
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None

def grade_noul(target: float, prediction: float) -> dict[str, float | bool]:
    absolute_error = abs(target - prediction)
    soft_target = NOUL_SOFT_TARGET_MARGIN < target < 1 - NOUL_SOFT_TARGET_MARGIN
    saturated = (
        prediction <= NOUL_SATURATION_EPSILON
        or prediction >= 1 - NOUL_SATURATION_EPSILON
    )
    saturation_failure = soft_target and saturated
    return {
        "absolute_error": absolute_error,
        "squared_error": absolute_error * absolute_error,
        "saturated": saturation_failure,
        "passed": absolute_error <= NOUL_MAE_TOLERANCE and not saturation_failure,
    }

def probability_diagnostics(probabilities: object, expected: str) -> dict[str, object] | None:
    if not isinstance(probabilities, dict):
        return None
    target = finite_number(probabilities.get(expected))
    if target is None:
        return None
    competitors = [
        value
        for name, raw in probabilities.items()
        if str(name) != expected and (value := finite_number(raw)) is not None
    ]
    strongest = max(competitors, default=0.0)
    margin = target - strongest
    return {
        "target_probability": target,
        "margin": margin,
        "tied": abs(margin) <= 1e-6,
        "unique_top": margin > 1e-6,
    }

def sample_signature(samples: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        samples,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def objective_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    exact_matches: list[bool] = []
    by_output_type: dict[str, dict[str, Any]] = {}
    noul_errors: list[float] = []
    noul_squared_errors: list[float] = []
    pinball_errors: list[float] = []

    for sample in samples:
        if sample.get("schema") == "river-universal-output-telemetry-v1":
            supervision = sample.get("supervision")
            response = sample.get("response")
            if not isinstance(supervision, dict) or not isinstance(response, dict):
                continue
            label = supervision.get("label")
            target = finite_number(supervision.get("target"))
            answers = response.get("answers")
            answer = answers.get(label) if isinstance(answers, dict) and isinstance(label, str) else None
            prediction = finite_number(answer.get("noul")) if isinstance(answer, dict) else None
            if target is None or prediction is None:
                continue
            grade = grade_noul(target, prediction)
            error = float(grade["absolute_error"])
            squared_error = float(grade["squared_error"])
            passed = bool(grade["passed"])
            saturated = bool(grade["saturated"])
            exact_matches.append(passed)
            noul_errors.append(error)
            noul_squared_errors.append(squared_error)
            metrics = by_output_type.setdefault(
                "noul",
                {
                    "total": 0,
                    "strict_passes": 0,
                    "saturated": 0,
                    "absolute_error_sum": 0.0,
                    "squared_error_sum": 0.0,
                },
            )
            metrics["total"] += 1
            metrics["strict_passes"] += int(passed)
            metrics["saturated"] += int(saturated)
            metrics["absolute_error_sum"] += error
            metrics["squared_error_sum"] += squared_error
            continue

        output_type = sample.get("output_type")
        matched = sample.get("matched")
        expected = sample.get("expected")
        predicted = sample.get("predicted")

        if output_type == "choice" and isinstance(expected, str) and isinstance(predicted, dict):
            selected_correct = predicted.get("choice") == expected
            diagnostic = probability_diagnostics(predicted.get("probabilities"), expected)
            unique_top = bool(diagnostic and diagnostic["unique_top"])
            tied = bool(diagnostic and diagnostic["tied"])
            exact_matches.append(selected_correct and unique_top)
            metrics = by_output_type.setdefault(
                "choice",
                {
                    "total": 0,
                    "label_correct": 0,
                    "unique_top_correct": 0,
                    "ties": 0,
                    "target_probability_sum": 0.0,
                    "target_probability_count": 0,
                },
            )
            metrics["total"] += 1
            metrics["label_correct"] += int(selected_correct)
            metrics["unique_top_correct"] += int(selected_correct and unique_top)
            metrics["ties"] += int(tied)
            if diagnostic:
                metrics["target_probability_sum"] += diagnostic["target_probability"]
                metrics["target_probability_count"] += 1
            continue

        if output_type == "score" and isinstance(expected, dict) and isinstance(predicted, dict):
            target = finite_number(expected.get("score"))
            prediction = finite_number(predicted.get("score"))
            if target is None or prediction is None:
                continue
            error = abs(target - prediction)
            expected_level = str(int(target)) if target.is_integer() else str(target)
            diagnostic = probability_diagnostics(predicted.get("probabilities"), expected_level)
            unique_top = bool(diagnostic and diagnostic["unique_top"])
            tied = bool(diagnostic and diagnostic["tied"])
            exact_matches.append(error <= 1e-6 and unique_top)
            metrics = by_output_type.setdefault(
                "score",
                {
                    "total": 0,
                    "absolute_error_sum": 0.0,
                    "point_exact": 0,
                    "unique_top_correct": 0,
                    "ties": 0,
                    "target_probability_sum": 0.0,
                    "target_probability_count": 0,
                },
            )
            metrics["total"] += 1
            metrics["absolute_error_sum"] += error
            metrics["point_exact"] += int(error <= 1e-6)
            metrics["unique_top_correct"] += int(unique_top)
            metrics["ties"] += int(tied)
            if diagnostic:
                metrics["target_probability_sum"] += diagnostic["target_probability"]
                metrics["target_probability_count"] += 1
            continue

        if isinstance(output_type, str) and isinstance(matched, bool):
            exact_matches.append(matched)
            metrics = by_output_type.setdefault(output_type, {"total": 0, "exact_correct": 0})
            metrics["total"] += 1
            metrics["exact_correct"] += int(matched)
            accuracy = finite_number(sample.get("accuracy"))
            if output_type == "structured" and accuracy is not None:
                metrics["field_accuracy_sum"] = metrics.get("field_accuracy_sum", 0.0) + accuracy
                metrics["schema_valid"] = metrics.get("schema_valid", 0) + int(
                    sample.get("schema_valid") is True
                )

        if str(sample.get("input_modality", sample.get("modality", "unknown"))) != "pinball":
            continue
        if not isinstance(expected, list) or not isinstance(predicted, list) or len(expected) != len(predicted):
            continue
        for target, prediction in zip(expected, predicted):
            target_value = finite_number(target)
            prediction_value = finite_number(prediction)
            if target_value is not None and prediction_value is not None:
                error = abs(target_value - prediction_value)
                noul_errors.append(error)
                noul_squared_errors.append(error * error)
                pinball_errors.append(error)

    for metrics in by_output_type.values():
        total = int(metrics.get("total", 0))
        if total and "absolute_error_sum" in metrics:
            metrics["mae"] = metrics["absolute_error_sum"] / total
        if total and "squared_error_sum" in metrics:
            metrics["brier"] = metrics["squared_error_sum"] / total
        if total and "field_accuracy_sum" in metrics:
            metrics["field_accuracy"] = metrics["field_accuracy_sum"] / total
        probability_count = int(metrics.get("target_probability_count", 0))
        if probability_count:
            metrics["mean_target_probability"] = (
                metrics["target_probability_sum"] / probability_count
            )
        for internal in (
            "absolute_error_sum",
            "squared_error_sum",
            "field_accuracy_sum",
            "target_probability_sum",
            "target_probability_count",
        ):
            metrics.pop(internal, None)

    return {
        "exact_accuracy": sum(exact_matches) / len(exact_matches) if exact_matches else None,
        "exact_correct": sum(exact_matches),
        "exact_total": len(exact_matches),
        "by_output_type": by_output_type,
        "noul_mae": sum(noul_errors) / len(noul_errors) if noul_errors else None,
        "noul_brier": (
            sum(noul_squared_errors) / len(noul_squared_errors)
            if noul_squared_errors
            else None
        ),
        "pinball_mae": sum(pinball_errors) / len(pinball_errors) if pinball_errors else None,
    }


def audit_state(training: dict[str, Any], samples: list[dict[str, Any]], objective: dict[str, Any]) -> dict[str, Any]:
    compact_samples = []
    for sample in samples:
        if sample.get("schema") == "river-universal-output-telemetry-v1":
            compact_samples.append(
                {
                    "kind": sample.get("kind"),
                    "inputs": sample.get("request", {}).get("inputs")
                    if isinstance(sample.get("request"), dict)
                    else None,
                    "requested_outputs": sample.get("request", {}).get("outputs")
                    if isinstance(sample.get("request"), dict)
                    else None,
                    "answers": sample.get("response", {}).get("answers")
                    if isinstance(sample.get("response"), dict)
                    else None,
                    "supervision": sample.get("supervision"),
                }
            )
        else:
            compact_samples.append(
                {
                    "dataset_id": sample.get("dataset_id"),
                    "input_modality": sample.get("input_modality"),
                    "output_type": sample.get("output_type"),
                    "task": sample.get("task"),
                    "input": sample.get("input"),
                    "predicted": sample.get("predicted"),
                    "expected": sample.get("expected"),
                    "matched": sample.get("matched"),
                    "accuracy": sample.get("accuracy"),
                    "diff": sample.get("diff"),
                    "schema_valid": sample.get("schema_valid"),
                    "masked_values": sample.get("masked_values"),
                    "input_values": sample.get("input_values"),
                }
            )
    return {
        "task": "Audit Song's current named input-to-output behavior. Judge only evidence present in these training samples and use explicit supervision when supplied.",
        "model_state": {
            "run": training.get("run"),
            "epoch": training.get("epoch"),
            "batch": training.get("batch"),
            "mean_free_energy": training.get("mean_free_energy", training.get("free_energy")),
            "mean_positive_energy": training.get("mean_positive_energy", training.get("positive_energy")),
        },
        "objective_metrics": objective,
        "samples": compact_samples,
    }


def call_jev(api_key: str, state: dict[str, Any], timeout: float) -> dict[str, Any]:
    payload = json.dumps(
        {"model": "jev-latest", "state": state, "questions": QUESTIONS},
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    request = urllib.request.Request(
        os.environ.get("TYPESAFE_API_URL", DEFAULT_URL),
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "river-song-quality-auditor/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"TypeSafe HTTP {error.code}: {body[:500]}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"TypeSafe request failed: {error}") from error
    if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
        raise RuntimeError("TypeSafe response omitted typed answers")
    return result


def score_value(answers: dict[str, Any], key: str) -> float | None:
    answer = answers.get(key)
    if not isinstance(answer, dict):
        return None
    score = finite_number(answer.get("score"))
    return score / (len(SCORE_LEVELS) - 1) if score is not None else None


def normalize_audit(
    training: dict[str, Any],
    samples: list[dict[str, Any]],
    objective: dict[str, Any],
    result: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    answers = result["answers"]
    context_fit = score_value(answers, "context_fit")
    language_quality = score_value(answers, "language_quality")
    behavioral_diversity = score_value(answers, "behavioral_diversity")
    learned = answers.get("learned_signal") if isinstance(answers.get("learned_signal"), dict) else {}
    learned_signal = finite_number(learned.get("noul"))
    components = [value for value in (context_fit, language_quality, behavioral_diversity, learned_signal) if value is not None]
    jev_quality = sum(components) / len(components) if components else None
    strongest = answers.get("strongest_area") if isinstance(answers.get("strongest_area"), dict) else {}
    failure = answers.get("dominant_failure") if isinstance(answers.get("dominant_failure"), dict) else {}
    timestamp = int(time.time() * 1000)
    metrics = {
        "schema": "river-jev-quality-metrics-v4",
        "unix_millis": timestamp,
        "run": training.get("run"),
        "epoch": training.get("epoch"),
        "batch": training.get("batch"),
        "samples_graded": len(samples),
        "exact_accuracy": objective.get("exact_accuracy"),
        "exact_correct": objective.get("exact_correct"),
        "exact_total": objective.get("exact_total"),
        "pinball_mae": objective.get("pinball_mae"),
        "noul_mae": objective.get("noul_mae"),
        "noul_brier": objective.get("noul_brier"),
        "by_output_type": objective.get("by_output_type"),
        "context_fit": context_fit,
        "language_quality": language_quality,
        "behavioral_diversity": behavioral_diversity,
        "learned_signal": learned_signal,
        "jev_quality": jev_quality,
        "strongest_area": strongest.get("choice"),
        "dominant_failure": failure.get("choice"),
        "jev_confidence": {
            key: finite_number(value.get("confidence"))
            for key, value in answers.items()
            if isinstance(value, dict) and value.get("confidence") is not None
        },
        "mean_free_energy": training.get("mean_free_energy", training.get("free_energy")),
        "mean_positive_energy": training.get("mean_positive_energy", training.get("positive_energy")),
        "provider_model": result.get("model"),
        "usage": result.get("usage") if isinstance(result.get("usage"), dict) else {},
    }
    full = {
        **metrics,
        "schema": "river-jev-quality-audit-v4",
        "questions": QUESTIONS,
        "answers": answers,
        "objective_metrics": objective,
        "training_samples": samples,
        "future_training_candidate": True,
        "training_use": "quality/calibration metadata; requires promotion review before entering active training",
    }
    return metrics, full


def audit_once(
    args: argparse.Namespace,
    api_key: str,
    training: dict[str, Any],
    samples: list[dict[str, Any]],
    signature: str,
) -> int:
    objective = objective_metrics(samples)
    state = audit_state(training, samples, objective)
    result = call_jev(api_key, state, args.timeout)
    metrics, full = normalize_audit(training, samples, objective, result)
    metrics["sample_signature"] = signature
    full["sample_signature"] = signature
    append_jsonl(args.telemetry_dir / "audit-metrics.jsonl", metrics)
    append_jsonl(args.audit_archive, full)
    atomic_json(args.telemetry_dir / "audit.json", metrics)
    print(
        f"audit batch={metrics['batch']} exact={metrics['exact_accuracy']} "
        f"jev_quality={metrics['jev_quality']} strength={metrics['strongest_area']}",
        flush=True,
    )
    return int(training.get("batch") or 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry-dir", type=Path, required=True)
    parser.add_argument("--audit-archive", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0 or args.timeout <= 0 or not 2 <= args.sample_count <= 100:
        parser.error("interval and timeout must be positive; sample-count must be 2..100")
    return args


def main() -> None:
    args = parse_args()
    api_key = read_env_key(args.env_file)
    if not api_key:
        raise SystemExit("TYPESAFE_API_KEY is unavailable")
    last_batch = -1
    last_run = ""
    last_sample_signature = ""
    try:
        latest = read_json(args.telemetry_dir / "audit.json")
        if latest.get("schema") == "river-jev-quality-metrics-v4":
            last_batch = int(latest.get("batch") or -1)
            last_run = str(latest.get("run") or "")
            last_sample_signature = str(latest.get("sample_signature") or "")
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    while True:
        try:
            state = read_json(args.telemetry_dir / "state.json")
            batch = int(state.get("batch") or 0)
            run = str(state.get("run") or "")
            if args.once or run != last_run or batch > last_batch:
                samples = read_tail(args.telemetry_dir / "samples.jsonl", args.sample_count)
                if len(samples) < args.sample_count:
                    print(
                        f"audit waiting: {len(samples)}/{args.sample_count} current samples",
                        flush=True,
                    )
                    last_batch = batch
                else:
                    signature = sample_signature(samples)
                    if signature != last_sample_signature:
                        last_batch = audit_once(args, api_key, state, samples, signature)
                        last_sample_signature = signature
                    else:
                        last_batch = batch
                last_run = run
        except Exception as error:  # keep audit failure isolated from training
            print(f"audit error: {error}", flush=True)
            if args.once:
                raise SystemExit(1) from error
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
