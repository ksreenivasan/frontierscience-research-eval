from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .core import (
    DATASET_REVISION,
    DATASET_SHA256,
    RESEARCH_JUDGE_TEMPLATE,
    append_jsonl,
    content_hash,
    fetch_dataset,
    load_dataset,
    parse_verdict,
    read_jsonl,
    select_subset,
    sha256_bytes,
    validate_dataset,
)
from .providers import call_model, catalog_check, effective_settings


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def cost_usd(model: dict[str, Any], usage: dict[str, Any]) -> float:
    return (
        float(usage.get("input_tokens", 0)) * float(model["price_per_million"]["input"])
        + float(usage.get("output_tokens", 0)) * float(model["price_per_million"]["output"])
    ) / 1_000_000


def model_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {model["label"]: model for model in config["models"]}


def command_fetch(args: argparse.Namespace) -> int:
    fetch_dataset(args.data)
    print(f"fetched pinned dataset to {args.data}")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    result = validate_dataset(load_dataset(args.data))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def command_plan(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    rows = load_dataset(args.data)
    validation = validate_dataset(rows)
    subset_name = args.subset or str(config.get("subset", "research-full"))
    subset = select_subset(rows, subset_name)
    trials = args.trials if args.trials is not None else int(config.get("trials", 1))
    if trials < 1:
        raise ValueError("trials must be positive")
    model_count = len(config["models"])
    answer_calls = len(subset) * model_count * trials
    plan = {
        "dry_run": True,
        "network_calls_made": 0,
        "subset": subset_name,
        "tasks": len(subset),
        "subjects": dict(sorted(Counter(item["subject"] for item in subset).items())),
        "unique_sample_ids": len({item["sample_id"] for item in subset}),
        "canonical_unique_payloads": validation["unique_payloads"],
        "models": model_count,
        "trials_per_task": trials,
        "target_generation_calls": answer_calls,
        "judge_calls": answer_calls,
        "total_api_calls": answer_calls * 2,
    }
    estimates = config.get("pilot_observed_estimates", {})
    if estimates:
        per_cell_cost = sum(float(value) for value in estimates["total_cost_per_cell_usd"].values())
        generation_seconds = sum(float(value) for value in estimates["generation_mean_seconds"].values())
        judge_seconds = float(estimates["judge_mean_seconds"]) * model_count
        task_trials = len(subset) * trials
        plan["pilot_extrapolation"] = {
            "recorded_cost_usd": round(per_cell_cost * task_trials, 2),
            "serial_wall_hours": round((generation_seconds + judge_seconds) * task_trials / 3600, 1),
            "artifact_gb": round(
                float(estimates["artifact_bytes_per_cell"]) * answer_calls / 1_000_000_000,
                2,
            ),
            "warning": "straight-line pilot extrapolation; excludes retries, queueing, price drift, and missing telemetry",
        }
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


def command_catalog(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    failed = False
    for model in [*config["models"], config["judge"]]:
        try:
            result = catalog_check(model)
            failed |= not result["available"]
            safe = {key: value for key, value in result.items() if key != "endpoint_metadata"}
            print(f"{model['label']}: {json.dumps(safe, sort_keys=True)}")
        except Exception as exc:
            failed = True
            print(f"{model['label']}: catalog error ({type(exc).__name__})", file=sys.stderr)
    return 1 if failed else 0


def write_manifest(
    run_dir: Path,
    config: dict[str, Any],
    subset_name: str,
    subset: list[dict[str, Any]],
    trials: int,
) -> None:
    manifest = {
        "condition_id": "frontierscience-research-direct-v0",
        "created_at": utc_now(),
        "dataset": {
            "repository": "openai/frontierscience",
            "revision": DATASET_REVISION,
            "sha256": DATASET_SHA256,
            "track": "research",
        },
        "subset": subset_name,
        "sample_ids": [item["sample_id"] for item in subset],
        "trial_count": trials,
        "models": [
            {
                "label": model["label"],
                "provider": model["provider"],
                "model_id": model["model_id"],
                "settings": effective_settings(model["provider"], model),
            }
            for model in config["models"]
        ],
        "judge": {
            "label": config["judge"]["label"],
            "provider": config["judge"]["provider"],
            "model_id": config["judge"]["model_id"],
            "settings": effective_settings(config["judge"]["provider"], config["judge"]),
            "prompt_sha256": sha256_bytes(RESEARCH_JUDGE_TEMPLATE.encode()),
            "pass_threshold": 7.0,
        },
        "scaffold": {
            "single_turn": True,
            "tools": False,
            "browsing": False,
            "code_execution": False,
            "search_agent": False,
        },
    }
    manifest_path = run_dir / "manifest.json"
    subset_path = run_dir / "subset.jsonl"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        comparable_existing = {key: value for key, value in existing.items() if key != "created_at"}
        comparable_new = {key: value for key, value in manifest.items() if key != "created_at"}
        if comparable_existing != comparable_new:
            raise ValueError("existing run manifest does not match requested condition")
        if read_jsonl(subset_path) != subset:
            raise ValueError("existing run subset does not match requested condition")
        return
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with subset_path.open("w", encoding="utf-8") as handle:
        for item in subset:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def validate_config_against_manifest(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    expected_models = [
        {
            "label": model["label"],
            "provider": model["provider"],
            "model_id": model["model_id"],
            "settings": effective_settings(model["provider"], model),
        }
        for model in config["models"]
    ]
    expected_judge = {
        "label": config["judge"]["label"],
        "provider": config["judge"]["provider"],
        "model_id": config["judge"]["model_id"],
        "settings": effective_settings(config["judge"]["provider"], config["judge"]),
        "prompt_sha256": sha256_bytes(RESEARCH_JUDGE_TEMPLATE.encode()),
        "pass_threshold": 7.0,
    }
    if manifest.get("models") != expected_models or manifest.get("judge") != expected_judge:
        raise ValueError("config does not match the run manifest")


def import_answers(source: Path, destination: Path, allowed: set[tuple[str, str, int]]) -> None:
    existing = {
        (row["sample_id"], row["model_label"], int(row.get("trial", 0)))
        for row in read_jsonl(destination)
    }
    for row in read_jsonl(source / "answers.jsonl"):
        key = (row.get("sample_id"), row.get("model_label"), int(row.get("trial", 0)))
        if key in allowed and key not in existing and row.get("status") == "completed":
            copied = dict(row)
            copied["imported_from"] = str(source)
            append_jsonl(destination, copied)
            existing.add(key)


def command_generate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    rows = load_dataset(args.data)
    validate_dataset(rows)
    subset = select_subset(rows, args.subset)
    trials = args.trials if args.trials is not None else int(config.get("trials", 1))
    if trials < 1:
        raise ValueError("trials must be positive")
    if args.subset == "research-full" and not args.allow_full_run:
        raise ValueError("research-full generation requires explicit --allow-full-run")
    if args.subset == "research-full" and args.resume_from:
        raise ValueError("research-full does not import answers from another run")
    run_dir = args.artifact_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(run_dir, config, args.subset, subset, trials)
    answers_path = run_dir / "answers.jsonl"
    allowed = {
        (item["sample_id"], model["label"], trial)
        for item in subset
        for model in config["models"]
        for trial in range(trials)
    }
    if args.resume_from:
        import_answers(args.resume_from, answers_path, allowed)
    completed = {
        (row["sample_id"], row["model_label"], int(row.get("trial", 0)))
        for row in read_jsonl(answers_path)
        if row.get("status") == "completed"
    }
    failures = 0
    for sample in subset:
        for model in config["models"]:
            for trial in range(trials):
                key = (sample["sample_id"], model["label"], trial)
                if key in completed:
                    print(f"skip completed {model['label']} {sample['sample_id'][:12]} trial={trial}")
                    continue
                started = time.monotonic()
                print(f"generate {model['label']} {sample['sample_id'][:12]} trial={trial}", flush=True)
                try:
                    result = call_model(model, sample["problem"])
                    elapsed = time.monotonic() - started
                    raw_suffix = sha256_bytes(str(result.get("response_id") or utc_now()).encode())[:12]
                    raw_path = run_dir / "raw" / (
                        f"generation-{sample['sample_id'][:12]}-{model['label']}-t{trial:02d}-{raw_suffix}.json"
                    )
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    raw_path.write_text(json.dumps(result.pop("raw"), ensure_ascii=False, indent=2))
                    finish_reason = str(result["finish_reason"]).lower()
                    refusal = finish_reason == "refusal"
                    incomplete = not result["text"] and finish_reason in {"max_tokens", "length"}
                    if not result["text"] and not (refusal or incomplete):
                        raise ValueError(
                            "provider returned an empty visible answer "
                            f"(finish_reason={result['finish_reason']}, usage={result['usage']})"
                        )
                    record = {
                        "answer_id": sha256_bytes(
                            "\0".join([sample["sample_id"], model["label"], str(trial)]).encode()
                        ),
                        "sample_id": sample["sample_id"],
                        "task_group_id": sample["task_group_id"],
                        "subject": sample["subject"],
                        "stratum": sample["stratum"],
                        "model_label": model["label"],
                        "requested_model": model["model_id"],
                        "resolved_model": result["resolved_model"],
                        "resolved_provider": result["provider"],
                        "effective_settings": effective_settings(model["provider"], model),
                        "trial": trial,
                        "answer": result["text"],
                        "finish_reason": result["finish_reason"],
                        "refusal": refusal,
                        "incomplete": incomplete,
                        "usage": result["usage"],
                        "cost_usd": cost_usd(model, result["usage"]),
                        "latency_seconds": round(elapsed, 3),
                        "response_id": result["response_id"],
                        "status": "completed",
                        "created_at": utc_now(),
                        "raw_path": str(raw_path.relative_to(run_dir)),
                    }
                    append_jsonl(answers_path, record)
                    completed.add(key)
                except Exception as exc:
                    failures += 1
                    append_jsonl(
                        run_dir / "generation_errors.jsonl",
                        {
                            "sample_id": sample["sample_id"],
                            "model_label": model["label"],
                            "trial": trial,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:2000],
                            "created_at": utc_now(),
                        },
                    )
                    print(
                        f"generation failed: {model['label']} trial={trial} ({type(exc).__name__})",
                        file=sys.stderr,
                    )
    return 1 if failures else 0


def command_grade(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    judge = config["judge"]
    run_dir = args.artifact_root / args.run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["subset"] == "research-full" and not args.allow_full_run:
        raise ValueError("research-full judging requires explicit --allow-full-run")
    validate_config_against_manifest(config, manifest)
    subset = {item["sample_id"]: item for item in read_jsonl(run_dir / "subset.jsonl")}
    answers = [row for row in read_jsonl(run_dir / "answers.jsonl") if row.get("status") == "completed"]
    judgments_path = run_dir / "judgments.jsonl"
    completed = {row["answer_id"] for row in read_jsonl(judgments_path) if row.get("status") == "completed"}
    failures = 0
    for answer in answers:
        if answer["answer_id"] in completed:
            continue
        sample = subset[answer["sample_id"]]
        prompt = RESEARCH_JUDGE_TEMPLATE.format(
            problem=sample["problem"], rubric=sample["rubric"], answer=answer["answer"]
        )
        print(f"judge {answer['model_label']} {answer['sample_id'][:12]}", flush=True)
        started = time.monotonic()
        try:
            result = call_model(judge, prompt)
            elapsed = time.monotonic() - started
            raw_suffix = sha256_bytes(str(result.get("response_id") or utc_now()).encode())[:12]
            raw_path = run_dir / "raw" / f"judge-{answer['answer_id'][:16]}-{raw_suffix}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(json.dumps(result.pop("raw"), ensure_ascii=False, indent=2))
            score = parse_verdict(result["text"])
            append_jsonl(
                judgments_path,
                {
                    "answer_id": answer["answer_id"],
                    "sample_id": answer["sample_id"],
                    "model_label": answer["model_label"],
                    "judge_model": judge["model_id"],
                    "resolved_judge_model": result["resolved_model"],
                    "requested_reasoning": "high",
                    "rubric_points": score,
                    "pass": score >= 7.0,
                    "threshold": 7.0,
                    "judge_output": result["text"],
                    "finish_reason": result["finish_reason"],
                    "usage": result["usage"],
                    "cost_usd": cost_usd(judge, result["usage"]),
                    "latency_seconds": round(elapsed, 3),
                    "response_id": result["response_id"],
                    "status": "completed",
                    "created_at": utc_now(),
                    "raw_path": str(raw_path.relative_to(run_dir)),
                },
            )
            completed.add(answer["answer_id"])
        except Exception as exc:
            failures += 1
            append_jsonl(
                run_dir / "judge_errors.jsonl",
                {
                    "answer_id": answer["answer_id"],
                    "model_label": answer["model_label"],
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                    "created_at": utc_now(),
                },
            )
            print(f"judge failed: {answer['model_label']} ({type(exc).__name__})", file=sys.stderr)
    return 1 if failures else 0


def run_completeness(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    validate_config_against_manifest(config, manifest)
    subset = read_jsonl(run_dir / "subset.jsonl")
    trials = int(manifest["trial_count"])
    expected_keys = {
        (sample["sample_id"], model["label"], trial)
        for sample in subset
        for model in config["models"]
        for trial in range(trials)
    }
    answers = [row for row in read_jsonl(run_dir / "answers.jsonl") if row.get("status") == "completed"]
    answer_keys = [
        (row["sample_id"], row["model_label"], int(row.get("trial", 0)))
        for row in answers
    ]
    if len(answer_keys) != len(set(answer_keys)):
        raise ValueError("run contains duplicate authoritative answer cells")
    extra_answer_keys = set(answer_keys) - expected_keys
    if extra_answer_keys:
        raise ValueError("run contains answer cells outside the manifest condition")
    answer_ids = [row["answer_id"] for row in answers]
    if len(answer_ids) != len(set(answer_ids)):
        raise ValueError("run contains duplicate authoritative answer IDs")
    judgments = [
        row for row in read_jsonl(run_dir / "judgments.jsonl") if row.get("status") == "completed"
    ]
    judgment_ids = [row["answer_id"] for row in judgments]
    if len(judgment_ids) != len(set(judgment_ids)):
        raise ValueError("run contains duplicate authoritative judgments")
    if set(judgment_ids) - set(answer_ids):
        raise ValueError("run contains judgments without matching authoritative answers")
    return {
        "complete": set(answer_keys) == expected_keys and set(judgment_ids) == set(answer_ids),
        "expected_cells": len(expected_keys),
        "answer_cells": len(answer_keys),
        "judgment_cells": len(judgment_ids),
    }


def build_summary(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    completeness = run_completeness(run_dir, config)
    answers = [row for row in read_jsonl(run_dir / "answers.jsonl") if row.get("status") == "completed"]
    judgments = [row for row in read_jsonl(run_dir / "judgments.jsonl") if row.get("status") == "completed"]
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judgments:
        by_model[row["model_label"]].append(row)
    models = {}
    for label, rows in sorted(by_model.items()):
        passed = sum(bool(row["pass"]) for row in rows)
        models[label] = {
            "n": len(rows),
            "passed": passed,
            "pass_rate": passed / len(rows) if rows else None,
            "mean_rubric_points": sum(row["rubric_points"] for row in rows) / len(rows),
            "judge_cost_usd": sum(row.get("cost_usd", 0) for row in rows),
            "generation_cost_usd": sum(
                row.get("cost_usd", 0) for row in answers if row["model_label"] == label
            ),
        }
    is_full_protocol = (
        completeness["complete"]
        and manifest["subset"] == "research-full"
        and int(manifest["trial_count"]) == 30
    )
    summary = {
        "label": (
            "public gold set with 30 trials per question; locally specified provider/output-cap condition"
            if is_full_protocol
            else "directional subset or reduced-trial run; not comparable to the official 30-trial protocol"
        ),
        "condition": "public FrontierScience-Research; direct single-turn; no browsing/tools",
        "answer_count": len(answers),
        "judgment_count": len(judgments),
        "models": models,
        "total_recorded_cost_usd": sum(row.get("cost_usd", 0) for row in answers + judgments),
        "completeness": completeness,
        "generated_at": utc_now(),
    }
    return summary


def command_summarize(args: argparse.Namespace) -> int:
    run_dir = args.artifact_root / args.run_id
    config = load_config(args.config)
    summary = build_summary(run_dir, config)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (run_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "passed", "n", "pass_rate", "mean_rubric_points", "generation_cost_usd", "judge_cost_usd"])
        for label, row in summary["models"].items():
            writer.writerow([label, row["passed"], row["n"], row["pass_rate"], row["mean_rubric_points"], row["generation_cost_usd"], row["judge_cost_usd"]])
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# {args.run_id} report",
            "",
            f"**{summary['label']}.**",
            "",
            f"Condition: {summary['condition']}.",
            "",
            "| Model | Pass | Mean rubric points | Generation cost | Judge cost |",
            "|---|---:|---:|---:|---:|",
        ]
        for label, row in summary["models"].items():
            lines.append(
                f"| {label} | {row['passed']}/{row['n']} | {row['mean_rubric_points']:.2f}/10 | "
                f"${row['generation_cost_usd']:.4f} | ${row['judge_cost_usd']:.4f} |"
            )
        lines.extend(["", f"Total recorded API cost: ${summary['total_recorded_cost_usd']:.4f}.", ""])
        args.report.write_text("\n".join(lines))
    return 0 if summary["completeness"]["complete"] else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    common_data = argparse.ArgumentParser(add_help=False)
    common_data.add_argument("--data", type=Path, default=Path(os.environ.get("FS_DATA_PATH", "data/research-test.jsonl")))

    fetch = sub.add_parser("fetch-data", parents=[common_data])
    fetch.set_defaults(func=command_fetch)
    validate = sub.add_parser("validate-data", parents=[common_data])
    validate.set_defaults(func=command_validate)
    plan = sub.add_parser("plan-run", parents=[common_data])
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--subset", choices=["smoke", "research-pilot-v1", "research-full"])
    plan.add_argument("--trials", type=int)
    plan.set_defaults(func=command_plan)
    catalog = sub.add_parser("catalog")
    catalog.add_argument("--config", type=Path, required=True)
    catalog.add_argument("--no-inference", action="store_true")
    catalog.set_defaults(func=command_catalog)

    for name, func in [("generate", command_generate), ("grade", command_grade), ("summarize", command_summarize)]:
        command = sub.add_parser(name, parents=[common_data] if name == "generate" else [])
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--artifact-root", type=Path, default=Path(os.environ.get("FS_ARTIFACT_ROOT", "/work/artifacts")))
        command.add_argument("--run-id", required=True)
        command.set_defaults(func=func)
        if name == "generate":
            command.add_argument(
                "--subset",
                required=True,
                choices=["smoke", "research-pilot-v1", "research-full"],
            )
            command.add_argument("--trials", type=int)
            command.add_argument("--resume-from", type=Path)
            command.add_argument("--allow-full-run", action="store_true")
        elif name == "grade":
            command.add_argument("--resume", action="store_true")
            command.add_argument("--allow-full-run", action="store_true")
        else:
            command.add_argument("--report", type=Path)
    return root


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
