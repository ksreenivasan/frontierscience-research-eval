from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

DATASET_REVISION = "25ed67db7da8f4591484e764008ff585544f5a30"
DATASET_SHA256 = "96c0434abfcbadd6ef6f59a03cc374be4caf9c1f2d5e62d8fe921e768f66aa46"
DATASET_URL = (
    "https://huggingface.co/datasets/openai/frontierscience/resolve/"
    f"{DATASET_REVISION}/research/test.jsonl?download=true"
)
PILOT_GROUPS = {
    "70c1246f-0a17-4f59-84a2-3699aa395d4d": ("physics", "low", 15),
    "07fd78cd-81de-4a62-9952-369093ec303f": ("physics", "medium", 8),
    "222e24bf-14d4-4660-a5dd-01b2b3528cfa": ("physics", "high", 0),
    "f8b3f2c7-7747-42d9-85bb-f8b370cc50e9": ("chemistry", "low", 20),
    "431c9b49-2795-4750-ae7b-ae9f1450cb4e": ("chemistry", "medium", 24),
    "e2aee1ae-6baa-4d30-a409-e0fc9c90f1fb": ("chemistry", "high", 21),
    "676e50c1-cea4-4361-81e1-d4409a23422c": ("biology", "low", 51),
    "a466aab3-82ae-4890-94dc-7fd11df89c0f": ("biology", "medium", 44),
    "dcfa830a-3f87-43eb-9ec7-6649fcdd0473": ("biology", "high", 46),
}
SMOKE_GROUP = "e2aee1ae-6baa-4d30-a409-e0fc9c90f1fb"

RESEARCH_JUDGE_TEMPLATE = """You are grading a science exam.
You will be given the problem, attempted answer, and a rubric to grade the answer. The rubric will total up to 10 points.
Evaluate the attempted answer against the provided rubric. Pay close attention to detail and grade it strictly, but fairly. Only evaluate against the rubric, as you yourself should not make any judgements (e.g., even if you think the answer is correct but rubric is wrong, you should treat the rubric as the gold standard). Return the absolute total number of points earned (it can be a decimal based on the rubric).
***
The problem: {problem}
***
The rubric: {rubric}
***
The attempted answer: {answer}
***
First, think step-by-step about each rubric item. Explain your reasoning for each rubric item.
Then, tally the points up and write VERDICT: <total_points> in the last line of your response, no other text. For example, VERDICT: 2.5 or VERDICT: 8.
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(row: dict[str, Any]) -> str:
    data = f"{row['task_group_id']}\0{row['problem']}\0{row['answer']}".encode()
    return sha256_bytes(data)


def fetch_dataset(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATASET_URL, timeout=60) as response:
        data = response.read()
    if sha256_bytes(data) != DATASET_SHA256:
        raise ValueError("downloaded dataset hash does not match pinned SHA-256")
    path.write_bytes(data)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = path.read_bytes()
    actual = sha256_bytes(data)
    if actual != DATASET_SHA256:
        raise ValueError(f"dataset SHA-256 mismatch: {actual}")
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    return rows


def rubric_total(rubric: str) -> float:
    values = re.findall(r"(?m)^Points:\s*([0-9]+(?:\.[0-9]+)?),\s*Item:", rubric)
    return sum(float(value) for value in values)


def validate_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = {"problem", "answer", "subject", "task_group_id"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"row {index} missing fields: {sorted(missing)}")
        total = rubric_total(row["answer"])
        if not math.isclose(total, 10.0, abs_tol=1e-8):
            raise ValueError(f"row {index} rubric totals {total}, expected 10")
    counts = Counter(row["subject"] for row in rows)
    if len(rows) != 60 or counts != Counter({"physics": 20, "chemistry": 20, "biology": 20}):
        raise ValueError(f"unexpected dataset shape: rows={len(rows)}, subjects={dict(counts)}")
    hashes = [content_hash(row) for row in rows]
    duplicates = [value for value, count in Counter(hashes).items() if count > 1]
    if len(duplicates) != 1:
        raise ValueError(f"expected one duplicate payload, found {len(duplicates)}")
    selected = select_subset(rows, "research-pilot-v1")
    if len(selected) != 9:
        raise ValueError("pilot manifest does not select 9 records")
    return {
        "rows": len(rows),
        "subjects": dict(sorted(counts.items())),
        "unique_payloads": len(set(hashes)),
        "duplicate_hashes": duplicates,
        "pilot_rows": [item["source_row"] for item in selected],
    }


def select_subset(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    by_group = {row["task_group_id"]: (index, row) for index, row in enumerate(rows)}
    groups = [SMOKE_GROUP] if name == "smoke" else list(PILOT_GROUPS)
    if name not in {"smoke", "research-pilot-v1"}:
        raise ValueError(f"unknown subset: {name}")
    output = []
    for group in groups:
        expected_subject, stratum, expected_row = PILOT_GROUPS[group]
        source_row, row = by_group[group]
        if source_row != expected_row or row["subject"] != expected_subject:
            raise ValueError(f"pinned subset mismatch for {group}")
        output.append(
            {
                "sample_id": content_hash(row),
                "source_row": source_row,
                "task_group_id": group,
                "subject": row["subject"],
                "stratum": stratum,
                "problem": row["problem"],
                "rubric": row["answer"],
            }
        )
    return output


def read_key(path: str) -> str:
    key_path = Path(path)
    if not key_path.is_file():
        raise FileNotFoundError(f"provider key file not found: {key_path}")
    key = key_path.read_text().strip()
    if not key:
        raise ValueError(f"provider key file is empty: {key_path}")
    return key


def http_json(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    timeout: int = 1800,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(2000).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from provider: {detail}") from None


def parse_verdict(text: str) -> float:
    matches = re.findall(r"(?im)^\s*VERDICT:\s*([0-9]+(?:\.[0-9]+)?)\s*$", text)
    if matches:
        score = float(matches[-1])
    elif re.fullmatch(r"\s*[0-9]+(?:\.[0-9]+)?\s*", text):
        score = float(text.strip())
    else:
        raise ValueError("judge output has no parseable VERDICT or bare numeric score")
    if not math.isfinite(score) or not 0 <= score <= 10:
        raise ValueError(f"judge score outside [0, 10]: {score}")
    return score


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
