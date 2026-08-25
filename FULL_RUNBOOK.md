# FrontierScience-Research full-run preparation

**Status:** prepared and dry-validated only. No full generation or judging has been launched.

## Canonical protocol audit

Canonical sources:

- OpenAI paper and Appendix B judge prompt: <https://arxiv.org/abs/2601.21165>
- OpenAI benchmark page: <https://openai.com/index/frontierscience/>
- Public gold dataset: <https://huggingface.co/datasets/openai/frontierscience>
- Pinned dataset revision: `25ed67db7da8f4591484e764008ff585544f5a30`
- Pinned Research JSONL SHA-256: `96c0434abfcbadd6ef6f59a03cc374be4caf9c1f2d5e62d8fe921e768f66aa46`

The official public Research gold set has 60 rows, exactly 20 each in physics, chemistry, and biology. The paper defines a self-contained, text-only question-answering evaluation. Each question has a rubric totaling 10 points. GPT-5 at high reasoning receives the problem, attempted answer, and rubric; an answer passes at `>=7/10`. Published Research results average the binary pass outcome across **30 independent answer trials per question**. The paper states that evaluated models used high reasoning (GPT-5.2 used xhigh) **without browsing**.

The paper's task-writing guidelines say the question must provide the inputs an expert needs and should test complex reasoning rather than search or recency. Web search is therefore neither required nor part of the official direct condition. A separately published AgentCompass search-agent evaluation is a different, non-comparable condition.

No official OpenAI evaluator repository was found. The canonical evaluator specification is the paper's Appendix B prompt and threshold, plus the released rubric data. The dated judge ID `gpt-5-2025-08-07` is our reproducible mapping to the paper's undated “GPT-5 at high reasoning,” not an ID printed in the paper.

The pinned JSONL has 60 rows but 59 unique payloads: source rows 6 and 11 are exact duplicates. The full selector keeps both official rows and assigns distinct row-suffixed sample IDs. A final report should show the official 60-row result and may add a 59-unique-payload sensitivity calculation without new model calls.

## Implementation lineage

There is no canonical OpenAI FrontierScience code repository or package. The canonical implementation inputs reused here are:

- OpenAI's paper/protocol and Appendix B Research judge prompt;
- the Apache-2.0 `openai/frontierscience` Hugging Face dataset at revision `25ed67db7da8f4591484e764008ff585544f5a30`; and
- the dataset's original `problem`, rubric-bearing `answer`, `subject`, and `task_group_id` fields.

During planning, community implementations were inspected read-only: `UKGovernmentBEIS/inspect_evals` at `ce9e8b3fd1793027a70260f3c2413561b60787e3`, `medicalsphere/FrontierScience` at `e29c2b6421e19127982a358a93522123afc86d31`, `EnvCommons/FrontierScience` at `fa85db6a855f72fd0429ebddfe347b50c9d4f403`, and `open-compass/AgentCompass` at `512af1cac726fb08389670d9e31a69b1524f69d4`. None is a runtime dependency and no community runner code was vendored. In particular, the local runner does not inherit Inspect Evals' normalized-mean scorer or AgentCompass's search scaffold.

The `frontierscience_eval` package was implemented locally from scratch using the Python standard library. It includes direct provider HTTP adapters, pinned-data validation, deterministic subset selection, immutable JSONL artifacts, the rubric judge/parser, cost summaries, and Docker run isolation. The local judge template is transcribed from Appendix B with the documented typo/whitespace normalization.

Pilot implementation changes were: four direct high-reasoning model payloads; a common 32,768-token cap; the deterministic 9-question subset; Alibaba-pinned OpenRouter routing; refusal/output-cap handling; preservation of raw provider and judge responses; per-answer `>=7/10` scoring; no-tools container isolation; and sanitized smoke/pilot reports.

Full-run preparation adds: the complete 60-row selector with unique IDs for the duplicate payload; 30 independent trial indices and trial-specific answer IDs; condition-safe resumability; explicit generation and judging launch guards; strict manifest/config and expected-cell completeness checks; `configs/direct-full.json`; a network-free `plan-run`; and this runbook. No pilot artifact was rewritten.

## Environment today

### What the evaluated model receives

For every answer trial, the target provider receives exactly one user message containing the canonical `problem` string.

The target model does **not** receive:

- the rubric or reference material in the dataset's `answer` field;
- the subject, stratum, task-group ID, source row, or other metadata;
- previous answers or judge feedback;
- a system prompt, retrieval results, files, images, or follow-up messages;
- tool definitions of any kind.

The providers receive high-reasoning settings and a 32,768-total-output-token campaign cap. Qwen additionally receives its documented thinking sampling settings and a provider-routing block pinned to Alibaba with routing fallbacks disabled.

### Web and network access

The evaluated model has **no web-search or visit tool**. The local run container has ordinary bridged egress only so the runner can call the selected model API. Network transport from runner to provider is a harness necessity; it does not give the model an interactive browser or URL-fetch function. Offline validation and reporting run with `--network none`.

Web search must remain disabled for an official-direct comparable lane. Enabling it would create a separate agent/scaffold condition and invalidate direct comparison with the paper's table.

### Code execution

No code-execution tool or scientific environment is exposed to the model. Model output is treated as inert text: it is never imported, interpreted, shelled, or executed. FrontierScience-Research does not require code execution; the paper describes constrained, text-only research subproblems. A model may include code snippets in its prose, but the harness only stores and grades that text.

### Judging

After answer generation, the judge receives one user message containing:

1. the canonical problem;
2. the complete 10-point rubric;
3. that single attempted answer; and
4. the local Appendix B Research judge template.

The local template is semantically equivalent but not byte-identical to the PDF typography: it corrects the paper's `attemped` typo and normalizes whitespace. This is the template used for the smoke/pilot and is retained for condition continuity. The configured judge is `gpt-5-2025-08-07` at high reasoning with a 32,768-token output cap and no tools. One judge call scores each answer. The parser accepts the requested final `VERDICT: <points>` form or an otherwise unambiguous response consisting only of a number. Scores must be finite and in `[0,10]`; API and parse failures are not converted to zero. The primary metric is the mean of per-answer `score >= 7` indicators, not a threshold applied after averaging rubric points.

`gpt-5-2025-08-07` is the pinned primary adjudication condition. Alternative judge models may be run against frozen answer artifacts to measure judge dependence, but must use separate condition IDs and record exact model/provider, reasoning settings, output cap, prompt hash, parser, threshold, retry policy, timestamp, and answer IDs. Alternative scores must be compared with the primary judge and never silently substituted, pooled, averaged, or voted into the primary result.

For a small sensitivity check, predeclare 12–24 existing answers balanced across target models and subjects, with coverage across primary-score bands below 5, 5 to below 7, and at/above 7. Regrade each frozen answer once per alternative judge. Report mean absolute point difference, pass/fail disagreement, a 2×2 pass confusion table, and threshold-adjacent disagreements overall and by target model. Expert review may adjudicate disagreements as a separate analysis; it must not rewrite the pinned primary artifacts. No generalized multi-judge framework is required.

### Isolation and resources

The reusable image is a small `python:3.12-slim` standard-library runner, executing as numeric user `65534`. The runbook applies the controls at `docker run` time:

- read-only container root filesystem;
- all Linux capabilities dropped and `no-new-privileges`;
- no privileged mode, Docker socket, host networking, or broad writable host mount;
- 1 CPU, 512 MiB memory, 128 PID limit, and a 64 MiB no-exec tmpfs;
- dataset and the four explicitly authorized `~/secrets_and_keys/*.key` files mounted read-only;
- only the scoped run-artifact directory writable;
- raw answers, rubrics, judge explanations, and provider responses kept under ignored `artifacts/`;
- secrets read into request memory only and never placed in manifests or reports.

These limits protect the host and make the local runner reproducible. They do not constrain remote provider inference compute. Because no generated code executes, a scientific sandbox, GPU, package manager, or larger local resource allocation is unnecessary.

## Benchmark requirements versus harness choices

| Item | Benchmark requirement / published condition | Local harness choice |
|---|---|---|
| Dataset | Public 60-question Research gold set; 20/subject | Pinned revision and SHA; both duplicate rows retained with unique internal IDs |
| Prompt | Self-contained text problem | Verbatim problem as one user message; no system prompt |
| Browsing | Published evaluations ran without browsing | No tools or web interface; provider API transport only |
| Code | Not required; text-only benchmark | No execution path; output remains inert text |
| Target effort | High for paper reasoning models, except GPT-5.2 xhigh | High/native thinking for all four requested lanes |
| Repetitions | 30 independent Research answer trials per question | `configs/direct-full.json` sets 30; generation requires explicit launch flag |
| Judge | GPT-5 high with Appendix B prompt | Dated snapshot `gpt-5-2025-08-07`, high; snapshot mapping inferred; typo/whitespace-normalized local template retained from pilot |
| Success | Individual answer earns at least 7/10 | Same binary threshold; rubric mean retained only as a diagnostic |
| Target output limit | Not published | 32,768 total tokens for each target lane |
| Judge output limit | Not published | 32,768 tokens |
| Provider routing | Not specified | Native APIs for closed models; Qwen pinned to Alibaba through OpenRouter |
| Retries | Not specified | Manual/resumable transport retries; valid refusals/truncations remain outcomes |
| Container | Not part of benchmark | Local isolation and artifact hygiene only |

## Prepared configuration

`configs/direct-full.json` specifies:

- `subset=research-full`;
- `trials=30`;
- four target models and the same high-reasoning payloads as the pilot;
- the same judge and threshold;
- the existing 32,768-token cap;
- a required explicit `--allow-full-run` launch flag; and
- pilot observations used only for dry-run extrapolation.

The runner now supports multiple trial indices, unique answer IDs per trial, resumable completed-cell detection, all 60 official rows, and distinct internal IDs for the duplicate payload. Pilot artifacts are untouched and full runs are forbidden from importing answers from another run, preventing condition mixing.

## No-cost offline validation

Build without pulling and run with no network and no secrets:

```bash
docker build -t frontierscience-eval:dev .

docker run --rm --name frontierscience-eval-full-tests \
  --read-only --network none --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 128 --memory 512m --cpus 1 \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  -v "$PWD/data:/work/data:ro" \
  frontierscience-eval:dev \
  python -m unittest discover -s tests -v

docker run --rm --name frontierscience-eval-full-plan \
  --read-only --network none --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 128 --memory 512m --cpus 1 \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  -v "$PWD/data:/work/data:ro" \
  frontierscience-eval:dev \
  python -m frontierscience_eval plan-run \
    --config /work/configs/direct-full.json \
    --data /work/data/research-test.jsonl
```

`plan-run` performs no network request and reports 60 tasks, 4 models, 30 trials, 7,200 target calls, 7,200 judge calls, and 14,400 total API calls.

An optional authenticated catalog check also performs no inference, but does contact provider APIs and is intentionally separate from the offline plan.

## Launch runbook — not authorized yet

Create a new scoped artifact directory; do not reuse or rewrite `artifacts/pilot`:

```bash
mkdir -p artifacts/full-30trial
chmod 0700 artifacts/full-30trial
```

Generation is resumable. The command below is documented but must not be invoked until launch is explicitly approved:

```bash
docker run --rm --name frontierscience-eval-full-generate \
  --read-only --network bridge --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 128 --memory 512m --cpus 1 \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  -v "$HOME/secrets_and_keys/openai.key:/run/secrets/openai:ro" \
  -v "$HOME/secrets_and_keys/anthropic.key:/run/secrets/anthropic:ro" \
  -v "$HOME/secrets_and_keys/gemini.key:/run/secrets/gemini:ro" \
  -v "$HOME/secrets_and_keys/openrouter.key:/run/secrets/openrouter:ro" \
  -v "$PWD/data:/work/data:ro" \
  -v "$PWD/artifacts/full-30trial:/work/artifacts" \
  frontierscience-eval:dev \
  python -m frontierscience_eval generate \
    --config /work/configs/direct-full.json \
    --data /work/data/research-test.jsonl \
    --artifact-root /work/artifacts --run-id full-30trial \
    --subset research-full --trials 30 --allow-full-run
```

Re-running the identical command skips authoritative completed `(sample, model, trial)` cells. Transport failures remain in error JSONL and must be reviewed before retrying; valid refusals, visible truncations, and no-visible-answer output-cap exhaustion are completed model outcomes, not retry candidates.

After generation completeness is verified, judge with only the OpenAI key mounted:

```bash
docker run --rm --name frontierscience-eval-full-judge \
  --read-only --network bridge --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 128 --memory 512m --cpus 1 \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  -v "$HOME/secrets_and_keys/openai.key:/run/secrets/openai:ro" \
  -v "$PWD/artifacts/full-30trial:/work/artifacts" \
  frontierscience-eval:dev \
  python -m frontierscience_eval grade \
    --config /work/configs/direct-full.json \
    --artifact-root /work/artifacts --run-id full-30trial \
    --resume --allow-full-run
```

Summarize offline with `--network none`. Commit only a sanitized report/manifest, never the run directory.

The current implementation is deliberately sequential and append-only. Do not point multiple containers at the same writable JSONL directory. Parallel execution would require a separate, explicitly reviewed sharding/merge change rather than an ad hoc concurrent launch.

## Scope, cost, runtime, and storage

Observed pilot means were used as straight-line estimates:

- target latency: GPT-5.6 Sol 199.1 s, Claude Opus 5 156.0 s (one missing-latency terminal response), Gemini Flash 36.0 s, Qwen 148.2 s;
- judge latency: 92.2 s per answer;
- authoritative cost: $9.0862 for 36 model/judge cells;
- pilot artifact footprint: 3.45 MB for 36 cells.

| Scope | Target calls | Judge calls | Total calls | Recorded-cost extrapolation | Sequential wall-time extrapolation | Artifact extrapolation |
|---|---:|---:|---:|---:|---:|---:|
| Full public set, 1 trial | 240 | 240 | 480 | ~$60.57 | ~15.1 h | ~0.02 GB |
| Published Research repetition count, 30 trials | 7,200 | 7,200 | 14,400 | ~$1,817 | ~454 h / 18.9 d | ~0.69 GB |

These are not quotes or guarantees. They exclude transport retries, queueing, rate-limit backoff, unknown billed malformed responses, future price changes, and the pilot's missing latency telemetry. The current sequential runner is correct and resumable but slow; concurrency is a separate engineering/operational decision.

## Required pre-launch decision

The target output cap is not specified by the paper, and the current 32,768-token cap materially affected Claude in the pilot: two of nine Claude trials hit the cap, including one that used all output tokens for thinking and produced no visible answer.

Choose one condition before launch:

1. **Keep 32,768 tokens** to preserve condition continuity with the pilot. The prepared full run still starts in a fresh directory and does not import pilot cells. Cap exhaustion remains a scored model outcome.
2. **Raise all target lanes to 65,536 tokens** to reduce truncation under a portable common cap. This is a new local condition, costs may rise materially, and pilot cells must not be reused.

Recommendation: if the objective is the strongest capability estimate rather than continuity with the pilot, use a fresh 65,536-token condition. Regardless of cap, retain the 32,768 judge cap because no pilot judgment truncated. Also decide whether the immediate expansion means the full 60 questions at one trial (~$61, ~15 serial hours) or the paper's 30-trial repetition count (~$1.8K, ~19 serial days). Do not call a 60×1 result equivalent to the published protocol.
