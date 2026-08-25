# FrontierScience-Research evaluation plan

## 1. Objective and execution posture

Implement a reproducible evaluation of the **public 60-question FrontierScience-Research gold set**, beginning with:

1. a **one-question, end-to-end plumbing smoke test**;
2. a **9-question stratified pilot** (15% of the public Research set; optionally extend to 12/60 only after review); and
3. no full evaluation until the smoke and pilot expose and resolve adapter, scoring, cost, and reporting problems.

The default lane will be the simplest lane comparable in spirit to OpenAI's original evaluation:

- direct, single-turn text response;
- no browsing, search, code execution, retrieval, tools, files, or agent loop;
- one sampled answer per task in the smoke/pilot;
- the published FrontierScience-Research rubric prompt;
- GPT-5 at high reasoning effort as judge when the original snapshot remains available;
- pass for an individual answer when its rubric score is **at least 7/10**.

This pilot is **not** an official-score reproduction. OpenAI reports Research results from **30 independent answer generations per question**. The pilot's one sample per model/question is a plumbing and directional comparison only.

### Authorized execution overrides (2026-08-25)

- Execute through the clean 9-question × four-model bounded pilot, but not a full benchmark run.
- Use **high** reasoning/thinking for GPT-5.6 Sol, Claude Opus 5, Gemini 3.7 Flash, and Qwen3.5, and log the exact effective settings.
- Record and report spend, but do **not** stop generation or judging based on a dollar budget gate.
- Prefer the smallest correct implementation over defensive frameworks or infrastructure. Local hackiness is acceptable when it does not change benchmark semantics, lose raw artifacts, or expose secrets.
- Use a lightweight localized container and direct provider APIs; no agent framework.
- Make local Git commits at meaningful milestones: approved plan, offline scaffold/correctness checks, smoke report, and pilot report. Never push without separate authorization.
- Keep raw, large, or protected run artifacts outside Git. Commit only code, secret-free manifests, small fixtures, and sanitized reports.

## 2. Canonical-source assessment

### 2.1 Official sources

- Paper: [FrontierScience: Evaluating AI's Ability to Perform Expert-Level Scientific Tasks](https://arxiv.org/abs/2601.21165), especially the evaluation protocol and Appendix B judge prompt.
- OpenAI page: [Evaluating AI's ability to perform scientific research tasks](https://openai.com/index/frontierscience/).
- Public gold data: [`openai/frontierscience`](https://huggingface.co/datasets/openai/frontierscience) on Hugging Face.
- Dataset revision inspected: `25ed67db7da8f4591484e764008ff585544f5a30`.

The official public release contains:

- `research/test.jsonl`: 60 Research records, exactly 20 each in physics, chemistry, and biology;
- `olympiad/test.jsonl`: 100 Olympiad records, out of scope for the first implementation;
- fields `problem`, `answer`, `subject`, and `task_group_id`; for Research, `answer` contains the complete 10-point rubric.

The official protocol is clear enough to implement the direct lane:

- the target model receives the self-contained problem without browsing;
- the Research judge receives the problem, attempted answer, and rubric;
- the published prompt tells the judge to treat the rubric as the sole gold standard, grade each item, total the points, and end with `VERDICT: <points>`;
- one answer passes at `score >= 7.0`;
- official aggregate results use 30 independently generated answers per model/question.

No official OpenAI FrontierScience GitHub evaluator was found. Therefore, the **paper prompt and dataset**, not a community runner, are canonical.

### 2.2 Data-quality observation

The inspected Research JSONL has 60 rows but only 59 unique `(task_group_id, problem, answer)` payloads. Physics rows 6 and 11 (zero-based) are exact duplicates with task group `af50243e-3a60-4460-9536-f9a02c4f8eb8`.

Plan:

- exclude the second copy from smoke/pilot selection;
- retain the official 60-row definition if a later official-protocol run is approved, but report both the official 60-row metric and a 59-unique-record sensitivity result;
- open an upstream dataset issue only after user approval; do not silently alter the canonical data.

### 2.3 Community implementation assessment

Repositories were inspected read-only at the following revisions:

| Source | Revision | Assessment |
|---|---|---|
| `UKGovernmentBEIS/inspect_evals` FrontierScience task | `ce9e8b3fd1793027a70260f3c2413561b60787e3` | Best starting chassis: MIT-licensed, pins the official dataset revision, supports subject/track filtering, and carries the paper judge prompt. Its Research scorer currently returns normalized rubric score and reports `mean`; it does **not** implement the official binary `>=7/10` pass-rate metric as the primary score. Its documented 27.3% "mean" comparison with OpenAI's 25.2% pass rate conflates metrics. Use the loader/model/logging infrastructure, but locally replace or wrap the scorer and aggregation. |
| `medicalsphere/FrontierScience` | `e29c2b6421e19127982a358a93522123afc86d31` | Useful reference only. It uses LiteLLM with silent parameter dropping and thresholds the **average rubric score across trials**, rather than averaging per-trial binary passes. No license file was present in the inspected checkout, so do not copy code. |
| `EnvCommons/FrontierScience` | `fa85db6a855f72fd0429ebddfe347b50c9d4f403` | Useful defensive-grading reference: fail-open errors are avoided and criterion results are retained. It changes the paper protocol by judging criteria in separate calls with GPT-5.2, and its README's track counts disagree with the canonical 100/60 split. Do not use for the comparable lane. |
| `open-compass/AgentCompass` | `512af1cac726fb08389670d9e31a69b1524f69d4` | Relevant only to a later search-agent lane. It uses a different prompt/scorer, search/visit tools, a different judge, and three-run aggregation. Do not mix its scores or scaffold with the direct lane. |

### 2.4 Repository and upstream strategy

Implement in this repository as a thin, benchmark-specific package rather than creating a remote fork.

Proposed layout:

```text
.
├── PLAN.md
├── README.md
├── Dockerfile
├── configs/
│   ├── direct-smoke.json
│   └── direct-pilot.json
├── src/frontierscience_eval/
│   ├── adapters/
│   ├── dataset.py
│   ├── generate.py
│   ├── judge.py
│   ├── report.py
│   └── schemas.py
├── subsets/
│   └── research-pilot-v1.jsonl
└── tests/
```

Do not use Inspect AI, Inspect Evals, LiteLLM, or another agent/evaluation framework for this bounded campaign. Implement a small standard-library CLI with direct HTTP calls to the four provider APIs, a pinned local copy of the public Research JSONL, the published judge prompt, deterministic subset manifests, and simple JSONL artifacts. Add only dependencies strictly needed for correct HTTP behavior.

After the pilot, consider a narrowly scoped upstream contribution to `inspect_evals` that adds the official Research pass-rate metric while retaining rubric mean as a diagnostic. A remote fork, issue, or pull request requires separate user approval and is not part of initial implementation.

## 3. License and access gates

| Gate | Status / required action |
|---|---|
| Dataset license | Hugging Face card declares Apache-2.0. Preserve notices and the dataset's explicit instruction that benchmark data must not enter training corpora. |
| Dataset access | Public and ungated at the inspected revision. Pin by commit and verify file hashes before each campaign. |
| Paper/page | Public. Copy the judge template with source attribution and record the appendix/version. |
| `inspect_evals` code | MIT-licensed. Safe to depend on or adapt with notice. |
| Community runners | Do not copy code from sources lacking a clear license. |
| API access | Before inference, verify account entitlement and billing for OpenAI, Anthropic, Gemini Developer API, and OpenRouter. Public catalog presence does not prove account-level access. |
| Provider terms/data handling | Confirm whether prompts/outputs may be retained or used for provider improvement. Use paid/API data-control settings appropriate for benchmark material. Do not use Gemini's free tier if its data-use terms are unsuitable. |
| Publication | Results remain local/internal by default. Any public upload, leaderboard submission, issue, or PR is a separate user decision. |

## 4. Model and provider matrix

Catalog/docs were checked on **2026-08-25 UTC** without inference.

| Label | Provider | Exact requested model ID | Live status and planned condition |
|---|---|---|---|
| GPT-5.6 Sol | OpenAI Responses API | `gpt-5.6-sol` | Listed in the official API catalog. Use native OpenAI access, `reasoning.effort=high`, no tools. Documented context 1,050,000 and max output 128,000 tokens. Current standard price: $4/M input, $20/M output. |
| Claude Opus 5 | Anthropic Messages API | `claude-opus-5` | Listed in Anthropic's official model catalog. Use native Anthropic access, adaptive thinking with `output_config.effort=high`, no tools. Documented context 1,000,000 and max synchronous output 128,000 tokens. Current standard price: $5/M input, $25/M output. |
| Gemini Flash | Gemini Developer API | `gemini-3.7-flash` | Newest stable Flash listed in Google's official catalog at check time. Use native Google access, `thinking_level=high`, no tools. Documented context 1,048,576 and max output 65,536 tokens. Current promotional standard price: $0.75/M input, $3.75/M output. Recheck immediately before execution because "newest" can change. |
| Qwen | OpenRouter | `qwen/qwen3.5-397b-a17b` | Present in the live OpenRouter models API with multiple healthy endpoints. Use native thinking, Qwen's recommended thinking sampling (`temperature=0.6`, `top_p=0.95`, `top_k=20` where the pinned endpoint supports them), and no tools. The portable advertised limit is 262,144 context / 65,536 output. |

### OpenRouter routing

Do not allow OpenRouter to move between endpoints invisibly. For Qwen:

- preflight `GET /api/v1/models/qwen/qwen3.5-397b-a17b/endpoints` without inference;
- prefer the first-party Alibaba endpoint if it remains healthy, supports every requested parameter, and the user accepts its undeclared quantization; otherwise select one endpoint explicitly;
- set `provider.only=[<selected provider>]`, `allow_fallbacks=false`, and `require_parameters=true`;
- persist OpenRouter's requested ID, canonical slug, resolved provider, quantization, context/output limits, and catalog timestamp in the run manifest.

At the catalog check, the model's canonical slug was `qwen/qwen3.5-397b-a17b-20260216`; Alibaba, DeepInfra FP8, Parasail FP8, and several other endpoints had live status.

If Qwen has no usable live OpenRouter endpoint at execution time, replace it—do not silently alias it—with the following fallback order and exact IDs:

1. `z-ai/glm-5.1`
2. `nvidia/nemotron-3-ultra-550b-a55b`

Both IDs were present in the live OpenRouter catalog. The free Nemotron variant is intentionally excluded because queueing and routing constitute a different condition. A fallback receives its own model label and cannot be pooled with or described as Qwen.

### Common generation condition

- Send the canonical `problem` verbatim as a single user message; no solution hints, rubric, reference answer, or subject-specific system prompt.
- No system prompt unless a provider technically requires one; any provider-only wrapper must be recorded byte-for-byte.
- No tools, search, URL retrieval, files, code execution, memory, follow-up turn, or answer repair.
- One answer sample per model/task in smoke and pilot.
- Set a campaign output cap of **32,768 total generated tokens**, including billed reasoning tokens where the provider defines it that way. This is below every selected model's documented hard limit and is an explicit pilot constraint because the paper did not publish its cap.
- Record visible-answer tokens, reasoning tokens when the provider reports them, total output tokens, finish reason, latency, retries, and provider request/response IDs. Do not store hidden chain-of-thought content even if a provider can return it.
- Use provider-native `high` effort where supported. For Qwen, map the requested high-thinking condition to `reasoning.enabled=true` plus the recommended thinking sampling settings because OpenRouter exposes no reliable `high` effort enum; log both the requested label and effective payload. This is a declared system-plus-provider condition, not proof of equal test-time compute.
- Do not silently drop unsupported parameters. Catalog validation or a provider error must fail the adapter preflight.

## 5. Judge and metrics

### 5.1 Primary judge

Use the dated GPT-5 snapshot `gpt-5-2025-08-07` with `reasoning.effort=high` if it remains accessible. The paper names GPT-5 at high effort but does not print a dated API ID; this snapshot is the closest reproducible mapping and is currently documented, though deprecated.

Use the Appendix B Research prompt exactly, with only deterministic insertion of:

- canonical problem;
- full rubric from the record's `answer` field;
- attempted answer.

Set a 32,768-token judge output cap. Parse the final case-insensitive `VERDICT: <number>` line, require a finite value in `[0,10]`, and store the complete visible judge explanation. Never turn API, empty-output, or parse failures into a score of zero.

If a judgment fails:

1. preserve the failed artifact;
2. retry the identical judge request only for transport/service failure, within the configured retry budget;
3. for a syntactically malformed verdict, mark `judge_parse_error` and require an explicit regrade command; do not regenerate the target answer;
4. record every attempt and which judgment is authoritative.

If `gpt-5-2025-08-07` becomes unavailable, stop the comparable lane. A newer judge can be used only as a separately named judge-sensitivity lane after the user chooses it; it must not be presented as the original judge protocol.

### 5.2 Primary and diagnostic metrics

For every answer trial:

```text
rubric_points = parsed judge score in [0, 10]
pass = 1[rubric_points >= 7.0]
```

Report:

- primary: mean of per-trial binary `pass` values;
- diagnostics: mean/median rubric points, raw pass counts, subject breakdown, model/task matrix, errors, truncations, refusal rate, token use, latency, and cost;
- no thresholding of a question's average rubric score across repeated trials;
- no mixing of direct and search-agent conditions.

The smoke/pilot has one answer trial per question. Thus a task/model cell is one observed binary outcome, not an estimate of that model's per-question success probability.

### 5.3 Judge quality checks

Before trusting pilot totals:

- unit-test score parsing using synthetic valid, decimal, malformed, out-of-range, and multiple-verdict fixtures;
- manually audit all smoke judgments and at least one judgment per model plus one per subject in the pilot;
- verify rubric item totals equal 10 for every selected task;
- compare the stored judge prompt hash against the pinned canonical template;
- optionally regrade a small, predeclared 4-answer sensitivity sample with the same judge to measure judge instability, but keep those secondary judgments out of the primary score.

## 6. Subset selection

### 6.1 Default pilot: 9/60 (15%)

Use a deterministic, domain-balanced subset that does not claim to be a statistically representative random sample:

1. pin dataset revision `25ed67db7da8f4591484e764008ff585544f5a30`;
2. de-duplicate exact payloads for pilot selection, retaining the first row;
3. within each subject, sort unique tasks by the transparent length proxy `len(problem) + len(rubric)`;
4. split each subject into low, medium, and high contiguous tertiles;
5. select one record per tertile by the minimum SHA-256 of `dataset_revision + "\0pilot-v1\0" + task_group_id`.

This produces three tasks per subject and coverage across short, medium, and long problem/rubric payloads. Length is a plumbing/complexity proxy, not a claim about scientific difficulty.

Planned manifest:

| Subject | Stratum | Source row | `task_group_id` | Content hash prefix |
|---|---|---:|---|---|
| physics | low | 15 | `70c1246f-0a17-4f59-84a2-3699aa395d4d` | `939dd1da1c59` |
| physics | medium | 8 | `07fd78cd-81de-4a62-9952-369093ec303f` | `d5f4adb0139e` |
| physics | high | 0 | `222e24bf-14d4-4660-a5dd-01b2b3528cfa` | `2ed9ab0f2e3e` |
| chemistry | low | 20 | `f8b3f2c7-7747-42d9-85bb-f8b370cc50e9` | `88c81bb8f096` |
| chemistry | medium | 24 | `431c9b49-2795-4750-ae7b-ae9f1450cb4e` | `b7681abb37b9` |
| chemistry | high | 21 | `e2aee1ae-6baa-4d30-a409-e0fc9c90f1fb` | `cb64e023ff7a` |
| biology | low | 51 | `676e50c1-cea4-4361-81e1-d4409a23422c` | `513d6cb89e35` |
| biology | medium | 44 | `a466aab3-82ae-4890-94dc-7fd11df89c0f` | `85b3b5c463cf` |
| biology | high | 46 | `dcfa830a-3f87-43eb-9ec7-6649fcdd0473` | `6e363da67c93` |

### 6.2 Smoke task

Use the chemistry-high pilot record at source row 21 (`task_group_id=e2aee1ae-6baa-4d30-a409-e0fc9c90f1fb`) as the one-question smoke. Run it once through **all four target adapters** and the common judge. This remains a one-task smoke while validating the entire matrix. Reuse successful smoke artifacts in the pilot so the smoke does not add a second sample for that question.

If the user prefers the absolute cheapest smoke, Gemini alone can run first, but the default recommendation is all four adapters because most early failures are provider-specific.

### 6.3 Optional extension

Only after reviewing the 9-task pilot, extend to 12/60 by adding one predeclared task per subject from the remaining records. Do not choose additions based on which tasks appear favorable to a model. A 12-task result remains a one-sample stratified pilot, not an official score.

## 7. Isolation and environment design

Although the direct lane never executes model-generated code, use a lightweight rootless Docker/Podman container for dependency and credential hygiene:

- pin the Python base image by digest and avoid a dependency framework when the standard library suffices;
- use explicit project/container names such as `frontierscience-eval` and remove them after each phase;
- run as a non-root user with read-only root filesystem, dropped Linux capabilities, `no-new-privileges`, bounded CPU/memory/PIDs, and a tmpfs scratch directory;
- never use privileged mode, the Docker socket, host networking, or broad writable host mounts;
- mount source/data read-only and only the specific per-run artifact directory as writable;
- pass API keys at runtime through secret injection or an excluded env file; never bake, print, or commit them, and never expose them to generated-code sandboxes;
- use `--network=none` for offline validation/reporting and ordinary bridged networking only for provider calls;
- fetch and checksum the dataset in a separate preparation step;
- do not execute, import, or shell-evaluate model output;
- keep this proportional: one image and a few explicit `docker run` commands, not a platform.

A later optional search-agent lane must use a separate hardened browser/search container, separate config namespace, and separate result tables. It must never be enabled by a flag on the direct-lane container without an explicit condition change.

## 8. Planned implementation phases and commands

This is the authorized execution sequence. Generation, grading, and reporting remain separate and resumable. Commit locally after the approved plan, offline scaffold, smoke report, and pilot report; do not push.

### Phase 0 — Build, offline checks, and catalog preflight

```bash
docker build --pull -t frontierscience-eval:dev .

docker run --rm --name frontierscience-eval-data-check --read-only --network none \
  frontierscience-eval:dev python -m frontierscience_eval validate-data

docker run --rm --name frontierscience-eval-tests --read-only --network none \
  frontierscience-eval:dev python -m unittest discover -s tests -v

docker run --rm --name frontierscience-eval-catalog --read-only --network bridge \
  --env-file .env.catalog frontierscience-eval:dev \
  python -m frontierscience_eval catalog --config configs/direct-pilot.json --no-inference
```

Gates:

- exact model IDs appear in official/native catalogs or the authenticated account list;
- an OpenRouter endpoint is explicitly pinned;
- no adapter parameter would be silently dropped;
- dataset hash, counts, subject balance, rubric totals, and duplicate report match expectations;
- synthetic scorer tests pass.

Commit the offline scaffold and correctness checks locally before inference.

### Phase 1 — One-question, all-adapter smoke

```bash
docker run --rm --name frontierscience-eval-smoke-generate --read-only --network bridge \
  --env-file .env.run -v "$PWD/data:/work/data:ro" \
  -v "$PWD/artifacts/smoke:/work/artifacts" frontierscience-eval:dev \
  python -m frontierscience_eval generate \
    --config configs/direct-smoke.json --subset smoke --trials 1

docker run --rm --name frontierscience-eval-smoke-judge --read-only --network bridge \
  --env-file .env.judge -v "$PWD/data:/work/data:ro" \
  -v "$PWD/artifacts/smoke:/work/artifacts" frontierscience-eval:dev \
  python -m frontierscience_eval grade --resume --run-id <smoke-run-id>

docker run --rm --name frontierscience-eval-smoke-report --read-only --network none \
  -v "$PWD/artifacts/smoke:/work/artifacts" frontierscience-eval:dev \
  python -m frontierscience_eval summarize --run-id <smoke-run-id>
```

Stop after smoke for a narrow artifact and judge review. Fix only issues needed for correct benchmark semantics or reliable execution. Commit the sanitized smoke report locally. Proceed to the pilot only if the smoke acceptance criteria are clean.

### Phase 2 — Nine-question pilot

```bash
docker run --rm --name frontierscience-eval-pilot-generate --read-only --network bridge \
  --env-file .env.run -v "$PWD/data:/work/data:ro" \
  -v "$PWD/artifacts/pilot:/work/artifacts" frontierscience-eval:dev \
  python -m frontierscience_eval generate \
    --config configs/direct-pilot.json --subset research-pilot-v1 \
    --trials 1 --resume-from <smoke-run-id>

docker run --rm --name frontierscience-eval-pilot-judge --read-only --network bridge \
  --env-file .env.judge -v "$PWD/data:/work/data:ro" \
  -v "$PWD/artifacts/pilot:/work/artifacts" frontierscience-eval:dev \
  python -m frontierscience_eval grade --resume --run-id <pilot-run-id>
docker run --rm --name frontierscience-eval-pilot-report --read-only --network none \
  -v "$PWD/artifacts/pilot:/work/artifacts" frontierscience-eval:dev \
  python -m frontierscience_eval summarize --run-id <pilot-run-id>
```

Use low provider-aware concurrency. Inspect errors, truncations, refusals, effective settings, resolved providers, and spend; rerun only failed/incomplete cells while preserving attempt lineage. Commit only the sanitized pilot report and secret-free manifest.

### Phase 3 — Full evaluation remains deferred

A full official-style Research campaign would require 60 questions × 30 trials × 4 models = **7,200 target generations**, plus roughly 7,200 judge calls. It is not authorized by this execution request and no full-run command will be implemented or invoked.

## 9. Artifacts and results schema

Use immutable, append-only JSONL for attempts and derived JSON/CSV for summaries:

```text
artifacts/<run_id>/
├── manifest.json
├── subset.jsonl
├── generation_attempts.jsonl
├── answers.jsonl
├── judge_attempts.jsonl
├── judgments.jsonl
├── trials.jsonl
├── summary.json
├── summary.csv
├── cost.csv
└── checksums.sha256
```

`manifest.json` must include:

- run ID, condition ID (`frontierscience-research-direct-v0`), code revision, container digest, lockfile hash;
- dataset repository/revision/file hashes and duplicate policy;
- exact model and judge IDs, provider API, resolved model snapshot, OpenRouter endpoint/quantization/routing policy;
- complete generation/judge parameters and prompt hashes;
- output caps, retry policy, concurrency, start/end timestamps, and catalog snapshot;
- subset-selection algorithm/version and trial count;
- explicit `tools=false`, `browsing=false`, `code_execution=false`.

Each answer/trial record must include:

- stable sample/content hash, task group, source row, subject, stratum;
- model label, requested ID, resolved ID/provider, trial index and deterministic idempotency key;
- visible answer, finish reason, status/error, request/response timestamps and IDs;
- input, cached-input, visible-output, reasoning-output, and total tokens where available;
- retry lineage, latency, and estimated/reconciled cost.

Each judgment must include:

- answer artifact ID, judge ID/snapshot/effort, prompt hash;
- raw visible judge output, parsed rubric points, `pass`, threshold;
- parse/status/error and retry lineage;
- input/output/reasoning token usage and cost.

Never store API keys, authorization headers, hidden reasoning traces, or unrestricted environment dumps. Summaries must be reproducible solely from `trials.jsonl` and the manifest.

## 10. Early reporting

Every smoke/pilot table and headline must state, in the title or adjacent text:

> Public FrontierScience-Research gold subset; direct/no-browsing; stratified 9/60 tasks; one answer sample per task; GPT-5 snapshot judge at high effort; directional pilot, not comparable to the official 30-trial protocol.

Report raw fractions such as `2/9`, not only percentages. Include subject rows (`n=3` each), rubric-point diagnostics, errors, cost, and model/provider/scaffold metadata. Any interval is descriptive only: the deterministic stratified subset is not a simple random sample, and one generation per task does not estimate within-task success probability well.

Never:

- place pilot values in the same ranking column as OpenAI's 30-trial results;
- compare the direct lane with AgentCompass search-agent values;
- call a fallback model Qwen;
- imply that provider-native "high" settings equalize test-time compute;
- select or remove pilot questions after seeing model performance.

## 11. Time and cost estimate

### Engineering and wall time

- implementation and static tests: approximately 0.5-1 working day;
- catalog/data preflight: 15-30 minutes;
- one-task × four-model smoke plus grading: approximately 15-60 minutes of API wall time, followed by 30-60 minutes of artifact review;
- 9-task × four-model pilot plus grading: approximately 1-4 hours at conservative concurrency, plus 1-2 hours of manual review and reporting.

Actual high-effort latency is uncertain; smoke observations should replace these estimates before the pilot.

### API cost

Using current list prices and an illustrative average of roughly 3K target-input tokens, 8K billed target-output/reasoning tokens, 10K judge-input tokens, and 3K judge-output/reasoning tokens:

- four-model target generation for 9 tasks: roughly **$4**;
- 36 GPT-5 judge calls: roughly **$1.50**;
- likely pilot order of magnitude: **$6-$15**, depending mainly on reasoning use and output length.

The configured 32K caps make a conservative all-calls-hit-cap estimate materially higher, around **$30 before retries**. Record actual and estimated spend per call and summarize it after smoke and pilot, but do not implement or enforce budget-based stop gates.

A full official-style four-model campaign is orders of magnitude larger in call count and must be costed from pilot observations rather than extrapolated from undocumented official usage.

## 12. Failure modes and mitigations

| Failure mode | Mitigation |
|---|---|
| Model is in public docs but unavailable to the account | Authenticated catalog/entitlement preflight without inference; stop before generation. |
| Gemini "newest Flash" changes | Resolve and pin the exact ID in the manifest; do not use `*-latest` aliases. |
| OpenRouter endpoint/routing drift | Pin one provider, disable fallbacks, require parameters, record canonical slug/provider/quantization. |
| Qwen unavailable | Substitute only with separately labeled `z-ai/glm-5.1`, then `nvidia/nemotron-3-ultra-550b-a55b`; never merge results. |
| Adapter silently drops effort/sampling/output controls | Fail closed on unsupported parameters; avoid LiteLLM-style global `drop_params`. |
| Output truncation or empty visible answer | Record finish reason and mark incomplete; do not selectively extend only favorable models. Decide a campaign-wide cap change and rerun all affected cells under a new condition. |
| Rate limit, timeout, transient 5xx | Bounded exponential retry with idempotency keys; preserve every attempt; no retries for valid low-quality answers. |
| Safety refusal | Count and report as a model outcome if the call completed normally; distinguish from transport errors. |
| Judge unavailable/deprecated | Stop official-inspired scoring; user must approve a separately named replacement-judge lane. |
| Judge parse/API failure | Never coerce to zero; preserve artifact and regrade explicitly. |
| Judge bias/instability | Fixed judge for all models, manual audit, optional predeclared same-judge sensitivity sample, raw grading artifacts retained. |
| Metric mismatch | Primary metric is per-answer `score >= 7`; rubric mean remains diagnostic. Tests must catch average-then-threshold behavior. |
| Dataset drift or duplicate | Pin revision and hashes, validate 60/20-20-20 counts, report duplicate, use immutable subset manifest. |
| Accidental tool/search use | Direct-lane adapter exposes no tool interface; manifest asserts no tools; container has no browser/search service. |
| Unexpected spend | Low concurrency, per-call token/cost recording, and observed-cost reconciliation after smoke; no budget-based stop gate. |
| Secret leakage | Runtime secrets only, sanitized structured logs, no headers/env dumps, secret-pattern scan over final artifacts. |
| Public-set contamination | State that the gold set is public and model training contamination is unknown; do not describe results as uncontaminated capability estimates. |

## 13. Cleanup and reproducibility

After each run:

- stop and remove ephemeral containers; retain only the pinned image reference/digest and intended artifacts;
- delete container scratch volumes and temporary dataset clones after checksum verification;
- keep API keys outside the repository and rotate/revoke only if leakage is detected;
- remove failed download/build caches only when verified disposable;
- run a secret scan and `git status --short` before any future commit;
- do not delete superseded API attempts from the run artifact tree—mark lineage and authoritative records instead;
- document exact commands, exit codes, and environment versions in the manifest.

## 14. Acceptance criteria

### Implementation/preflight

- dataset revision and hashes are pinned; counts are 60 Research, 20 per subject; duplicate is detected;
- all rubrics in the selected subset parse and total 10 points;
- all four exact requested model IDs and the judge are available to the configured accounts;
- Qwen resolves to one explicitly pinned OpenRouter endpoint;
- unsupported generation parameters fail rather than disappear;
- scorer/parser/error-path tests pass;
- container isolation and secret handling match Section 7.

### Smoke

- exactly one canonical question yields one completed answer from each matrix model and one valid judgment per answer;
- manifests contain resolved providers/models, parameters, usage, finish reasons, cost, and prompt hashes;
- no browsing/tool/code path is reachable;
- no malformed judgment, unexplained retry, truncation, secret, or missing raw artifact remains;
- all four judgments are manually reviewed before pilot approval.

### Pilot

- exactly 9 unique tasks, 3 per subject and 1 per length stratum, are evaluated once per model;
- 36 authoritative answer artifacts and 36 authoritative judgments exist, with failed attempts separately retained;
- no unresolved adapter/judge errors or silent model/provider substitutions remain;
- actual and estimated spend is fully recorded and reported;
- summary is reproducible from trial artifacts and carries the one-sample/non-equivalence warning;
- result review explicitly decides stop, extend to 12, run a 3-trial sensitivity pass, or scope a later full campaign.

A full run is not accepted or scheduled merely because the pilot executes successfully.

## 15. Execution decisions and remaining blockers

The user has resolved the execution choices for this campaign:

1. Use `gpt-5-2025-08-07` at high effort as the judge if account access is available; otherwise stop rather than silently change the judge.
2. Prefer a healthy first-party Alibaba endpoint for Qwen, with OpenRouter routing pinned and fallbacks disabled. If Qwen is unavailable, use distinctly labeled `z-ai/glm-5.1`, then `nvidia/nemotron-3-ultra-550b-a55b`.
3. Use high reasoning/thinking for every target model and log effective settings.
4. Use the uniform 32,768-token campaign cap.
5. Run the one-task smoke across all four adapters, then the deterministic 9/60 pilot if clean.
6. Record spend without budget stop gates.
7. Keep the direct/no-tools lane only; defer search-agent work.
8. Keep raw/large/protected artifacts outside Git and commit sanitized milestone reports locally; do not push.

Remaining runtime blockers are operational rather than design choices: API credentials/account entitlement must be injected without printing them, the exact Qwen endpoint must still be healthy at execution time, and the deprecated judge snapshot must still be callable.
