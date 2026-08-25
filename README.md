# FrontierScience-Research Eval

A minimal, reproducible runner for the public **FrontierScience-Research** gold set. It evaluates direct, single-turn, text-only model responses and grades them against the released 10-point rubrics.

This is an independent implementation. It is not affiliated with or endorsed by OpenAI.

## Status

- One-question four-model smoke: complete — [report](reports/smoke-20260825.md)
- Deterministic 9/60-question pilot: complete — [report](reports/pilot-20260825.md)
- Full 60-question / multi-trial support: prepared and dry-validated, **not launched**

See [`FULL_RUNBOOK.md`](FULL_RUNBOOK.md) for the canonical protocol audit, environment details, isolation, full-run commands, estimates, and unresolved output-cap decision. [`PLAN.md`](PLAN.md) records the original bounded campaign design.

## Implementation lineage

There is **no official OpenAI FrontierScience runner or evaluator repository**. The canonical inputs for this project are:

1. OpenAI's [FrontierScience paper](https://arxiv.org/abs/2601.21165), including the Research protocol and Appendix B judge prompt;
2. OpenAI's [benchmark page](https://openai.com/index/frontierscience/); and
3. the Apache-2.0 [`openai/frontierscience`](https://huggingface.co/datasets/openai/frontierscience) Hugging Face dataset, pinned here at revision:

   ```text
   25ed67db7da8f4591484e764008ff585544f5a30
   ```

   The pinned `research/test.jsonl` SHA-256 is:

   ```text
   96c0434abfcbadd6ef6f59a03cc374be4caf9c1f2d5e62d8fe921e768f66aa46
   ```

During planning, several community implementations were inspected read-only: UK AISI's `inspect_evals`, `medicalsphere/FrontierScience`, `EnvCommons/FrontierScience`, and `open-compass/AgentCompass`. None is vendored or used at runtime. This repository does not inherit Inspect Evals' normalized-mean Research scorer or AgentCompass's search-agent condition.

The local `frontierscience_eval` package was written from scratch using the Python standard library. It provides direct provider HTTP adapters, pinned-data validation, deterministic subset/full-set selection, rubric judging, resumable JSONL artifacts, cost summaries, and a small Docker environment. No agent or general evaluation framework is required.

## Evaluation condition

### What target models can access

Each target model receives exactly one user message containing the released `problem` text. It does **not** receive the rubric, answer, subject metadata, previous attempts, judge feedback, files, or tool definitions.

The published FrontierScience results use **no browsing**. Research questions are designed to be self-contained and to test reasoning rather than search or recency. Accordingly, this runner provides no browser, search, retrieval, or URL-visit tool. Enabling search would create a separate, non-comparable scaffold condition.

No code-execution environment is available or required. Model output remains inert text and is never imported, interpreted, or executed.

### Target model matrix

The current configuration uses high reasoning/thinking and a 32,768-token output cap:

| Lane | Exact model ID | Provider condition |
|---|---|---|
| GPT-5.6 Sol | `gpt-5.6-sol` | OpenAI Responses API, `reasoning.effort=high` |
| Claude Opus 5 | `claude-opus-5` | Anthropic Messages API, adaptive thinking, `output_config.effort=high` |
| Gemini Flash | `gemini-3.7-flash` | Gemini API, `thinkingLevel=HIGH` |
| Qwen3.5 | `qwen/qwen3.5-397b-a17b` | OpenRouter, reasoning enabled, Alibaba endpoint pinned, provider fallbacks disabled |

Provider/scaffold settings are part of the result condition and are recorded in each run manifest.

### Primary judge

The primary adjudication condition is pinned to:

```text
model: gpt-5-2025-08-07
reasoning effort: high
output cap: 32,768
pass threshold: rubric score >= 7/10
```

The paper specifies “GPT-5 at high reasoning” but does not print a dated API snapshot. `gpt-5-2025-08-07` is this implementation's reproducible mapping to that reference. The local prompt is semantically equivalent to Appendix B, with the paper's `attemped` typo corrected and whitespace normalized; its hash is recorded in manifests.

The judge receives the problem, full rubric, and one attempted answer. The primary metric is the mean of per-answer binary `score >= 7` outcomes. Mean rubric points are diagnostic only; the runner does not average points first and then threshold.

## Alternative judges and judge-dependence sensitivity

Rubric judging is model-dependent. Alternative judge models may be evaluated to measure disagreement, self-preference, or other judge dependence, but each must be treated as a **separate, versioned adjudication condition**.

Never silently replace, average, vote, or pool an alternative judge with the pinned primary judge. Every judge condition should record at least:

- exact judge model ID, provider, and snapshot/alias status;
- reasoning setting, output cap, and sampling parameters;
- judge prompt hash and parser version;
- scoring threshold and retry policy; and
- run timestamp and source answer-artifact IDs.

A small sensitivity study does not require a generalized framework:

1. Freeze the target-model answer artifacts; never regenerate answers for judge comparison.
2. Predeclare a small set, for example 12–24 answers balanced across target models and subjects. It may deliberately include primary-judge score bands below 5, from 5 to below 7, and at/above 7 to test threshold sensitivity.
3. Regrade those exact answers once with each alternative judge under a new condition ID.
4. Report point-score mean absolute difference, pass/fail disagreement rate, a 2×2 pass confusion table, and disagreements near the 7-point threshold, both overall and by target model.
5. If expert adjudication is available, review disagreements separately; do not retroactively rewrite or pool the pinned primary scores.

This protocol diagnoses judge dependence. It does not by itself prove that an alternative judge is less biased.

## Dataset setup and offline validation

The benchmark data is intentionally not committed. Fetch and validate the pinned Research JSONL:

```bash
python3 -m frontierscience_eval fetch-data --data data/research-test.jsonl
python3 -m frontierscience_eval validate-data --data data/research-test.jsonl
python3 -m unittest discover -s tests -v
```

The release contains 60 Research rows, 20 per subject. Two released rows are exact duplicates, so the canonical file has 59 unique payloads. Full-set evaluation retains both official rows while assigning distinct internal IDs; reports should optionally include a 59-unique-payload sensitivity view.

A no-network, no-secret full-protocol dry plan:

```bash
docker build -t frontierscience-eval:dev .

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

The prepared 60×30 configuration reports 7,200 target calls plus 7,200 judge calls. Both full generation and full judging require an explicit `--allow-full-run` flag. No full run has been performed.

## Credentials and isolation

The container expects provider key files mounted read-only at:

```text
/run/secrets/openai
/run/secrets/anthropic
/run/secrets/gemini
/run/secrets/openrouter
```

Host paths are user-controlled; the runbook uses `~/secrets_and_keys/*.key` as an example. Keys are read into request memory only. They are not copied into the image, printed, or written to artifacts.

Recommended run controls include a read-only root filesystem, non-root user, all capabilities dropped, `no-new-privileges`, no host network, no Docker socket, bounded CPU/memory/PIDs, read-only dataset and secret mounts, and one scoped writable artifact mount. Offline checks and reporting use `--network none`.

## Artifacts and reproducibility

Raw answers, rubrics, judge explanations, and provider responses live under ignored `artifacts/`. The downloaded dataset lives under ignored `data/`. Only source, secret-free configurations, small sanitized manifests, and reports are tracked.

Relevant files:

```text
frontierscience_eval/       # local standard-library runner
configs/                    # smoke, pilot, and prepared full conditions
reports/                    # sanitized smoke/pilot outputs
FULL_RUNBOOK.md             # protocol audit and full-run operations
PLAN.md                     # original bounded evaluation plan
```

## Results and comparability

The committed smoke and 9-question pilot are directional. They use one answer per task and are **not equivalent** to the paper's 30-independent-trial Research protocol. Search-agent results, alternative judges, changed output caps, changed providers, and fallback models must be reported as separate conditions.

## License and attribution

Repository code and documentation are released under the [Apache License 2.0](LICENSE).

The FrontierScience dataset is separately distributed by OpenAI under Apache-2.0. This repository does not redistribute the dataset. Follow the dataset card's instruction that benchmark data must not appear in training corpora.

When using the benchmark, cite the canonical FrontierScience paper and dataset:

```bibtex
@article{wang2026frontierscience,
  title={FrontierScience: Evaluating AI's Ability to Perform Expert-Level Scientific Tasks},
  author={Wang, Miles and Lin, Robi and Hu, Kat and Jiao, Joy and Chowdhury, Neil and Chang, Ethan and Patwardhan, Tejal},
  journal={arXiv preprint arXiv:2601.21165},
  year={2026}
}
```
