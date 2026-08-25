# FrontierScience-Research bounded evaluation

Minimal direct single-turn runner for the pinned public FrontierScience-Research gold set. It has no tools, browser, code execution, agent loop, or evaluation framework. The one-question smoke and deterministic 9-question pilot are complete; full-set/multi-trial support is prepared but has not been launched.

See [`PLAN.md`](PLAN.md) for the bounded campaign and [`FULL_RUNBOOK.md`](FULL_RUNBOOK.md) for the canonical environment audit, offline full-run plan, estimates, and required pre-launch decision.

## Offline checks

```bash
python3 -m frontierscience_eval validate-data --data data/research-test.jsonl
python3 -m unittest discover -s tests -v
```

Containerized:

```bash
docker build -t frontierscience-eval:dev .
docker run --rm --name frontierscience-eval-tests --read-only --network none \
  -v "$PWD/data:/work/data:ro" frontierscience-eval:dev \
  python -m unittest discover -s tests -v
```

## Credentials

Only the authorized files below are mounted, read-only, at runtime:

- `~/secrets_and_keys/openai.key` → `/run/secrets/openai`
- `~/secrets_and_keys/anthropic.key` → `/run/secrets/anthropic`
- `~/secrets_and_keys/gemini.key` → `/run/secrets/gemini`
- `~/secrets_and_keys/openrouter.key` → `/run/secrets/openrouter`

Keys are read directly into request memory. They are not printed, copied into the image, or written to artifacts.

## Artifacts

Raw answers, judge explanations, provider responses, and protected manifests live under ignored `artifacts/`. `summarize --report reports/<name>.md` writes a small sanitized report suitable for local Git.

All reported smoke/pilot values are directional and non-equivalent to OpenAI's official 30-trial-per-question protocol.
