# AutoCover-Lite

A multi-agent Python test generator, modelled on Uber's AutoCover
([ICSE-SEIP 2026](https://homes.cs.washington.edu/~rjust/publ/auto_cover_icse_2026.pdf))
and scaled down to run on free-tier LLM APIs.

Five LangGraph agents (**Preparer → Generator → Executor → Validator → Fixer**) write pytest
tests for a module. A test is kept only if:
- it passes in an isolated Docker sandbox;
- it adds line or branch coverage **or** covers a previously untested scenario;
- it survives the quality gates: best-practice rules, plus mutation testing to catch weak
  assertions.

## Status

| Milestone | What | State |
|---|---|---|
| 1. Foundation | Config, LLM router, cache, telemetry, context retriever, mutator, splicer, Docker sandbox, CLI | ✅ done |
| 2. Happy path | Preparer → Generator → Executor graph with a line-coverage gate | ✅ done |
| 3. Quality loop | Validator (rules, mutation, scenario judge) + Fixer with rollback | ⏳ next |
| 4. Scale & ops | Per-function fan-out, budgets, run summaries | |
| 5. Benchmark | 9 subjects vs a single-prompt baseline; `results.md` | |
| 6. Shipping | GitHub Action that opens test PRs; architecture write-up | |

## How a run works

```
prepare -> generate -> execute -> plan_next --(lines still uncovered, rounds/time left)--> generate
                                         \-> finalize (whole-suite check, write file)
```

1. **Preparer.** Runs the existing tests plus an import probe to get a baseline, asks an LLM
   for happy / edge / error *scenarios* per function, and ranks functions by uncovered lines.
2. **Generator.** Makes one LLM call per function for all of its open scenarios, and splits
   the reply into standalone single-test candidates. Later rounds get the source lines that
   are still uncovered plus the previous round's failures.
3. **Executor.** Runs every candidate in its own sandbox, in parallel. A passing candidate is
   kept only if it adds line or branch coverage (greedy, biggest gain first). Kept tests are
   spliced into the suite.
4. **Finalize.** Runs the merged suite as a whole, drops any test that fails in combination
   with others, and writes `tests/test_<module>_autocover.py`.

```bash
autocover run examples/ticket_price ticket_price.py          # writes the test file
autocover run <repo> src/pkg/mod.py -f some_function --dry-run
```

Live runs on `examples/ticket_price` reach **15% -> 100% lines and 0% -> 100%
branches in one round with 4 LLM calls** (about 20-150s depending on which models answer). They
also show why milestone 3 matters: line
coverage alone kept a test whose expected value was computed *with* the code under test, and
dropped the direct `ticket_price(5) == 5` check as "no new coverage". Mutation testing and
scenario coverage fix exactly that.

## What's built so far

| Component | File | Notes |
|---|---|---|
| LLM router | `src/autocover/llm/router.py` | Per-role fallback chains over free tiers. Each provider has a token bucket and a concurrency semaphore. Retryable errors get jittered exponential backoff, and each model has a circuit breaker. |
| Response cache | `src/autocover/llm/cache.py` | SQLite exact-match cache, so re-runs cost nothing. |
| Telemetry | `src/autocover/telemetry.py` | JSONL events carrying a run correlation id, plus OpenTelemetry spans sent to Jaeger. |
| Context retriever | `src/autocover/tools/context.py` | Import path, module imports, callee *signatures* and the class header, so prompts stay small. |
| Mutator | `src/autocover/tools/mutator.py` | Bounded AST mutation: arithmetic, comparison, boolean, `not`, constants, return values. The number of mutants is capped per function, with a seeded sample. |
| Splicer | `src/autocover/tools/splicer.py` | libcst merges tests without disturbing existing formatting. Imports are deduplicated, and name clashes are renamed deterministically (references included). |
| Sandbox | `src/autocover/tools/sandbox.py` | One Docker image per target repo. Each run gets its own container with `--network none`, memory/CPU/pid limits and a hard timeout, and reports per-test outcomes plus line/branch coverage. |

Models are chosen per role, pinned to exact versions, with 4-7 fallbacks each (full chains
in `config.yaml`):

| Role | Primary | Fallbacks, in order |
|---|---|---|
| Generator | GPT-OSS 120B (Ollama Cloud) | Gemini 3.5 Flash -> Gemini 2.5 Flash -> Nemotron 3 Super (Cloudflare) -> Nemotron 3 Super (NVIDIA NIM) -> Codestral 2508 |
| Fixer | Nemotron 3 Super (NIM) | Codestral 2508 -> GPT-OSS 120B (Ollama) -> Nemotron 3 Super (Cloudflare) -> GLM-4.7-Flash -> Gemini 3.5 / 2.5 Flash |
| Preparer | Nemotron 3 Ultra (NIM) | Nemotron 3 Ultra (Ollama) -> Gemini 3.5 Flash -> GPT-OSS 120B (Groq) |
| Validator judge | GPT-OSS 20B (Groq) | Ministral 14B -> GPT-OSS 20B (Ollama) -> GPT-OSS 20B (Cloudflare) -> Gemini 3.5 Flash-Lite -> GLM-4.5-Flash |

The Generator order comes from `bench/model_bakeoff.py` (one round per model on identical
cached scenarios; results in `bench/results/model_bakeoff.md`):

| Generator model | Tests passing | Coverage | Latency/call |
|---|---|---|---|
| Gemini 3.5 Flash | 100% | 100% | 7s |
| GPT-OSS 120B (Ollama) | 100% | 100% | 9s |
| Nemotron 3 Super (Cloudflare / Ollama / NIM) | 75% / 67% / 58% | 100% / 100% / 92% | 11s / 15s / 96s |
| Codestral 2508 | 53% | 92% | 5s |

That's one small module, so treat it as a first signal. Milestone 5 reruns it on 9 subjects.

The same model on several providers (Nemotron 3 Super on NVIDIA NIM, Ollama Cloud and
Cloudflare) gives independent quotas at one quality level. Gemini Flash sits late in every chain because the free tier
allows only **20 requests per day per Flash version**. NVIDIA NIM has no daily cap (~40 RPM
for the account).

Free tiers can use your prompts for training (Mistral's free plan requires opting in, and
Google does outside the EU/UK), so only point this tool at code you are allowed to share.

Free-tier quotas are metered **per model version**, so `config.yaml` pins versions (no
`-latest` aliases) and sets per-model `rpm` / `tpm` / `rpd` limits taken from each
provider's rate-limit headers. The router skips a model when:
- its daily cap is used up (tracked in `.autocover/usage.sqlite` per provider quota day;
  Gemini resets at midnight Pacific), or the provider has answered with a *daily-quota*
  429, which is never retried, or
- its limits would make a call wait longer than `max_queue_wait_s`.

`autocover usage` shows today's usage against each cap.

## Quickstart

Requires Python 3.11+ and Docker Desktop (WSL2 backend on Windows).

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate elsewhere
pip install -e ".[dev]"
cp .env.example .env                                # add the free API keys you have
autocover doctor                                    # checks keys, Docker, tracing
autocover llm-ping --role generator                 # one prompt through the fallback chain
```

Run hand-written tests through the sandbox and see coverage:

```bash
autocover sandbox-run examples/ticket_price ticket_price.py path/to/test_file.py
autocover mutants examples/ticket_price/ticket_price.py --function ticket_price
```

To view traces:

```bash
docker compose up -d jaeger
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
# ...run a command, then open http://localhost:16686 (service: autocover-lite)
```

## Tests

```bash
pytest -q          # Docker sandbox tests run when an engine is reachable, else skip
ruff check src tests
```
