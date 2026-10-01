# Changelog

## 0.1.0 - first release

- Five-agent pipeline (Preparer, Generator, Executor, Validator, Fixer) on LangGraph:
  tests are kept only if they pass in a Docker sandbox, add coverage or cover an untested
  scenario, pass the best-practice rules and kill mutants.
- Router over free-tier LLM providers: per-role fallback chains, rate and daily limits,
  reservations for high-leverage roles, deadlines on every call.
- Time budget enforced inside the run (planning cap, round and fix-cycle gating,
  estimated finalize reserve).
- `autocover init`, `doctor`, `run`, `report`, `usage`; built-in free-tier defaults and
  an example config for paid models.
- GitHub Action that opens a pull request with generated tests for changed modules.
- Benchmark: 9 PyPI modules vs a single-prompt baseline (lines 80.9% -> 92.5%, mutation
  score 56.8% -> 74.8%).
