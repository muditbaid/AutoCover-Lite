# AutoCover-Lite

A multi-agent Python test generator, modelled on Uber's AutoCover
([ICSE-SEIP 2026](https://homes.cs.washington.edu/~rjust/publ/auto_cover_icse_2026.pdf)),
that runs on free-tier LLM APIs and plugs in stronger (paid) models through its config.

Five agents (Preparer, Generator, Executor, Validator, Fixer) write pytest tests for one
module at a time. A test is kept only if it passes in an isolated Docker sandbox, adds
coverage or covers an untested scenario, and survives the quality gates: best-practice
rules and mutation testing (it must catch deliberately planted bugs).

On 9 modules from popular PyPI packages, against a single-prompt baseline with the same
models: line coverage 80.9% -> 92.5%, branch coverage 72.4% -> 87.5%, mutation score
56.8% -> 74.8%.

## Install and use

Requires Python 3.11+ and Docker.

```bash
pipx install autocover-lite        # or: pip install autocover-lite
cd your-repo
autocover init                     # autocover.yaml, .env.example, .gitignore entries
cp .env.example .env               # add the API keys you have (free tiers work)
autocover doctor                   # checks keys, Docker and the config
autocover run . src/yourpkg/module.py
```

`autocover init --workflow` adds a GitHub Actions workflow that writes tests for the
modules a pull request changes and opens a follow-up pull request with them.

The code of the module under test is sent to the configured LLM providers; free tiers may
use prompts for training.

## Links

- [Full documentation, architecture and benchmark](https://github.com/muditbaid/AutoCover-Lite#readme)
- [Plugging in better models](https://github.com/muditbaid/AutoCover-Lite#plugging-in-better-models)
- [Changelog](https://github.com/muditbaid/AutoCover-Lite/blob/main/CHANGELOG.md)
- [Issues](https://github.com/muditbaid/AutoCover-Lite/issues)
