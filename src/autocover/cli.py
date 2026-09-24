"""Command-line entry point: `autocover <command>`."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer

from autocover.config import load_config
from autocover.llm.router import PROVIDER_KEY_ENV, AllModelsFailed, provider_of
from autocover.runtime import build_runtime, load_dotenv
from autocover.tools.mutator import generate_mutants
from autocover.tools.sandbox import RunRequest, docker_available, make_sandbox

app = typer.Typer(help="AutoCover-Lite: multi-agent Python test generation.",
                  no_args_is_help=True, add_completion=False)

ConfigOpt = typer.Option(Path("config.yaml"), "--config", "-c", help="Path to config.yaml")


@app.callback()
def _main() -> None:
    load_dotenv()


@app.command()
def doctor(config: Path = ConfigOpt) -> None:
    """Check config, API keys, Docker and tracing setup."""
    cfg = load_config(config)
    ok = True
    typer.echo(f"config: {config} ({'found' if config.exists() else 'missing, using defaults'})")
    typer.echo("\nLLM roles ([x] = API key present):")
    for role, chain in cfg.llm.roles.items():
        marks = []
        for model in chain:
            env = PROVIDER_KEY_ENV.get(provider_of(model))
            has_key = env is None or bool(os.environ.get(env))
            marks.append(f"{'[x]' if has_key else '[ ]'} {model}")
        usable = any(m.startswith("[x]") for m in marks)
        ok &= usable
        typer.echo(f"  {role:<10} {'OK  ' if usable else 'NONE'}  " + "  ->  ".join(marks))
    has_docker = docker_available()
    typer.echo(f"\nsandbox backend: {cfg.sandbox.backend}; docker engine reachable: {has_docker}")
    if cfg.sandbox.backend == "docker" and not has_docker:
        ok = False
        typer.echo("  start Docker Desktop, or set sandbox.backend: local for development")
    endpoint = cfg.telemetry.otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    typer.echo(f"tracing: {endpoint or 'off (JSONL only)'}  jsonl: {cfg.telemetry.jsonl_path}")
    typer.echo("\nall good" if ok else "\nfix the items above (see .env.example)")
    raise typer.Exit(0 if ok else 1)


@app.command("llm-ping")
def llm_ping(
    role: str = typer.Option("generator", help="Agent role whose fallback chain to test"),
    model: str = typer.Option(None, help="Test one model id instead of a role's chain"),
    prompt: str = typer.Option("Reply with exactly one word: pong"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    config: Path = ConfigOpt,
) -> None:
    """Send one tiny prompt through a role's fallback chain (or to a single model)."""
    cfg = load_config(config)
    if model:
        role = "ping"
        cfg.llm.roles[role] = [model]
    runtime = build_runtime(cfg)
    try:
        resp = asyncio.run(runtime.router.complete(
            role, [{"role": "user", "content": prompt}], use_cache=not no_cache,
            max_tokens=1024))
    except AllModelsFailed as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc
    finally:
        runtime.close()
    typer.echo(f"model={resp.model} fallbacks={resp.fallbacks} cached={resp.cached} "
               f"latency={resp.latency_s:.2f}s tokens={resp.prompt_tokens}+"
               f"{resp.completion_tokens}")
    typer.echo(resp.text.strip())


@app.command()
def usage(config: Path = ConfigOpt) -> None:
    """Show today's (UTC) requests and tokens per model against configured caps."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from autocover.llm.cache import UsageLedger

    cfg = load_config(config)
    ledger = UsageLedger(cfg.llm.usage_path)
    models = sorted({m for chain in cfg.llm.roles.values() for m in chain})
    typer.echo("usage for the current quota day of each provider\n")
    typer.echo(f"{'model':<48} {'day':<11} {'requests':>8} {'rpd cap':>8} {'tokens':>9}  "
               "rpm/tpm")
    for model in models:
        tz = cfg.llm.limits_for(provider_of(model)).quota_timezone
        day = datetime.now(ZoneInfo(tz)).date().isoformat()
        req, tok = ledger.day_summary(day).get(model, (0, 0))
        lim = cfg.llm.model_limits(model)
        cap = str(lim.rpd) if lim.rpd else "-"
        flag = "  CAP REACHED" if lim.rpd and req >= lim.rpd else ""
        rate = "/".join(f"{v:g}" if v else "-" for v in (lim.rpm, lim.tpm))
        typer.echo(f"{model:<48} {day:<11} {req:>8} {cap:>8} {tok:>9}  {rate}{flag}")
    ledger.close()


@app.command("sandbox-run")
def sandbox_run(
    repo: Path = typer.Argument(..., help="Repository root"),
    target: str = typer.Argument(..., help="Module under test, relative to the repo"),
    tests: list[Path] = typer.Argument(..., help="Test files to run"),
    backend: str = typer.Option(None, help="docker | local (default: from config)"),
    mutant: str = typer.Option(None, help="Run against this mutant id instead"),
    config: Path = ConfigOpt,
) -> None:
    """Run test files in the sandbox and print outcomes and coverage."""
    cfg = load_config(config)
    if backend:
        cfg.sandbox.backend = backend
    runtime = build_runtime(cfg)
    overrides = {}
    if mutant:
        source = (repo / target).read_text(encoding="utf-8")
        match = [m for m in generate_mutants(source) if m.id == mutant]
        if not match:
            typer.echo(f"unknown mutant {mutant}; list them with `autocover mutants`", err=True)
            raise typer.Exit(2)
        overrides = {target: match[0].source}
    try:
        box = make_sandbox(repo, cfg.sandbox, runtime.telemetry)
        box.prepare()
        result = box.run(RunRequest(
            target=target, overrides=overrides,
            tests={p.name: p.read_text(encoding="utf-8") for p in tests}))
    finally:
        runtime.close()
    typer.echo(f"status={result.status} passed={result.passed} ({result.duration_s}s)")
    for t in result.tests:
        typer.echo(f"  {t.outcome:<7} {t.nodeid}")
    if not result.passed:
        typer.echo("\n" + result.diagnostics(limit=2000))
    if result.coverage:
        cov = result.coverage
        lines, branches = len(cov.all_lines), len(cov.all_branches)
        typer.echo(f"\ncoverage of {target}: lines {len(cov.executed_lines)}/{lines}, "
                   f"branches {len(cov.executed_branches)}/{branches}; "
                   f"missing lines {sorted(cov.missing_lines)}")
    raise typer.Exit(0 if result.passed else 1)


@app.command()
def mutants(
    file: Path = typer.Argument(..., help="Python module to mutate"),
    function: str = typer.Option(None, help="Only this function (qualname)"),
    limit: int = typer.Option(None, help="Max mutants per function"),
) -> None:
    """List the mutants the Validator would use for a module."""
    found = generate_mutants(file.read_text(encoding="utf-8"),
                             functions={function} if function else None,
                             max_per_function=limit)
    for m in found:
        typer.echo(f"{m.id:<22} line {m.lineno:<4} {m.function or '-':<20} {m.description}")
    typer.echo(f"\n{len(found)} mutants")


if __name__ == "__main__":
    app()
