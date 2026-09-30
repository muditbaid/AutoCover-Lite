from pathlib import Path

from typer.testing import CliRunner

from autocover.cli import app

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "ticket_price"
runner = CliRunner()


def test_mutants_lists_ticket_price_mutants():
    result = runner.invoke(app, ["mutants", str(EXAMPLE / "ticket_price.py"),
                                 "--function", "ticket_price"])
    assert result.exit_code == 0
    assert ">= -> >" in result.output and "mutants" in result.output


def test_sandbox_run_local(tmp_path):
    test_file = tmp_path / "test_tp.py"
    test_file.write_text("from ticket_price import ticket_price\n\n"
                         "def test_adult():\n    assert ticket_price(30) == 10\n")
    config = tmp_path / "config.yaml"
    config.write_text(f"sandbox:\n  backend: local\n  workdir: {tmp_path.as_posix()}/ws\n"
                      f"telemetry:\n  jsonl_path: null\nllm:\n  cache:\n    enabled: false\n")
    result = runner.invoke(app, ["sandbox-run", str(EXAMPLE), "ticket_price.py",
                                 str(test_file), "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "passed=True" in result.output and "coverage of ticket_price.py" in result.output


def test_doctor_reports_missing_keys(tmp_path, monkeypatch):
    from autocover.config import DEFAULT_CONFIG, load_config
    from autocover.llm.router import PROVIDER_KEY_ENV

    providers = load_config(DEFAULT_CONFIG).llm.providers.values()
    for env in {*PROVIDER_KEY_ENV.values(), *(p.api_key_env for p in providers if p.api_key_env)}:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    result = runner.invoke(app, ["doctor", "--config", str(DEFAULT_CONFIG)])
    assert result.exit_code == 1
    assert "NONE" in result.output


def test_init_sets_up_a_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".gitignore").write_text("__pycache__/\n.env\n", encoding="utf-8")
    result = runner.invoke(app, ["init", "--workflow"])
    assert result.exit_code == 0, result.output
    from autocover.config import find_config, load_config

    assert find_config() == Path("autocover.yaml")  # the new file is picked up by default
    assert load_config().llm.roles["generator"]
    env = (tmp_path / ".env.example").read_text(encoding="utf-8").splitlines()
    assert "OLLAMA_API_KEY=" in env and "CLOUDFLARE_ACCOUNT_ID=" in env
    assert all(line.endswith("=") for line in env if not line.startswith("#"))  # no values
    assert "uses: muditbaid/AutoCover-Lite@" in (
        tmp_path / ".github" / "workflows" / "autocover.yml").read_text(encoding="utf-8")
    ignore = (tmp_path / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ignore.count(".env") == 1 and ".autocover/" in ignore

    (tmp_path / "autocover.yaml").write_text("run:\n  budget_min: 3\n", encoding="utf-8")
    again = runner.invoke(app, ["init"])
    assert "exists, kept: autocover.yaml" in again.output  # never overwrites without --force
    assert load_config().run.budget_min == 3
