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
    from autocover.config import load_config
    from autocover.llm.router import PROVIDER_KEY_ENV

    providers = load_config(ROOT / "config.yaml").llm.providers.values()
    for env in {*PROVIDER_KEY_ENV.values(), *(p.api_key_env for p in providers if p.api_key_env)}:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    result = runner.invoke(app, ["doctor", "--config", str(ROOT / "config.yaml")])
    assert result.exit_code == 1
    assert "NONE" in result.output
