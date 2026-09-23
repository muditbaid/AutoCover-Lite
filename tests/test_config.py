from pathlib import Path

from autocover.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_missing_file_gives_defaults(tmp_path):
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg == Config()
    assert cfg.sandbox.backend == "docker"


def test_repo_config_parses_roles_and_limits():
    cfg = load_config(ROOT / "config.yaml")
    assert set(cfg.llm.roles) == {"generator", "fixer", "preparer", "judge"}
    assert all(cfg.llm.roles.values())
    assert cfg.llm.limits_for("gemini").rpm > 0
    assert cfg.llm.limits_for("unknown-provider").max_concurrency == 1
