from pathlib import Path

from autocover.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_missing_file_gives_defaults(tmp_path):
    cfg = load_config(tmp_path / "nope.yaml")
    assert cfg == Config()
    assert cfg.sandbox.backend == "docker"


def test_repo_config_parses_roles_and_limits():
    cfg = load_config(ROOT / "config.yaml")
    assert set(cfg.llm.roles) == {"generator", "fixer", "fixer_final", "preparer", "judge"}
    assert all(cfg.llm.roles.values())
    # Reservations fit in their model's daily cap and name roles that exist.
    for model, limits in cfg.llm.models.items():
        if limits.reserve:
            assert limits.rpd and sum(limits.reserve.values()) <= limits.rpd, model
            assert set(limits.reserve) <= set(cfg.llm.roles), model
    # Versions are pinned: no floating "-latest" aliases in any chain.
    assert not any("latest" in m for chain in cfg.llm.roles.values() for m in chain)
    # Groq's measured free-tier limits are per model; NIM's RPM is account-wide.
    groq = cfg.llm.model_limits("groq/openai/gpt-oss-20b")
    assert groq.tpm and groq.tpm < 8000 and groq.rpd and groq.rpd < 1000
    assert cfg.llm.limits_for("nvidia_nim").rpm < 40
    assert cfg.llm.model_limits("unknown/model").rpm is None
    assert cfg.llm.limits_for("unknown-provider").max_concurrency == 1


def test_paid_example_config_is_a_drop_in_replacement():
    paid = load_config(ROOT / "examples" / "config.paid.yaml")
    free = load_config(ROOT / "config.yaml")
    assert set(paid.llm.roles) == set(free.llm.roles)  # every agent role has a chain
    assert all(paid.llm.roles.values())
    # Free-tier rationing is off; everything outside `llm` is the repository's config.
    assert paid.llm.runs_per_day is None and not paid.llm.role_queue_wait_s
    assert not any(limits.reserve for limits in paid.llm.models.values())
    assert (paid.sandbox, paid.mutation, paid.run) == (free.sandbox, free.mutation, free.run)
