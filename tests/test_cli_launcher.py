"""CLI parser + launcher policy tests (pure logic, no GPU, no model)."""

from __future__ import annotations

import json

from rocm_qlora.cli import _resolve_resume_checkpoint, build_parser
from rocm_qlora.config import Config
from rocm_qlora.launcher import build_worker_env, load_doctor_verdict, make_run_dir


def test_parser_accepts_every_command():
    parser = build_parser()
    for argv in (
        ["doctor"],
        ["data"],
        ["prepare"],
        ["report", "--run", "x"],
        ["merge", "--run", "x"],
        ["chat", "--run", "x"],
    ):
        args = parser.parse_args(argv)
        assert args.command == argv[0]
    train = parser.parse_args(
        ["train", "--config", "c.toml", "--gpus", "0,1", "--run-name", "renamed", "--seed", "7", "--dry-run"]
    )
    assert train.command == "train"
    assert (train.gpus, train.run_name, train.seed, train.dry_run) == ("0,1", "renamed", 7, True)


def test_load_config_applies_cli_overrides(tmp_path):
    from rocm_qlora.cli import _load_config

    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text('[run]\nname = "x"\n')
    args = build_parser().parse_args(
        ["train", "--config", str(cfg_file), "--gpus", "1,2", "--run-name", "renamed", "--seed", "7"]
    )
    cfg = _load_config(args)
    assert cfg.dist.gpus == [1, 2]
    assert cfg.run.name == "renamed" and cfg.run.seed == 7


def test_resolve_resume_checkpoint(tmp_path):
    run = tmp_path / "run"
    (run / "adapter" / "checkpoint-42").mkdir(parents=True)
    latest = run / "adapter" / "checkpoint-126"
    latest.mkdir()
    assert _resolve_resume_checkpoint(str(run)) == latest  # highest step wins

    (latest / "trainer_state.json").write_text("{}")
    assert _resolve_resume_checkpoint(str(latest)) == latest  # a checkpoint dir is itself

    assert _resolve_resume_checkpoint(str(tmp_path / "missing")) is None
    assert _resolve_resume_checkpoint(str(tmp_path)) is None


def test_make_run_dir_layout(tmp_path):
    cfg = Config.from_dict({"run": {"output_root": str(tmp_path), "name": "unit"}})
    run = make_run_dir(cfg, dry_run=True)
    assert run.name.endswith("-unit-dryrun")
    assert (run / "logs").is_dir() and (run / "charts").is_dir()


def test_make_run_dir_never_reuses_a_directory(tmp_path):
    """Minute-resolution slugs can collide; the second run gets a suffix."""
    from rocm_qlora import launcher
    from rocm_qlora.config import Config

    cfg = Config.from_dict({"run": {"name": "x", "output_root": str(tmp_path)}})
    first = launcher.make_run_dir(cfg)
    second = launcher.make_run_dir(cfg)
    assert first != second and second.name.endswith("-2")
    assert (second / "logs").is_dir() and (second / "charts").is_dir()


def test_sdpa_verdict_only_a_flash_pass_selects_sdpa():
    """The probe's decision: fused only on a pass; mem-efficient never on a pass alone."""
    from rocm_qlora.probes import _sdpa_verdict

    ok = {"ok": True}
    bad = {"ok": False, "error": "RuntimeError: No available kernel"}
    # flash passes with AOTriton on -> sdpa + the flag
    assert _sdpa_verdict({"flash_on": ok, "efficient_on": bad, "math_on": ok}) == ("aotriton_on", "sdpa")
    # mem-efficient passing alone does NOT select sdpa: its backward is the
    # observed intermittent-NaN source, so a probe pass does not justify it
    assert _sdpa_verdict({"flash_on": bad, "flash_off": bad, "efficient_on": ok, "math_on": ok}) == (
        "eager_only",
        "eager",
    )
    # flash only without the flag -> sdpa, flag off
    assert _sdpa_verdict({"flash_on": bad, "flash_off": ok, "math_on": ok}) == ("aotriton_off", "sdpa")
    # only the mem-efficient path survives, and only with the flag off -> eager
    # (its backward is the known intermittent-NaN source; not chosen on a probe pass)
    assert _sdpa_verdict({"flash_on": bad, "flash_off": bad, "efficient_off": ok, "math_on": ok}) == (
        "eager_only",
        "eager",
    )
    assert _sdpa_verdict({"flash_on": bad, "math_on": bad}) == ("all_failed", "eager")


def test_build_worker_env_policy():
    cfg = Config.from_dict({})
    env = build_worker_env(cfg, [0, 2])
    assert env["HIP_VISIBLE_DEVICES"] == "0,2" and env["CUDA_VISIBLE_DEVICES"] == "0,2"
    assert env["TOKENIZERS_PARALLELISM"] == "false"
    # The stable transport defaults (documented in the function).
    assert env["NCCL_P2P_DISABLE"] == "1" and env["HSA_ENABLE_IPC_MODE_LEGACY"] == "0"
    assert env["PYTHONHASHSEED"] == str(cfg.run.seed)

    # [env.extra] wins over every default; aotriton=off removes the flag.
    cfg2 = Config.from_dict({"env": {"extra": {"NCCL_P2P_DISABLE": "0"}, "aotriton": "off"}})
    env2 = build_worker_env(cfg2, [1])
    assert env2["NCCL_P2P_DISABLE"] == "0"
    assert "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL" not in env2


def test_doctor_verdict_readers(tmp_path, monkeypatch):
    from rocm_qlora import launcher

    verdict_file = tmp_path / "doctor-latest.json"
    monkeypatch.setattr(launcher, "DOCTOR_VERDICT", verdict_file)
    assert load_doctor_verdict() is None  # missing file → None, no raise

    verdict_file.write_text(
        json.dumps(
            {
                "created": "2026-10-23_0935",
                "sdpa": {
                    "verdict": "aotriton_on",
                    "attn_implementation": "sdpa",
                    "backends": {"flash_on": {"ok": True}, "efficient_on": {"ok": False, "error": "no kernel"}},
                },
            }
        )
    )
    cfg = Config.from_dict({})
    assert launcher.resolve_attn_impl(cfg) == "sdpa"
    assert launcher.resolve_aotriton(cfg) is True

    # config wins over the verdict
    cfg2 = Config.from_dict({"model": {"attn": "eager"}, "env": {"aotriton": "off"}})
    assert launcher.resolve_attn_impl(cfg2) == "eager"
    assert launcher.resolve_aotriton(cfg2) is False


def test_max_length_hint_rounds_up_past_p99(monkeypatch):
    """The suggestion must cover >= p99; rounding down would miss it (1425 -> 1536, not 1408)."""
    from rocm_qlora import cli
    from rocm_qlora.config import Config
    from rocm_qlora.data import Percentiles

    printed: list[str] = []
    monkeypatch.setattr(cli.console, "print", lambda *a, **k: printed.append(str(a[0])))
    cli._print_max_length_hint(Config.from_dict({}), Percentiles(400.0, 900.0, 1100.0, 1425.0, 2000.0))
    assert "1536" in printed[0]


def test_apply_sdpa_verdict_disables_only_flagged_backends(monkeypatch):
    """The NaN guard: a probe-flagged backend is disabled; a malformed entry is not."""
    import torch

    from rocm_qlora import launcher

    monkeypatch.setenv("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    monkeypatch.setattr(
        launcher,
        "load_doctor_verdict",
        lambda: {"sdpa": {"backends": {"flash_on": {"ok": True}, "efficient_on": {"ok": False}}}},
    )
    flash_before = torch.backends.cuda.flash_sdp_enabled()
    mem_before = torch.backends.cuda.mem_efficient_sdp_enabled()
    try:
        assert launcher.apply_sdpa_verdict() == ["mem_efficient"]
        assert torch.backends.cuda.mem_efficient_sdp_enabled() is False
        assert torch.backends.cuda.flash_sdp_enabled() is True
    finally:
        torch.backends.cuda.enable_flash_sdp(flash_before)
        torch.backends.cuda.enable_mem_efficient_sdp(mem_before)

    # a hand-edited or truncated entry must neither crash nor disable anything
    monkeypatch.setattr(
        launcher,
        "load_doctor_verdict",
        lambda: {"sdpa": {"backends": {"flash_on": "oops", "efficient_on": {"ok": True}}}},
    )
    assert launcher.apply_sdpa_verdict() == []
