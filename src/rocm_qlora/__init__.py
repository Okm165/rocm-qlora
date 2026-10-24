"""rocm-qlora: GPU-only QLoRA fine-tuning for AMD ROCm GPUs.

Package layout (each module explains itself — see module docstrings):

    config.py     TOML -> validated pydantic models (the single source of truth)
    data.py       ShareGPT loading, validation, splitting, token statistics
    prepare.py    checkpoint surgery: NF4 re-quantization + finetune_meta.json
    model.py      4-bit load + selective fp32 cast + LoRA (training-time assembly)
    merge.py      adapter -> bf16 merge (ready for GGUF export)
    hardware.py   ROCm/GPU discovery, rocm-smi parsing, GPU selection
    probes.py     SDPA / HIP-IPC / bnb-smoke / matmul GPU probes
    launcher.py   GPU selection + worker environment + torchrun orchestration
    sft.py        project config -> trl.SFTConfig mapping
    train.py      the training worker: phases + orchestration
    metrics.py    metrics JSONL: records, writer/reader, summaries, health guard
    callback.py   the training-time callback: progress lines, charts, GPU sampling
    plots.py      every matplotlib chart + the Markdown run report
    doctor.py     environment + hardware report (.context artifacts)
    chat.py       interactive sanity-check chat against a trained adapter
    utils.py      small shared helpers (formatting, timing, seeds)
    cli.py        `rocm-qlora <command>` entry point
"""

# From the installed distribution metadata (pyproject.toml is the one place
# the version is written); "0+unknown" for a bare source checkout.
try:
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("rocm-qlora")
except Exception:  # pragma: no cover - metadata always present in an install
    __version__ = "0+unknown"
