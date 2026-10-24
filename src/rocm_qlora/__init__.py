"""rocm-qlora: teaching-first QLoRA fine-tuning for AMD ROCm GPUs.

Package layout (each module explains itself — see module docstrings):

    config.py     TOML -> validated pydantic models (the single source of truth)
    data.py       ShareGPT loading, validation, splitting, token statistics
    model.py      NF4 pre-quantization + 4-bit model/LoRA assembly
    merge.py      adapter -> bf16 merge (ready for GGUF export)
    hardware.py   ROCm/GPU discovery, rocm-smi parsing, GPU selection
    probes.py     SDPA / RCCL-IPC / matmul probes (fresh subprocess each)
    launcher.py   GPU selection + worker environment + torchrun orchestration
    sft.py        project config -> trl.SFTConfig mapping
    train.py      the training worker: phases + orchestration
    monitor.py    metrics JSONL, progress lines, health guards, summaries
    plots.py      every matplotlib chart + the Markdown run report
    doctor.py     environment + hardware report (.context artifacts)
    chat.py       interactive sanity-check chat against a trained adapter
    utils.py      small shared helpers (formatting, timing, seeds)
    cli.py        `rocm-qlora <command>` entry point
"""

__version__ = "0.1.0"
