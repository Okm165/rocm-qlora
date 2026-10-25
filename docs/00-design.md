# 00 · Design notes (what to expect)

This tool is deliberately small and predictable. The rules below shape what
you will see while using it; each one exists because the alternative was worse
in practice.

* **Facts on the terminal, knowledge in the docs.** Commands print what
  happened and what to do next — status/progress lines, warnings, tables,
  artifact paths. Concepts, rationale and exact commands live in these docs;
  nothing is taught at runtime.
* **One artifact per fact.** `metrics.jsonl` is the single source of truth:
  charts, `report.md` and the end-of-run numbers all derive from it, so
  nothing needs keeping in sync. The other files in a run directory are
  records of one thing each (config snapshot, environment fingerprint).
* **No fit prediction.** The launcher reports visible GPUs, free VRAM and the
  processes holding them, then launches. A run either fits or OOMs — the
  measured per-GPU budget is in `docs/02`, the knobs to react with in
  `docs/05`.
* **Decisions are static, never discovered.** Attention (`model.attn`,
  `model.sdpa_disable`), the AOTriton flag (`env.aotriton`), the training
  template family (`data.training_template`) and the GPU that `prepare` /
  `chat` use (`dist.gpus`'s first entry) are values you set and can read
  back. Nothing probes the machine and reconfigures a run behind your back;
  `docs/02` explains what each one does and what to check on your hardware.
* **The console belongs to the tool.** Status, progress, warnings and
  artifact paths — nothing else. Library output (transformers/TRL INFO,
  fallback notices, tqdm bars) goes to `logs/rank-*.log`, and RCCL runs at
  `NCCL_DEBUG=ERROR` (`docs/02`). Path and command lines never wrap, so they
  stay copy-pasteable.
* **GPU-only, no CPU paths.** Model work (quantize, train, merge) runs on the
  GPUs; host RAM is staging for checkpoints too large for one card, nothing
  more. If a device set is too small, the run fails loudly — the numbers to
  react with are in `docs/02` and `docs/06`.
* **Loud guards, no silent rot.** Numerically broken training (NaN or
  exactly-zero loss, NaN `grad_norm`) aborts with a named `error` event and a
  doc pointer instead of continuing (`docs/05`). The same rule covers
  everything else: a second run in one directory and a `prepare` without a
  visible GPU fail loudly, and unknown numbers (tokens, wall time) render as
  `n/a` — never as zero.
