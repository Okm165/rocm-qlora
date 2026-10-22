"""The teaching layer: contextual explainers rendered as rich panels.

Design goals
------------
* **Just-in-time**: explainers appear when the concept first matters (e.g.
  the loss-reading guide appears on the first logged step, not in a wall of
  text at startup).
* **Substantive**: each explainer answers "what is this, what to
  expect, what to do if it looks wrong" — written for someone with a CS +
  statistics background who is new to LLM fine-tuning.
* **Non-intrusive**: explainers are deduplicated per run, can be disabled
  with ``--quiet``, and never block the dashboard (they are printed before
  or after the live display, not during it).
* **Deep links**: every explainer ends with "Read more" references, and the
  matching deep-dive lives in ``docs/``.

The registry is a plain dict, so tests can assert coverage and docs can be
generated from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text


@dataclass(frozen=True)
class Explainer:
    key: str
    title: str
    body: str
    level: str = "core"  # "core" | "advanced"
    links: tuple[str, ...] = ()
    tags: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

EXPLAINERS: dict[str, Explainer] = {
    e.key: e
    for e in [
        # ------------------------------------------------------------------
        Explainer(
            key="qlora",
            title="QLoRA in one panel",
            body=(
                "This run fine-tunes a model in **4-bit** while training only a tiny "
                "set of new weights (the **LoRA adapter**). Three ideas make this work:\n\n"
                "1. **NF4 quantization** — the frozen base weights are stored in a 4-bit "
                "format shaped for normally-distributed weights (~0.5 bytes/param instead of "
                "2 bytes). The forward/backward math still happens in bf16.\n"
                "2. **Double quantization** — the quantization constants themselves are "
                "quantized, saving another ~0.4 bits/param.\n"
                "3. **LoRA adapters** — small rank-`r` matrices are added next to the frozen "
                "layers. Only they receive gradients, so the optimizer state is tiny and "
                "the base model cannot 'forget' catastrophically.\n\n"
                "**What to expect**: memory drops ~4x vs bf16 training; quality stays very "
                "close to full fine-tuning for style/format/domain adaptation (less so for "
                "injecting large amounts of new knowledge)."
            ),
            links=(
                "QLoRA paper (Dettmers et al., 2023): https://arxiv.org/abs/2305.14314",
                "docs/01-qlora-explained.md",
            ),
            tags=("core", "memory"),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="nf4",
            title="Why NF4 and not INT4?",
            body=(
                "NF4 = 4-bit **NormalFloat**: the 16 quantization bins are placed at the "
                "quantiles of a standard normal distribution, so each bin holds an equal "
                "*probability mass* of typical weights. For weights that are approximately "
                "normal (which pre-trained transformers mostly are), this is "
                "information-theoretically optimal among 4-bit codes — strictly better "
                "rounding behaviour than uniform INT4.\n\n"
                "The *compute* dtype stays **bfloat16**: weights are dequantized on the fly "
                "per matrix-multiply tile. The cost is a small kernel overhead, not a precision "
                "catastrophe."
            ),
            links=("https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora",),
            tags=("core",),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="lora_targets",
            title="Which modules get LoRA adapters?",
            body=(
                "`all-linear` attaches adapters to **every linear layer except the output "
                "head**: attention projections, MLP projections, and (in hybrid models like "
                "Qwen3.8) the linear-attention projections too.\n\n"
                "The QLoRA ablations showed that adapting *all* linear layers matters more "
                "than the rank itself: attention-only adapters need a much larger rank to "
                "match the quality of all-linear adapters at r=16.\n\n"
                "**What to expect**: ~0.5–2% of total parameters become trainable. The "
                "printed table below shows the exact split."
            ),
            links=(
                "QLoRA paper, section on adapter placement: https://arxiv.org/abs/2305.14314",
                "docs/03-hyperparameters.md",
            ),
            tags=("core",),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="rank_alpha",
            title="Rank (r) and alpha: what do they control?",
            body=(
                "A LoRA update is `ΔW = (α/r) · B·A` with `B` initialized to zero.\n\n"
                "* **r** is the *capacity* of the adapter — the width of the low-rank "
                "bottleneck. 8 for light format adaptation, 16–32 for domain/style work, "
                "64+ for heavy rewrites.\n"
                "* **α/r** is the *effective step size* of the update. Doubling r while "
                "keeping α fixed halves the update scale — which is why people often think "
                "'higher rank got worse'. Common conventions: α = r, or α = 2r.\n\n"
                "**Rule of thumb**: fix the α/r ratio when sweeping r. On small datasets "
                "(hundreds of examples) prefer *lower* capacity — LoRA then acts as a "
                "regularizer against memorization."
            ),
            links=(
                "Sebastian Raschka's LoRA practical tips: https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms",
                "docs/03-hyperparameters.md",
            ),
            tags=("core",),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="effective_batch",
            title="Effective batch size math",
            body=(
                "`effective_batch = per_device_batch × grad_accum × num_gpus`\n\n"
                "Gradient accumulation computes the *same* gradient as a larger batch "
                "(it averages micro-batch gradients before stepping), so what matters for "
                "optimization is the effective number.\n\n"
                "* Small data (< ~1k examples): effective batch 8–32 keeps the number of "
                "optimizer steps reasonable — with a few hundred examples and effective batch 48 "
                "gives only ~12 steps/epoch, too few for a schedule to mean anything.\n"
                "* Large data: 64–256 is the usual range; larger batches trade gradient "
                "noise for throughput.\n\n"
                "**Per-device batch** is chosen for *memory*; **effective batch** for "
                "*optimization*."
            ),
            links=("docs/03-hyperparameters.md",),
            tags=("core",),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="warmup",
            title="Why warm up the learning rate?",
            body=(
                "Adam's second-moment estimate `v` starts as garbage: with a single sample "
                "it can be orders of magnitude off, and a full-size step through that noise "
                "can wreck early training. A linear warmup over the first ~3–5% of steps "
                "lets the estimates stabilize before the LR reaches its peak.\n\n"
                "**What to expect**: the first ~10–20 steps may show a jagged loss even "
                "with a good setup — that is normal. Judge the trend after warmup ends; the "
                "dashboard marks the warmup window."
            ),
            links=("docs/04-reading-the-plots.md",),
            tags=("core",),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="lr_schedule",
            title="Cosine decay and the end of training",
            body=(
                "The cosine schedule holds LR high through the middle of training and "
                "smoothly decays it to ~0 at the end. Intuition: early steps *explore* "
                "(large steps, fast loss drop), late steps *refine* (tiny steps, stable "
                "convergence).\n\n"
                "**What to expect**: loss drops fastest in the first 20–40% of steps; the "
                "tail looks flat. A flat tail is not failure — it is the schedule doing its "
                "job. Only worry if the *eval* loss starts rising (overfitting)."
            ),
            links=("docs/04-reading-the-plots.md",),
            tags=("core",),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="reading_loss",
            title="How to read the loss",
            body=(
                "**Loss is the average negative log-likelihood per predicted token** "
                "(cross-entropy, natural log). Interpretation: loss L means the model "
                "assigns roughly `e^-L` probability to each correct token on average — "
                "L=1.0 ≈ 37%, L=0.5 ≈ 61%, L=2.0 ≈ 14%.\n\n"
                "* **Starting point**: for a strong instruction model fine-tuning on "
                "in-domain data, expect ~0.5–1.5 at step 0 on *this* kind of data. A start "
                "above ~3 suggests the formatting/chat-template is wrong.\n"
                "* **Noise**: per-step loss is noisy — a batch is a sample, not a "
                "population. Watch the **EMA** (exponentially smoothed) line.\n"
                "* **Direction, not value**: the absolute number matters far less than a "
                "consistent downward trend and a healthy train/eval gap.\n"
                "* **Red flags**: NaN/Inf (aborts automatically), a flat line for many "
                "steps (LR too low or data/template mismatch), spikes that never recover "
                "(LR too high)."
            ),
            links=(
                "docs/04-reading-the-plots.md",
                "https://huggingface.co/docs/trl/sft_trainer",
            ),
            tags=("core",),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="grad_norm",
            title="Gradient norm: the process health monitor",
            body=(
                "Before each optimizer step, all gradients are clipped to a global norm of "
                "`max_grad_norm` (0.3 here — the QLoRA paper value). The *unclipped* norm "
                "is logged: it measures how violent the current gradient is.\n\n"
                "* **Stable training**: norm stays roughly 0.1–2, with occasional spikes.\n"
                "* **Spike > 10× the running median**: one batch disagrees strongly with "
                "the rest — usually a long/odd example. Clipping contains it; the EMA loss "
                "should barely flinch. Repeated spikes → lower LR or lower effective batch.\n"
                "* **Norm collapsing toward 0** while loss is flat: gradients are vanishing "
                "— check that adapters are actually attached and not all frozen."
            ),
            links=("docs/04-reading-the-plots.md",),
            tags=("core",),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="eval",
            title="Train vs eval loss: the overfitting dial",
            body=(
                "A 5% slice of the data is held out and never trained on. Each epoch, the "
                "model is evaluated on it — that curve is the **generalization** "
                "estimate.\n\n"
                "* **Both falling, small gap** → healthy learning.\n"
                "* **Train falling, eval flat/rising** → overfitting: the adapter is "
                "memorizing examples. Remedies: fewer epochs, more dropout, lower rank, "
                "more data.\n"
                "* **Both high and flat** → underfitting or a data/template problem.\n\n"
                "With a few hundred examples the holdout is only a few dozen samples — noisy. Read the *shape*, "
                "not the third decimal."
            ),
            links=("docs/04-reading-the-plots.md",),
            tags=("core",),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="epochs_small_data",
            title="How many epochs on a small dataset?",
            body=(
                "Community consensus (QLoRA paper, Unsloth guides, practitioner reports): "
                "**1–3 epochs** for instruction data, up to ~5 for small, high-quality "
                "sets. Beyond that, memorization dominates.\n\n"
                "With a few hundred examples, 3–5 epochs is a reasonable range *if* eval loss stays "
                "in step with train loss. Watch the epoch summary chart: the useful signal "
                "is per-epoch *eval* delta, not train loss going to zero."
            ),
            links=("docs/03-hyperparameters.md",),
            tags=("core",),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="assistant_only_loss",
            title="Loss masking: only the assistant talks",
            body=(
                "Training on the full rendered conversation would also teach the model to "
                "predict the **user's** tokens — wasted capacity and a source of noise. "
                "With `assistant_only_loss=True`, TRL renders the chat template and sets "
                "labels to `-100` for everything except assistant turns, so gradients flow "
                "only from the answers (including their reasoning blocks).\n\n"
                "This requires the template to expose assistant spans (`{% generation %}` "
                "markers). For known families (Qwen3, Qwen3.5/3.8, Llama 3, …) TRL swaps in "
                "a training template automatically; the startup log shows which "
                "template was used."
            ),
            links=("https://huggingface.co/docs/trl/chat_templates",),
            tags=("core",),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="reasoning_data",
            title="Reasoning traces in the dataset",
            body=(
                "This dataset's assistant turns contain ` thinking…</think>` reasoning "
                "blocks followed by a final answer. Training on them teaches the model a "
                "**thinking style**, not just an answer mapping: the loss includes the "
                "reasoning tokens, which is usually *desirable* for reasoning models — it "
                "preserves and steers the chain-of-thought.\n\n"
                "Qwen3.8 parses ` thinking…</think>` natively in its chat template, so the "
                "traces round-trip exactly as intended. For answer-only "
                "training instead, strip the reasoning blocks in a preprocessing step — but "
                "expect the model to lose some reasoning ability."
            ),
            links=("docs/01-qlora-explained.md",),
            tags=("core",),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="packing",
            title="Sequence packing: throughput vs correctness",
            body=(
                "Packing concatenates several short examples into one full-length sequence "
                "so the GPU never computes on padding. It is a real throughput win on short "
                "data — but it is also a **correctness hazard**: unless the attention "
                "implementation resets at document boundaries (FlashAttention-style "
                "document masks), tokens from different examples can attend to each other "
                "and the model learns to condition on unrelated preceding text.\n\n"
                "**This project defaults to `packing = false`** — with a small dataset, "
                "padding waste costs seconds while boundary bugs cost quality. If enabled, "
                "keep the caveat in mind and prefer it only when the padding fraction is large."
            ),
            links=(
                "https://huggingface.co/docs/trl/sft_trainer (Packing section)",
                "docs/03-hyperparameters.md",
            ),
            tags=("advanced",),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="truncation",
            title="Choosing max_length from the data, not the model",
            body=(
                "The right sequence cap comes from the **token-length distribution** of "
                "the data (the `data` command plots it), not from the model's context "
                "window. Setting it at the p95–p99 drops almost nothing while keeping "
                "batches small.\n\n"
                "Two failure modes: too small silently truncates answers (the model learns "
                "to stop mid-sentence), too large wastes memory on the rare long tail — "
                "every batch pays for the longest example in it. The data profile prints "
                "the exact coverage at the chosen cap."
            ),
            links=("docs/03-hyperparameters.md",),
            tags=("core",),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="prepared_checkpoint",
            title="Why a pre-quantized checkpoint?",
            body=(
                "`load_in_4bit=True` on a **bf16 checkpoint** first materializes the entire "
                "bf16 model in CPU RAM (~2 bytes/param) and only then quantizes it. For a "
                "27B model that is ~55 GB of transient RAM — and with 3 data-parallel "
                "processes it would be needed **three times over**.\n\n"
                "Instead, `rocm-qlora prepare` quantizes **once** into a local NF4 "
                "checkpoint under `.data/models/`. Every training run then loads the "
                "already-4-bit weights directly: small RAM, fast startup, reproducible "
                "quantization. The prepare step also drops the (unused) vision tower and "
                "can re-quantize modules the upstream checkpoint left in 16-bit to fit "
                "24 GB cards."
            ),
            links=(
                "docs/02-amd-rocm-notes.md",
                "https://huggingface.co/docs/bitsandbytes/main/en/non_cuda_backends",
            ),
            tags=("core", "memory"),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="hybrid_attention",
            title="Hybrid attention: DeltaNet + full attention",
            body=(
                "Qwen3.8 is not a plain transformer: 3 of every 4 layers are **Gated "
                "DeltaNet** (linear attention — a recurrent state updated per token, "
                "O(n) memory) and every 4th layer is standard causal attention.\n\n"
                "The fast kernels for these layers live in `flash-linear-attention`, a "
                "Triton package with spotty ROCm support; transformers ships **pure-PyTorch "
                "fallbacks**, which is what this project uses by default — correct, "
                "memory-frugal, somewhat slower on the linear-attention layers. Install "
                "`flash-linear-attention` as an experiment (see docs/02) — stability on "
                "RDNA3 is the open question."
            ),
            links=("docs/02-amd-rocm-notes.md",),
            tags=("advanced",),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="rocm_attention",
            title="Attention on RDNA3: SDPA + AOTriton",
            body=(
                "There is no `flash-attn` package for RDNA3. The working fast path is "
                "PyTorch **SDPA**, which dispatches to **AOTriton** kernels for flash / "
                "memory-efficient attention when the right images are present. On gfx1100 "
                "these images need `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` in some "
                "wheel versions.\n\n"
                "`rocm-qlora doctor` probes each SDPA backend **in a fresh process** and "
                "records which ones actually launch on the current machine — the training launcher "
                "then applies the verdict automatically. If the fused backends fail, "
                "training still works via the math fallback (slower, higher memory)."
            ),
            links=(
                "docs/02-amd-rocm-notes.md",
                "https://rocm.docs.amd.com/en/latest/compatibility/pytorch-compatibility.html",
            ),
            tags=("core",),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="multi_gpu_ddp",
            title="Multiple GPUs: data parallelism, not model splitting",
            body=(
                "Each process holds a **full copy** of the 4-bit model and trains on a "
                "different slice of the data (DDP). Gradients are averaged every step via "
                "RCCL. This is simple, robust, and scales throughput nearly linearly for "
                "small models.\n\n"
                "The catch: **VRAM does not add up across GPUs** — each GPU must fit the "
                "whole model + its activations. (Sharding the model itself is what FSDP "
                "does; `--dist fsdp` is available for that.)\n\n"
                "RCCL runs over PCIe on Radeon cards (no NVLink). For LoRA-sized gradient "
                "payloads this is rarely the bottleneck."
            ),
            links=("docs/02-amd-rocm-notes.md",),
            tags=("core",),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="vram_budget",
            title="Where the VRAM goes",
            body=(
                "Per GPU, roughly:\n\n"
                "`weights (NF4) + embeddings + LoRA weights + optimizer states + activations + logits`\n\n"
                "* **Weights**: ~0.55 bytes/param for quantized Linears; embeddings and "
                "anything outside the skip list stay in 16-bit.\n"
                "* **Optimizer**: AdamW keeps 2 fp32 moments per *trainable* param "
                "(`adamw_8bit` quantizes them to ~1 byte each).\n"
                "* **Activations**: the biggest lever is `max_length` × batch; gradient "
                "checkpointing recomputes them instead of storing them (slower, much "
                "smaller).\n"
                "* **Logits**: the output projection over a 248k vocabulary would "
                "materialize a huge fp32 tensor; TRL's fused chunked head computes "
                "the loss without ever building full logits.\n\n"
                "The preflight table below prints this decomposition for the resolved config."
            ),
            links=("docs/04-reading-the-plots.md",),
            tags=("core", "memory"),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="gpu_health",
            title="GPU health signals on Radeon",
            body=(
                "The dashboard samples `rocm-smi` every few seconds:\n\n"
                "* **Temp (edge/junction)**: < 80 °C fine; 80–95 °C is hot but within "
                "spec; > 95 °C throttle territory.\n"
                "* **Power**: flagship RDNA3 cards top out around 355 W; sustained max power "
                "with high temps is a reason to improve airflow.\n"
                "* **VRAM trend**: small sawtooth is allocator behaviour; a *monotonic "
                "climb* across steps is a leak (usually keeping tensors in the callback "
                "history — check the VRAM chart).\n"
                "* **Utilization**: LoRA training rarely pegs compute to 100% — data "
                "loading and small batches keep it lower. Judge throughput by tokens/s."
            ),
            links=("docs/02-amd-rocm-notes.md",),
            tags=("core",),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="next_steps",
            title="After training: chat, merge, GGUF",
            body=(
                "A run leaves a **LoRA adapter** (a few hundred MB) next to the "
                "frozen base model. Three ways to use it:\n\n"
                "1. `rocm-qlora chat --run <dir>` — load base + adapter and talk to it "
                "(fastest sanity check: does it *behave* differently?).\n"
                "2. `rocm-qlora merge --run <dir>` — fuse adapter into a bf16 model for "
                "serving (vLLM/transformers).\n"
                "3. Convert the merged model to **GGUF** and quantize for `llama.cpp` — "
                "this is what llama.cpp and similar servers run. Commands in the run report.\n\n"
                "Keep the adapter, not just the merged model: adapters compose, weigh "
                "little, and can be re-merged with better bases later."
            ),
            links=("README.md (Deploy section)",),
            tags=("core",),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="seed",
            title="Seeds and reproducibility",
            body=(
                "One seed drives data shuffling/splitting, weight init (adapters), dropout "
                "and dataloader order. On GPUs, exact bitwise reproducibility is *not* "
                "guaranteed (nondeterministic reduction order in some kernels), but "
                "run-to-run variance should be tiny. To A/B a hyperparameter, "
                "change **one** thing and keep the seed."
            ),
            tags=("advanced",),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="chunked_loss",
            title="How the loss is computed (fused chunked head)",
            body=(
                "Naive causal-LM loss materializes the full logits tensor: "
                "`batch × seq × vocab × 4 bytes` fp32 — for 1536 tokens and a 248k "
                "vocabulary that is ~1.5 GB of transient memory *every step*.\n\n"
                "TRL 1.16 patches the model with a **fused LM head**: it runs the "
                "backbone, then projects and scores **only the positions that "
                "contribute to the loss, in vocab chunks**, returning per-token "
                "log-probabilities directly — full logits are never built. This "
                "happens for every loss type (`nll`, `dft`).\n\n"
                "One consequence worth knowing: because the fused head slices the "
                "raw `lm_head.weight` tensor, the output head must stay **16-bit** "
                "— a 4-bit head (flattened uint8 buffer) cannot be sliced."
            ),
            links=("https://huggingface.co/docs/trl/sft_trainer",),
            tags=("advanced", "memory"),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="dft",
            title="Loss flavours: nll vs dft",
            body=(
                "* `nll` — standard next-token cross-entropy. Predictable, comparable "
                "curves, the default everywhere.\n"
                "* `dft` — Dynamic Fine-Tuning (2025): up-weights tokens the model is "
                "*uncertain* about and down-weights easy ones, which research shows "
                "improves generalization on reasoning tasks.\n\n"
                "`dft` curves sit slightly lower and behave a bit differently — use it "
                "deliberately, not by accident. Both are configurable per run."
            ),
            links=("https://huggingface.co/papers/2508.05629",),
            tags=("advanced",),
        ),
        # ------------------------------------------------------------------
        Explainer(
            key="prepare_surgery",
            title="Checkpoint surgery (prepare step)",
            body=(
                "Upstream 4-bit checkpoints are not always *fully* quantized: this one "
                "keeps `in_proj_qkv` (2.5B params!) and the vision tower in 16-bit. "
                "That is ~6 GB of extra weight per GPU in DDP.\n\n"
                "The `prepare` step re-quantizes those to NF4 (same math, far less "
                "memory) and keeps `lm_head` in 16-bit on purpose — TRL's chunked "
                "cross-entropy slices the raw weight tensor, which only works on a "
                "plain tensor."
            ),
            links=("docs/02-amd-rocm-notes.md",),
            tags=("advanced", "memory"),
        ),
    ]
}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_BORDER = {"core": "cyan", "advanced": "magenta"}


class ExplainerBook:
    """Per-run registry of already-shown explainers (dedupe + quiet mode)."""

    def __init__(self, console: Console, enabled: bool = True) -> None:
        self.console = console
        self.enabled = enabled
        self.shown: set[str] = set()

    def show(self, key: str, *, force: bool = False) -> None:
        if not self.enabled or (key in self.shown and not force):
            return
        explainer = EXPLAINERS.get(key)
        if explainer is None:  # pragma: no cover - programming error
            raise KeyError(f"Unknown explainer: {key}")
        self.shown.add(key)
        self.console.print(render_explainer(explainer))

    def hint(self, text: str, *, icon: str = "💡") -> None:
        """One-line technical hint, visually distinct from explainers."""
        if not self.enabled:
            return
        self.console.print(f"[dim]{icon} {text}[/dim]")


def render_explainer(explainer: Explainer) -> Panel:
    body = explainer.body
    if explainer.links:
        links = "\n".join(f"- {link}" for link in explainer.links)
        body += f"\n\n**Read more**\n{links}"
    return Panel(
        Markdown(body),
        title=f"[bold]{explainer.title}[/bold]",
        subtitle=f"[dim]{explainer.level}[/dim]",
        border_style=_BORDER.get(explainer.level, "cyan"),
        padding=(0, 2),
    )


def banner(console: Console, title: str, subtitle: str = "", *, style: str = "bold cyan") -> None:
    """Section banner used to structure long terminal output."""
    console.rule(Text(title, style=style))
    if subtitle:
        console.print(f"[dim]{subtitle}[/dim]")


def key_value_panel(console: Console, title: str, pairs: dict[str, str], *, style: str = "blue") -> None:
    """Compact 'settings' panel — used for resolved run configuration."""
    from rich.table import Table

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()
    for k, v in pairs.items():
        table.add_row(k, v)
    console.print(Panel(table, title=title, border_style=style, padding=(0, 1)))
