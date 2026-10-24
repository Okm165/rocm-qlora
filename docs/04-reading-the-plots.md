# 04 · Reading the plots

Every chart lives in `<run>/charts/` and is regenerated **while training
runs** (overview + loss every `chart_snapshot_steps`; the full set at every
epoch boundary and at the end). The per-step progress line shows the same numbers
live. This doc is the decoder ring.

## loss.png — the primary signal

* **Faint line** = per-step loss (one optimizer step). A step is one *sample*
  of the loss, not the population mean — noise is expected.
* **Bold line** = EMA(20). The trend is read from this line.
* **Shaded region** = warmup (LR ramping from 0). Early jaggedness here is
  normal; don't judge before it ends.
* **Vertical dotted lines** = epoch boundaries. A small step-down at each
  boundary often reflects fresh data order, not a real regime change.
* **Gold dot** = best EMA. If the best point is at the very end, training was
  still improving — more epochs would likely help.
* **Orange dashed** = eval loss on the 5% holdout.

Interpretation table:

| shape | meaning | action |
|---|---|---|
| smooth descent, flattening tail | textbook | none |
| descent then plateau mid-run | LR decay doing its job | none |
| sawtooth up | LR too high / grad noise | ↓ lr, ↑ grad_accum |
| flat from step 0 | no learning: template bug, LR too low, or adapters not attached | check the model lines, ↑ lr |
| sudden cliff down | a hard subset learned / schedule step | inspect per-epoch means |
| slow climb at the end | overfitting or schedule ended early | watch the eval markers |

## The eval markers — the overfitting dial

The orange dashed points on loss.png are the holdout evaluations (epochs.png
summarizes them per epoch). Rules:

* both falling, gap stable → healthy generalization.
* train falling, eval flat/rising → memorization. Cut epochs or raise dropout.
* both flat → underfitting (↑ lr) or data/template problem.

With a small holdout (a few dozen examples), the curve is noisy by
construction: read the **direction between epochs**, not the wiggle.

## lr.png — the schedule

A triangular warmup then a smooth cosine decay to ~0. The shaded warmup
region corresponds to the shaded region in loss.png. If the final LR doesn't
approach 0, the schedule didn't finish — check that `max_steps` wasn't hit
early.

## grad_norm.png — process health

Unclipped gradient norm per step (the *clipped* value is what the optimizer
sees; `max_grad_norm` is the dashed line).

* stable band (0.1–2) with occasional spikes → healthy; clipping contains
  outliers.
* a spike >10× the rolling median → one odd batch (long example, weird
  sample). The EMA loss should barely flinch; repeated spikes mean ↓ lr.
* sustained rise across steps → the model is entering a high-curvature
  region; ↓ lr or ↑ warmup.

## throughput.png — tokens/s (+ cumulative tokens)

Expect a stable plateau after the first steps (which include dataloader
warmup). A slow downward drift suggests thermal throttling (cross-check
gpu_health.png) or memory fragmentation. Cumulative tokens × epochs indicates
how much data the model actually saw — the honest denominator for any claim
about learning.

## vram.png — per-GPU memory

* small sawtooth → allocator behavior, normal.
* monotonic climb across many steps → leak. Usual suspects: callbacks holding
  tensors (this project's callback stores only scalars/dicts), or a
  non-empty `torch.cuda` cache growth pattern.
* step change upward at an epoch boundary → evaluation buffers.

## gpu_health.png — temperature and power

Typical edge temperatures: idle ~45 °C, training 55–75 °C is typical,
85 °C is warm-but-fine, 95 °C+ is throttle territory (the model clock will
drop and throughput.png will show it). Package power peaks ~355 W; a GPU
pinned at max power *and* >90 °C wants better airflow. The chart draws one
line at **90 °C** — the same threshold the run's `!` warning fires at.

## epochs.png — per-epoch bars

Train mean vs eval per epoch. The most useful derived quantity: the
**eval delta** between consecutive epochs. Improving → keep going; flat →
schedule is fine as-is; rising → stop earlier next time (or add data).

## overview.png — the live 2×2 snapshot

Loss, LR, throughput, VRAM in one image — the "is it healthy at a glance"
view. It updates during training, viewable in an image viewer to watch it
evolve.

## token_lengths.png (data profile, `rocm-qlora data`)

Histogram of chat-templated token lengths with p50/p90/p99 markers and the
chosen `max_length`; the right subplot shows coverage vs candidate caps. Set
the cap where the coverage curve saturates — for typical Q&A data, around
1.5k tokens.

## What "good" looks like for a small dataset (a few hundred examples, 3 epochs)

* loss: starts ~1.5–2.0 → ends ~1.0–1.4 (domain Q&A with reasoning traces)
* eval: tracks train within ~0.1–0.15
* grad norm: mostly 0.2–1.5, occasional 3–5 spikes
* throughput: hundreds to low-thousands of tokens/s per GPU with batch 1 at
  1536 tokens (small batches underutilize the GPU — expected for QLoRA)

If the numbers fall inside those bands and the trend arrows agree, the run is
healthy. The progress-line warnings summarize exactly these
checks.
