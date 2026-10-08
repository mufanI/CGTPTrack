# Paper alignment and release status

The source of truth is the authors' local `lunwen` manuscript, specifically
`main.tex`, `sections/method.tex`, and `sections/experiments.tex`.
This is a revision of an older local implementation. The latest server code,
trained weights, and benchmark logs were not available for comparison.

## Method-to-code mapping

| Manuscript component | Implementation | Alignment |
| --- | --- | --- |
| BDM-ViT / BDMA | `lib/models/cgtptrack/vit.py`, `base_backbone.py` | Search-only queries; read-only memory; separate search/memory Softmax; explicit mixing coefficient; separate Query and KV LayerNorm; shared K/V source projections |
| Dual prompt initialization and fusion | `confidence_temporal_prompt.py` | Learnable short/long states; normalized 0.7/0.3 fusion |
| Warm-response reliability | `ConfidenceTemporalPrompt._estimate_reliability` | Sigmoid warm logits; peak, normalized spatial entropy and top-two gap weighted 0.45/0.35/0.20 |
| Final response | `ConfidenceTemporalPrompt.forward` | Original logits plus alpha × reliability × tanh similarity; no double addition |
| Prompt updates | `ConfidenceTemporalPrompt.forward` | GRU candidate, separate reliability gates/rates, long-branch stop-gradient; closed gates preserve the previous state exactly |
| Token importance | `ConfidenceTemporalPrompt.token_importance` | `(1-r)*raw + r*refined*(1+bias)/2` |
| Memory candidate | Model and tracker | Final normalized search tokens plus embedding of `importance > 0.5` |
| Memory writing | `ConfidenceTemporalPrompt.update_tcm_memory` | Only scheduled frames with current reliability above the memory threshold; TopK history ranked by current importance-weighted memory attention, followed by current tokens |
| Training objective | `lib/train/actors/cgtptrack.py` | Focal + 5 × L1 + 2 × GIoU |
| Learning rates | `lib/train/base_functions.py` | New KV LayerNorm and head/CTP parameters use 1e-4; pretrained backbone parameters use 1e-5 |

Training and online tracking share the same token-importance and memory-write
functions. Training schedules a write at each sampled search (interval 1);
inference uses the configured interval. At other frames, no candidate is cached
for later writing. Historical interval averaging, best-frame selection, and
low-confidence fallback writes have been removed from the main tracker.

For batches with different accepted write histories, zero padding is accompanied
by a boolean validity mask. Padded keys are excluded from memory Softmax and
TopK. Rejected samples retain exactly their existing valid tokens. A single-video
tracker never needs padding. Memory capacity is at most retained history tokens
(one search-grid size) plus one current search-grid, i.e. 512 tokens at 256 pixels
and stride 16. The mixing coefficient controls attention strength, not capacity.

The attention returned for memory ranking is the separately normalized branch
probability before mixing/dropout; feature aggregation still uses the configured
mixing coefficient. This also avoids degenerate rankings at coefficient zero.

## Two maintained experiment configurations

Only `experiments/cgtptrack/baseline_full.yaml` and `baseline.yaml` are retained.
Both use the manuscript's 256 × 256 template/search inputs, crop factor 4,
one template plus four searches, maximum sample interval 200, 15,000 training
samples per epoch, batch size 8, 300 epochs, LR 1e-4, backbone multiplier 0.1,
decay at epoch 240, weight decay 1e-4, gradient clipping 0.1, drop path 0.1,
and disabled AMP.

`baseline_full` uses LaSOT, GOT-10k, COCO17 and TrackingNet with equal 1:1:1:1
weights. Its GOT-10k split remains the local `GOT10K_vottrain` split.
`baseline` uses only `GOT10K_train_full`; validation uses the disjoint official
validation directory rather than the VOT subset of the training set.

Only the paper architecture is exposed. Configuration switches and corresponding
main-path branches for prompt-as-KV tokens, source-specific K/V specialization,
score-drop/bounding-box-stability gates, recovery/search expansion, candidate
elimination, alternate backbones, extra mergers and distillation are removed.
BDMA role-specific norms, shared source K/V projections, CTP response refinement
and prompt-aware memory importance are always used. The single
`MODEL.BACKBONE.MEMORY_LAMBDA` controls both training and inference for each
protocol, replacing the historical dataset-dependent memory-gate settings.

The manuscript does not specify every numeric CTP threshold or memory interval.
These local defaults must be compared with the latest server configuration:

| Setting | baseline_full | baseline |
| --- | --- | --- |
| Alpha / temperature | 0.30 / 0.07 | 0.50 / 0.07 |
| Short / long reliability threshold | 0.50 / 0.75 | 0.20 / 0.55 |
| Short / long update rate | 0.10 / 0.015 | 0.25 / 0.03 |
| Memory-write reliability threshold | 0.50 | 0.20 |
| Inference memory-write interval | 400 | 10 |
| Training memory mixing coefficient | 0.15 | 0.35 |

The memory-write threshold explicitly follows the local short-prompt threshold;
it is a configurable implementation choice, not a numeric value supplied by the
manuscript. GOT-10k inference uses mixing coefficient 0.35, matching the paper's
reported setting. General inference keeps the local 0.15 setting in
`baseline_full`. This does not establish equality with the server implementation.

## Naming and checkpoints

Public namespaces, classes, training actors, tracker/parameter modules, experiment
paths and checkpoint filenames use `CGTPTrack` / `cgtptrack`. The main model class
is `CGTPTrack`, its builder is `build_cgtptrack`, and its actor is `CGTPTrackActor`.
No old namespace alias is provided. Tensor parameter names are retained where
possible, but state-update behavior has changed and old weights are not certified
for the revised method. Main evaluation checks checkpoint keys strictly.

The maintained source directory and release ZIP root are both `CGTPTrack/`.
Windows denied renaming the original directory, so the new named directory was
created without copying Git metadata, logs or caches. The previous directory and
the original ZIP backup remain outside this release. Historical comparison scripts,
alternate trackers, obsolete configurations, and unused distillation/CE entry
points are removed from the maintained source.

## Verification and limits

The local CPU validation environment uses Python 3.12 and PyTorch 2.4.1+cpu,
torchvision 0.19.1+cpu, timm 0.9.2 and NumPy 1.26.4. It is distinct from the
paper's Python 3.8.20 / RTX 4090 training environment. `tests/` contains numerical
regressions, configuration checks, attention checks and a real ViT-Base
forward/backward smoke test with reduced spatial input, a 256-pixel forward test,
and the real online tracker loop on CPU (CUDA placement and checkpoint disk I/O
are bypassed only in that test). Final test results are
recorded in `validation.txt` after running the suite.

No training run or benchmark evaluation has been performed for this revision.
The latest server implementation, checkpoint provenance, and complete benchmark
results must be reconciled before advertising this archive as a reproduction of
the reported accuracy. Original third-party copyright and license notices remain.
