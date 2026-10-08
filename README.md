# CGTPTrack

Code for **CGTPTrack: Branch-Decoupled Historical Retrieval and Reliability-Guided Temporal Prompts for Visual Object Tracking**.

CGTPTrack combines BDM-ViT (branch-independent search/memory attention with role-specific LayerNorm) and CTP (dual temporal prompts, response reliability, response refinement, and gated historical-memory writing).

This local revision has been aligned with the manuscript. The authors' latest training code and benchmark results reside on their server and have not been synced into this revision. Model weights are not included. The manuscript's reported scores are **not verified results of this revised local code**. See [paper alignment and validation](docs/paper-alignment.md).

## Installation

The manuscript uses Python 3.8.20 and PyTorch 2.4.1. Training and online tracking require a CUDA GPU. Run from the repository root on Linux:

```bash
conda create -n cgtptrack python=3.8.20
conda activate cgtptrack
bash install.sh
```

`jpeg4py` also requires the system libjpeg-turbo library. CPU-only component tests can use `TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu bash install.sh`.

## Data and paths

Place datasets under `data/lasot`, `data/got10k/{train,val,test}`, `data/coco/{annotations,images}`, and `data/trackingnet/{TRAIN_0,...,TRAIN_11,TEST}`. Generate local paths with:

```bash
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

Check `lib/train/admin/local.py` and `lib/test/evaluation/local.py` for your dataset locations, including the LaSOT extension subset. These machine-specific files are excluded from source control.

Place the MAE ViT-Base initialization checkpoint `mae_pretrain_vit_base.pth` in `pretrained_models/`. Obtain it from the upstream MAE project; it is a backbone initialization, not a trained CGTPTrack checkpoint.

## Training

Only the paper modules are configurable; recovery, candidate elimination,
distillation and historical extra-gating variants have been removed.
Only two experiment configurations are maintained:

| Configuration | Training data | Intended evaluation |
| --- | --- | --- |
| `baseline_full` | LaSOT, GOT-10k training subset, COCO17, TrackingNet; weights 1:1:1:1 | LaSOT, LaSOT extension, TrackingNet |
| `baseline` | GOT-10k training set only | GOT-10k official one-shot protocol |

```bash
python tracking/train.py --script cgtptrack --config baseline_full --save_dir ./output --mode single --use_wandb 0
python tracking/train.py --script cgtptrack --config baseline --save_dir ./output --mode single --use_wandb 0
```

Both use 256-pixel crops, one template and four searches, batch size 8, 15,000 samples per epoch, and 300 epochs with learning-rate decay at epoch 240, as described in the manuscript. Hyperparameters not numerically specified in the manuscript remain explicitly documented as local defaults.

## Evaluation

Expected checkpoints:

```text
output/checkpoints/train/cgtptrack/baseline_full/CGTPTrack_ep0300.pth.tar
output/checkpoints/train/cgtptrack/baseline/CGTPTrack_ep0300.pth.tar
```

```bash
python tracking/test.py cgtptrack baseline_full --dataset_name lasot --threads 0 --num_gpus 1
python tracking/test.py cgtptrack baseline_full --dataset_name lasot_extension_subset --threads 0 --num_gpus 1
python tracking/test.py cgtptrack baseline_full --dataset_name trackingnet --threads 0 --num_gpus 1
python tracking/test.py cgtptrack baseline --dataset_name got10k_test --threads 0 --num_gpus 1
python tracking/analysis_results.py
```

The epoch comes from `TEST.EPOCH`; `--runid` identifies a result run and does not select a checkpoint epoch. Renaming an old checkpoint file does not establish that it is compatible with this implementation. Main evaluation loads matching model parameters strictly, so incomplete weights cannot silently initialize CTP.

## Validation

```bash
python -m pytest tests -q
```

The tests cover mathematical components and synthetic model execution. Benchmark reproduction additionally requires server code comparison, trained weights, datasets, and a complete evaluation.

## Attribution and license

This repository builds on the tracking framework by Chenlong Xu and the OSTrack framework by Botao Ye and collaborators, with ViT components derived from timm. Upstream copyright and MIT license notices are preserved in [LICENSE](LICENSE) and the source files. Project naming does not replace these attributions. No publication status, author list, DOI, or weight-download link is asserted in this working release.
