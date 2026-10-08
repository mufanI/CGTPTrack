# CGTPTrack paper alignment

Source of truth: `../lunwen/main.tex`, `sections/method.tex`, and
`sections/experiments.tex` (local manuscript, 2026-10-08).

- [x] Rename public modules, classes, configuration directories, commands and
  checkpoint conventions to CGTPTrack / cgtptrack; preserve third-party notices.
- [x] Match the manuscript's reliability-weighted token importance and write
  memory only on scheduled frames whose current reliability exceeds the threshold.
  Share this implementation between training and online tracking.
- [x] Preserve prompt states exactly when their update gates are closed.
- [x] Provide explicit 256-pixel general and GOT-10k-only training configurations;
  distinguish manuscript settings from inherited, unspecified hyperparameters.
- [x] Update installation, README, portable paths and release exclusions.
- [x] Verify naming/imports, configuration loading, CTP numerical behavior,
  branch attention, forward/backward smoke tests where dependencies are available.

The existing BDM-ViT has role-specific Query / Key-Value LayerNorm, shared K/V
projections (when QKV_FIRST_THIRD is false), and independent branch Softmax.
Keep tensor checkpoint parameter names stable where possible. This work does not establish that
changed code reproduces the manuscript's benchmark numbers. No remote publication
is included. The user requested the source folder name CGTPTrack and removal of non-paper
module configurations. Maintain the new named directory and produce its clean
source-only ZIP after verification.
