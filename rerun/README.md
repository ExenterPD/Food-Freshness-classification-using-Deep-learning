# Revised pipeline (reviewer rerun)

This folder replaces the training, optimisation and SHAP scripts in `Model_Code_Final/` and
`SHAP_Code/`, which are kept for reference only. The revised manuscript describes this pipeline.

What changed relative to the old scripts:

| Problem in the old code | Fixed in |
| --- | --- |
| Attention received a single token, so it reduced to a linear map | `models.py` `Fusion` (attention over all 98 / 245 patch tokens) |
| XGBoost was trained on pre-attention features | `s04_train_fusion.py` saves the pooled attention output `z` |
| Train augmentation was overwritten by the validation transform | `common.make_dataset` (one dataset object per split) |
| Plain random split, possible near-duplicate leakage | `s01_audit_split.py` (duplicate clusters, stratified group split, frozen `split.csv`) |
| BOHB call could not run; fallbacks were silently used | `s05_xgb_hpo.py` (nameserver + worker, matched budget with TPE) |
| Model 1 and Model 2 refit XGBoost differently | `s05_xgb_hpo.py` (one rule: refit on train+val) |
| Weighted recall reported as TPR; FPR = 1 − accuracy | `common.binary_metrics`, `s06_metrics.py` (Fresh = positive class) |
| No multi-seed loop or statistics | seeds in `config.yaml`, `s07_stats.py` |
| SHAP explained a surrogate model | `s09_xai.py` (TreeSHAP on the deployed XGBoost) |

## Layout

```
Foods/
  data/Master Dataset/          <- unzipped UC-FCD (not in git)
  runs/                         <- every output (not in git)
  Food-Freshness-classification-using-Deep-learning/
    rerun/                      <- this folder
```

Paths are set in `config.yaml` relative to that file. Values marked `CONFIRM` are choices to
settle before the final runs.

## Setup (every machine)

```bash
python3.10 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Windows + RTX 3050: install the CUDA build of PyTorch first, e.g.
#   pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install rembg                                            # only on the machine that makes masks
python smoke_test.py                                         # ~10-30 min on CPU; must end with SMOKE TEST PASSED
pip freeze > env_<machine>.txt
```

## Run order

All commands run from this folder. `--seed` / `--seeds` default to `config.yaml: seeds`.

| Step | Machine | Command |
| --- | --- | --- |
| 1a | M1 | `python s00_manifest.py` (prints the category tree for the paper) |
| 1b | M1 | `python s01_audit_split.py --inspect`, open `runs/audit/closest_pairs.html` |
| 1c | M1 | `python s01_audit_split.py --phash-threshold T --cosine-threshold S`, then copy `runs/split/split.csv` to the other machines |
| 2 | RTX 3050 | `python s02_train_backbone.py --backbone effnetv2m --seed S --amp` for S = 0..4 (`--batch 8 --accum 2` if out of memory) |
| 2 | M3 #1 | `python s02_train_backbone.py --backbone deit_s --seed S` for S = 0..4 |
| 2 | M3 #2 | `python s02_train_backbone.py --backbone mobilevit_s --seed S` for S = 0..4 |
| 2 (conv.) | each | `python s02_train_backbone.py --backbone <b> --seed 0 --epochs <long_epochs> --tag _long` |
| 3 | RTX 3050 | copy every `runs/backbones/*/seed*/` folder to one machine, then `python s03_cache_tokens.py` |
| 4a | RTX 3050 | `python s04_train_fusion.py --pair cnn_tr --fusion attention --seed 0 --sweep` (and `tr_tr`), set `fusion.d`/`heads` |
| 4a | RTX 3050 | `python s04_train_fusion.py --pair {cnn_tr,tr_tr} --fusion all --seed S` for S = 0..4 |
| 4b | M1 | `python s05_xgb_hpo.py --pair P --fusion concat gated --seed S --method default` |
| 4b | M1 | `python s05_xgb_hpo.py --pair P --fusion attention --seed S` (default, TPE, BOHB) |
| 4b | M1 | `python s05_xgb_hpo.py --pair P --fusion attention --seed 0 --method tpe bohb --opt-seeds 0 1 2 3 4` |
| 5 | any | `python s06_metrics.py` then `python s07_stats.py` |
| 6 | M3 #1 | `python s08_robustness.py --make-masks`, check `runs/masks/review.html`, then `--evaluate --probe` |
| 7 | M3 #2 | `python s09_xai.py --shap --maps --figures` |
| 8 | every device | `python s10_complexity.py` |

`run_all.sh` / `run_all.ps1` contain the per-machine loops.

Steps 3 onwards need the checkpoints of all three backbones in one `runs/` folder; copy the
`runs/backbones/` subfolders from the other machines first. The token cache (`runs/cache/`) is
large, so keep it on the PC and run Step 4a there; the other steps need only `runs/backbones/`,
`runs/fusion/` (embeddings and models), `runs/xgb/` and the `*_index.csv` files in `runs/cache/`. Keep `runs/split/split.csv` identical
everywhere.

## Outputs used in the paper (`runs/results/`)

| File | Paper |
| --- | --- |
| `split/audit_report.csv`, `split/audit_summary.csv` | Table 2, Table 8, Section 5.1 |
| `table_backbones.csv`, `backbones/*/seed*/curves.png` | Table 9, Figure 1 |
| `table_ablation.csv` | Table 10 |
| `table_final.csv` | Table 11 |
| `table_category.csv`, `category_summary.csv` | Table 12 |
| `table_robustness.csv`, `background_probe.csv` | Table 13 |
| `table_hpo.csv`, `figure_hpo_convergence.png` | Table 14, Figure 4 |
| `complexity.csv` | Table 15 |
| `table_stats.csv`, `stats.csv`, `mcnemar_detail.csv` | Table 16 |
| `table_xai.csv`, `shap_summary.csv`, `shap/`, `xai/` | Table 17, Figures 5–6 |
| `consolidated_metrics.csv` | every other number (one row per run) |
