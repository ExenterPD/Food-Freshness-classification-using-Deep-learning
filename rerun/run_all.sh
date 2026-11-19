#!/usr/bin/env bash
# Per-machine loops for macOS / Linux. Usage:  bash run_all.sh <step>
#   audit-inspect | audit-split T S | deit | mobilevit | effnet | long <backbone> |
#   cache | sweep | fusion | xgb | metrics | robustness | xai | complexity
set -euo pipefail
cd "$(dirname "$0")"
SEEDS="0 1 2 3 4"
case "${1:-}" in
  audit-inspect) python s00_manifest.py && python s01_audit_split.py --inspect ;;
  audit-split)   python s01_audit_split.py --phash-threshold "$2" --cosine-threshold "$3" ;;
  deit)          caffeinate -i bash -c "for s in $SEEDS; do python s02_train_backbone.py --backbone deit_s --seed \$s; done" ;;
  mobilevit)     caffeinate -i bash -c "for s in $SEEDS; do python s02_train_backbone.py --backbone mobilevit_s --seed \$s; done" ;;
  effnet)        for s in $SEEDS; do python s02_train_backbone.py --backbone effnetv2m --seed $s --amp; done ;;
  long)          python s02_train_backbone.py --backbone "$2" --seed 0 --epochs "$(python -c "import yaml;print(yaml.safe_load(open('config.yaml'))['backbone']['long_epochs'])")" --tag _long ;;
  cache)         python s03_cache_tokens.py ;;
  sweep)         for p in cnn_tr tr_tr; do python s04_train_fusion.py --pair $p --fusion attention --seed 0 --sweep; done ;;
  fusion)        for p in cnn_tr tr_tr; do for s in $SEEDS; do python s04_train_fusion.py --pair $p --fusion all --seed $s; done; done ;;
  xgb)           for p in cnn_tr tr_tr; do
                   for s in $SEEDS; do
                     python s05_xgb_hpo.py --pair $p --fusion concat gated --seed $s --method default
                     python s05_xgb_hpo.py --pair $p --fusion attention --seed $s
                   done
                   python s05_xgb_hpo.py --pair $p --fusion attention --seed 0 --method tpe bohb --opt-seeds 1 2 3 4
                 done ;;
  metrics)       python s06_metrics.py && python s07_stats.py ;;
  robustness)    python s08_robustness.py --make-masks && echo "Check runs/masks/review.html, then run: python s08_robustness.py --evaluate --probe" ;;
  xai)           python s09_xai.py --shap --maps --figures ;;
  complexity)    python s10_complexity.py ;;
  *) echo "usage: bash run_all.sh <step>   (see README.md)"; exit 1 ;;
esac
