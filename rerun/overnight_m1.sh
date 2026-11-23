#!/bin/bash
# Overnight run on the M1 (T7 attached). Usage, from this folder:
#   caffeinate -dims bash overnight_m1.sh
# Each step is logged to overnight_<time>.log. Part A rebuilds seed 3 from the retrained MobileViT-S
# (cache -> fusion -> XGBoost). If any part of A fails, part B runs without seed 3.
cd "$(dirname "$0")" || exit 1
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1
LOG="overnight_$(date +%Y%m%d_%H%M).log"
STATUS=(); OK3=1

run() {
  echo "=== $(date '+%H:%M:%S') START: $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  echo "=== $(date '+%H:%M:%S') END (exit $rc): $*" | tee -a "$LOG"
  STATUS+=("exit $rc  $*")
  return $rc
}

# ---- A. rebuild seed 3 (old fusion/XGBoost for seed 3 used the half-trained MobileViT tokens)
run python s03_cache_tokens.py --backbone effnetv2m deit_s mobilevit_s --seeds 3 --batch 16 || OK3=0
if [ $OK3 = 1 ]; then
  for p in cnn_tr tr_tr; do
    run python s04_train_fusion.py --pair $p --fusion all --seed 3 || OK3=0
  done
fi
if [ $OK3 = 1 ]; then
  for p in cnn_tr tr_tr; do
    run python s05_xgb_hpo.py --pair $p --fusion concat gated --seed 3 --method default || OK3=0
    run python s05_xgb_hpo.py --pair $p --fusion attention --seed 3 || OK3=0
  done
fi
if [ $OK3 = 1 ]; then SEEDS="0 1 2 3 4"; else SEEDS="0 1 2 4"; echo "!!! seed 3 rebuild failed; continuing without seed 3" | tee -a "$LOG"; fi

# ---- B. everything else
run python s06_metrics.py                                   # Step 5
run python s07_stats.py                                     # Step 5
run python s10_complexity.py                                # Step 8, M1 row
run python s08_robustness.py --make-masks                   # Step 6a (rembg, every image)
run python s08_robustness.py --probe                        # Step 6b, background probe
run python s08_robustness.py --evaluate --seeds $SEEDS      # Step 6b
run python s09_xai.py --shap --maps --figures --seeds $SEEDS   # Step 7

echo "" | tee -a "$LOG"
echo "===== SUMMARY ($(date '+%H:%M:%S')), seeds used: $SEEDS =====" | tee -a "$LOG"
for s in "${STATUS[@]}"; do echo "$s" | tee -a "$LOG"; done
