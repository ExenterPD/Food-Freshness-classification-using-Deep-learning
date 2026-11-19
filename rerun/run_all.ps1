# Windows (RTX 3050) loops. Usage in PowerShell from this folder:
#   .\run_all.ps1 effnet           # EfficientNetV2-M, seeds 0-4, mixed precision
#   .\run_all.ps1 effnet-small     # same with batch 8 x 2 accumulation (if CUDA runs out of memory)
#   .\run_all.ps1 long             # one extended run, seed 0
#   .\run_all.ps1 cache            # token caching once all checkpoints are copied here
#   .\run_all.ps1 sweep            # choose fusion d and heads (seed 0)
#   .\run_all.ps1 fusion           # all fusion runs, both pairs, seeds 0-4
#   .\run_all.ps1 complexity
param([string]$step)
Set-Location $PSScriptRoot
$seeds = 0..4
switch ($step) {
  'effnet'       { foreach ($s in $seeds) { python s02_train_backbone.py --backbone effnetv2m --seed $s --amp; if ($LASTEXITCODE) { exit 1 } } }
  'effnet-small' { foreach ($s in $seeds) { python s02_train_backbone.py --backbone effnetv2m --seed $s --amp --batch 8 --accum 2; if ($LASTEXITCODE) { exit 1 } } }
  'long'         { $e = python -c "import yaml;print(yaml.safe_load(open('config.yaml'))['backbone']['long_epochs'])"; python s02_train_backbone.py --backbone effnetv2m --seed 0 --amp --epochs $e --tag _long }
  'cache'        { python s03_cache_tokens.py }
  'sweep'        { foreach ($p in 'cnn_tr','tr_tr') { python s04_train_fusion.py --pair $p --fusion attention --seed 0 --sweep } }
  'fusion'       { foreach ($p in 'cnn_tr','tr_tr') { foreach ($s in $seeds) { python s04_train_fusion.py --pair $p --fusion all --seed $s; if ($LASTEXITCODE) { exit 1 } } } }
  'complexity'   { python s10_complexity.py }
  default        { Write-Host 'usage: .\run_all.ps1 effnet | effnet-small | long | cache | sweep | fusion | complexity' }
}
