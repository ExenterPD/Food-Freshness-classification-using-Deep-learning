"""End-to-end smoke test on a tiny synthetic dataset (random weights, CPU-friendly).

    python smoke_test.py

Builds Foods/runs_smoke/ (outside the repo), generates fake Good/Bad images in the UC-FCD folder layout
(including near-duplicates), and runs every stage with tiny settings. Use it to check an environment
before the real runs. It does not download pretrained weights and its numbers mean nothing.
"""
import os; os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')  # torch 2.0 on Apple GPUs: run missing ops on CPU (must precede `import torch`)
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent / 'runs_smoke'   # Foods/runs_smoke, outside the repo


def make_data(root, n_per=12):
    rng = np.random.RandomState(0)
    groups = {'Fish': ['Shrimp', 'Rohu'], 'Rice': [''], 'Egg': ['']}
    for g, subs in groups.items():
        for s in subs:
            for lab, colour in (('Good', (60, 170, 60)), ('Bad', (150, 90, 40))):
                d = root / g / s / lab if s else root / g / lab
                d.mkdir(parents=True, exist_ok=True)
                for i in range(n_per):
                    a = np.clip(rng.normal(colour, 40, (96, 96, 3)), 0, 255).astype(np.uint8)
                    im = Image.fromarray(a)
                    im.save(d / f'img{i}.jpg')
                    if i == 0:  # a near duplicate and an exact duplicate
                        im.filter(ImageFilter.GaussianBlur(0.5)).save(d / 'img0_near.jpg')
                        im.save(d / 'img0_copy.png')
                        shutil.copy(d / 'img0.jpg', d / 'img0 - Copy.jpg')   # exact byte copy


def run(*args):
    print('\n$', ' '.join(args), flush=True)
    subprocess.run([sys.executable, *args, '--config', str(ROOT / 'config.yaml')], cwd=HERE, check=True)


def main():
    if ROOT.exists():
        shutil.rmtree(ROOT)
    make_data(ROOT / 'data')
    cfg = yaml.safe_load(open(HERE / 'config.yaml'))
    cfg.update(data_root='data', work_dir='runs', seeds=[0, 1])
    cfg['backbone'].update(epochs=2, long_epochs=3, batch_size=8)
    cfg['fusion'].update(d=32, heads=4, mlp_hidden=16, epochs=3, batch_size=16, sweep_d=[16, 32], sweep_heads=[2, 4])
    cfg['xgb'].update(b_min=5, b_max=20, total_rounds=120, hpo_opt_seeds=[0, 1])
    cfg['metrics']['bootstrap'] = 50
    cfg['robustness'].update(mask_method='grabcut', brightness=[0.7], contrast=[0.7], blur_sigma=[1.0])
    cfg['xai'].update(n_per_stratum=1, deletion_steps=4)
    cfg['complexity'].update(warmup=1, timed_runs=2)
    yaml.safe_dump(cfg, open(ROOT / 'config.yaml', 'w'))
    shutil.copy(HERE / 'config.yaml', ROOT / 'config.reference.yaml')
    # scripts resolve paths relative to the config file
    run('s00_manifest.py', '--workers', '2')
    run('s01_audit_split.py', '--inspect', '--no-pretrained')
    run('s01_audit_split.py', '--phash-threshold', '4', '--cosine-threshold', '0.9', '--no-pretrained')
    for b in ['effnetv2m', 'mobilevit_s', 'deit_s']:
        for s in (0, 1):
            run('s02_train_backbone.py', '--backbone', b, '--seed', str(s), '--no-pretrained')
    run('s03_cache_tokens.py')
    run('s04_train_fusion.py', '--pair', 'cnn_tr', '--fusion', 'attention', '--seed', '0', '--sweep')
    for p in ('cnn_tr', 'tr_tr'):
        for s in (0, 1):
            run('s04_train_fusion.py', '--pair', p, '--fusion', 'all', '--seed', str(s))
            run('s05_xgb_hpo.py', '--pair', p, '--fusion', 'concat', 'gated', '--seed', str(s), '--method', 'default')
            run('s05_xgb_hpo.py', '--pair', p, '--fusion', 'attention', '--seed', str(s))
        run('s05_xgb_hpo.py', '--pair', p, '--fusion', 'attention', '--seed', '0', '--method', 'tpe', 'bohb', '--opt-seeds', '1')
    run('s06_metrics.py')
    run('s07_stats.py')
    run('s08_robustness.py', '--make-masks', '--evaluate', '--probe', '--no-pretrained')
    run('s09_xai.py', '--shap', '--maps', '--figures')
    run('s10_complexity.py')
    print('\nSMOKE TEST PASSED. Outputs in', ROOT / 'runs')


if __name__ == '__main__':
    main()
