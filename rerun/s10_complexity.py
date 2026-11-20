"""Step 8. Parameters, compute, size, latency, throughput and peak memory on THIS device.

    python s10_complexity.py [--seed 0]

Run it on every machine that trained or will serve a model; rows are appended to
runs/results/complexity.csv with the device name. GMACs are counted by fvcore
(one multiply-accumulate = 1). Latency = whole pipeline at batch 1 after warm-up passes;
throughput at batch 16. Hybrid latency includes both backbones, the fusion module and XGBoost.
"""
import os; os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')  # torch 2.0 on Apple GPUs: run missing ops on CPU (must precede `import torch`)
import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import PAIRS, PAIR_NAMES, device_name, get_device, load_config, sync, wd
from models import Pipeline, backbone_ckpt, fusion_dir, load_backbone, total_params, trainable_params, xgb_dir


def gmacs(fn, x):
    try:
        from fvcore.nn import FlopCountAnalysis

        class Wrap(torch.nn.Module):
            def __init__(self, f):
                super().__init__(); self.f = f

            def forward(self, x):
                return self.f(x)
        a = FlopCountAnalysis(Wrap(fn), x); a.unsupported_ops_warnings(False); a.uncalled_modules_warnings(False)
        return a.total() / 1e9
    except Exception as e:
        print('fvcore failed:', e); return float('nan')


def peak_reset(dev):
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()


def peak_mb(dev):
    if dev.type == 'cuda':
        return torch.cuda.max_memory_allocated() / 2 ** 20
    if dev.type == 'mps':
        try:
            return torch.mps.driver_allocated_memory() / 2 ** 20
        except AttributeError:
            return float('nan')
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2 ** 20 if platform.system() == 'Darwin' else r / 1024
    except ImportError:  # Windows
        import psutil
        return psutil.Process().memory_info().peak_wset / 2 ** 20


def timeit(fn, x, dev, warm, runs):
    with torch.no_grad():
        for _ in range(warm):
            fn(x)
        sync(dev); t = []
        for _ in range(runs):
            t0 = time.perf_counter(); fn(x); sync(dev); t.append(time.perf_counter() - t0)
    return float(np.median(t))


def size_mb(paths):
    return sum(Path(p).stat().st_size for p in paths if Path(p).exists()) / 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--device', default='auto')
    a = ap.parse_args()
    cfg = load_config(a.config); dev = get_device(a.device); cc = cfg['complexity']
    x1 = torch.randn(1, 3, cfg['image_size'], cfg['image_size'], device=dev)
    x16 = torch.randn(16, 3, cfg['image_size'], cfg['image_size'], device=dev)
    rows = []
    for b in ['effnetv2m', 'mobilevit_s', 'deit_s']:
        if not backbone_ckpt(cfg, b, a.seed).exists():
            continue
        m = load_backbone(cfg, b, a.seed, dev)
        peak_reset(dev)
        lat = timeit(m, x1, dev, cc['warmup'], cc['timed_runs'])
        thr = 16 / timeit(m, x16, dev, cc['warmup'], max(5, cc['timed_runs'] // 5))
        rows.append(dict(model=b, total_params=total_params(m), trainable_in_fusion_stage='—',
                         gmacs=gmacs(m.cpu(), x1.cpu()), size_mb=size_mb([backbone_ckpt(cfg, b, a.seed)]),
                         latency_ms=lat * 1000, throughput_img_s=thr, peak_memory_mb=peak_mb(dev),
                         device=device_name(dev)))
        print(rows[-1], flush=True)
    for p in PAIRS:
        try:
            pipe = Pipeline(cfg, p, 'attention', a.seed, dev)
        except FileNotFoundError:
            continue
        fn = lambda x: pipe.prob_fresh(x)
        peak_reset(dev)
        lat = timeit(fn, x1, dev, cc['warmup'], cc['timed_runs'])
        thr = 16 / timeit(fn, x16, dev, cc['warmup'], max(5, cc['timed_runs'] // 5))
        bbs = PAIRS[p]
        nets_params = total_params(pipe.a) + total_params(pipe.b) + total_params(pipe.fusion)
        pipe_cpu = Pipeline(cfg, p, 'attention', a.seed, torch.device('cpu'), classifier='mlp')
        files = [backbone_ckpt(cfg, b, a.seed) for b in bbs] + [fusion_dir(cfg, p, 'attention', a.seed) / 'best.pt',
                 xgb_dir(cfg, p, 'attention', a.seed, cfg['xgb']['final_method'], a.seed) / 'model.json']
        fus_train = sum(q.numel() for n_, q in pipe.fusion.named_parameters())
        rows.append(dict(model=PAIR_NAMES[p], total_params=nets_params, trainable_in_fusion_stage=fus_train,
                         gmacs=gmacs(lambda x: pipe_cpu.embed(x), x1.cpu()), size_mb=size_mb(files),
                         latency_ms=lat * 1000, throughput_img_s=thr, peak_memory_mb=peak_mb(dev), device=device_name(dev)))
        print(rows[-1], flush=True)
    out = wd(cfg, 'results') / 'complexity.csv'
    df = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out); df = pd.concat([old[old.device != device_name(dev)], df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
