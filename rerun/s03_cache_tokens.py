"""Step 3. Freeze the fine-tuned backbones and cache last-stage tokens for every split.

    python s03_cache_tokens.py                 # all backbones, all seeds with a checkpoint
    python s03_cache_tokens.py --backbone deit_s --seeds 0 1

Evaluation transform only (no augmentation). Tokens are stored as float16 .npy:
  runs/cache/<backbone>/seed<s>/<split>_tokens.npy  (N, T, C)   and <split>_index.csv
"""
import os; os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')  # torch 2.0 on Apple GPUs: run missing ops on CPU (must precede `import torch`)
import argparse

import numpy as np
import torch

from common import BACKBONES, build_transforms, get_device, load_config, load_split, make_dataset, make_loader, wd
from models import TOKEN_DIMS, backbone_ckpt, load_backbone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    ap.add_argument('--backbone', nargs='*', default=BACKBONES)
    ap.add_argument('--seeds', nargs='*', type=int)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--limit', type=int, help='smoke tests only')
    a = ap.parse_args()
    cfg = load_config(a.config); dev = get_device(a.device)
    sp = load_split(cfg)
    if a.limit:
        sp = {k: v.groupby('label', group_keys=False).apply(lambda d: d.head(a.limit // 2)).reset_index(drop=True)
              for k, v in sp.items()}
    tf = build_transforms(cfg['image_size'], False)
    for name in a.backbone:
        for s in (a.seeds if a.seeds is not None else cfg['seeds']):
            if not backbone_ckpt(cfg, name, s).exists():
                print(f'skip {name} seed {s}: no checkpoint'); continue
            m = load_backbone(cfg, name, s, dev).backbone
            out = wd(cfg, 'cache', name, f'seed{s}')
            n_tok, c = TOKEN_DIMS[name]
            for split, df in sp.items():
                arr = np.lib.format.open_memmap(out / f'{split}_tokens.npy', mode='w+', dtype=np.float16,
                                                shape=(len(df), n_tok, c))
                dl = make_loader(make_dataset(df, cfg['data_root'], tf), a.batch, False, 0, cfg['backbone']['num_workers'])
                k = 0
                with torch.no_grad():
                    for x, _, _ in dl:
                        t, _ = m.tokens(x.to(dev))
                        arr[k:k + len(t)] = t.float().cpu().numpy().astype(np.float16); k += len(t)
                arr.flush(); del arr
                df[['path', 'group', 'subcategory', 'label']].to_csv(out / f'{split}_index.csv', index=False)
                print(f'{name} seed {s} {split}: {len(df)} images cached')


if __name__ == '__main__':
    main()
