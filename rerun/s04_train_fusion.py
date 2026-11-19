"""Step 4a. Train a fusion operator + MLP head on cached tokens of one backbone pair.

    python s04_train_fusion.py --pair cnn_tr --fusion attention --seed 0
    python s04_train_fusion.py --pair cnn_tr --fusion all --seed 0
    python s04_train_fusion.py --pair cnn_tr --fusion attention --seed 0 --sweep    # choose d and h (seed 0 only)

pair: cnn_tr = EfficientNetV2-M + MobileViT-S (Hybrid Model 1); tr_tr = DeiT-Small + MobileViT-S (Hybrid Model 2)
fusion: concat | gated | attention.  All three share the projections and are trained identically.
Outputs in runs/fusion/<pair>/<fusion>/seed<s>/: best.pt, training_log.csv, meta.json,
  val_preds.csv / test_preds.csv (MLP head), z_train.npy / z_val.npy / z_test.npy (pooled embeddings).
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import PAIRS, binary_metrics, get_device, load_config, save_json, seed_everything, wd, weighted_f1, write_preds
from models import TOKEN_DIMS, Fusion, trainable_params

FUSIONS = ['concat', 'gated', 'attention']


def load_cache(cfg, pair, seed):
    a, b = PAIRS[pair]; data = {}
    for split in ('train', 'val', 'test'):
        ca = wd(cfg, 'cache', a, f'seed{seed}'); cb = wd(cfg, 'cache', b, f'seed{seed}')
        ia = pd.read_csv(ca / f'{split}_index.csv', keep_default_na=False)
        ib = pd.read_csv(cb / f'{split}_index.csv', keep_default_na=False)
        assert (ia.path.values == ib.path.values).all(), 'caches are not aligned; re-run s03 with the same split'
        data[split] = (np.load(ca / f'{split}_tokens.npy', mmap_mode='r'),
                       np.load(cb / f'{split}_tokens.npy', mmap_mode='r'), ia)
    return data


def batches(ta, tb, y, bs, shuffle, rng):
    idx = np.arange(len(y))
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, len(idx), bs):
        j = np.sort(idx[s:s + bs])
        yield (torch.from_numpy(np.asarray(ta[j], dtype=np.float32)), torch.from_numpy(np.asarray(tb[j], dtype=np.float32)),
               torch.from_numpy(y[j]), j)


@torch.no_grad()
def predict(model, ta, tb, y, dev, bs=256):
    model.eval(); P, Z = [], []
    for xa, xb, _, _ in batches(ta, tb, y, bs, False, None):
        z = model.embed(xa.to(dev), xb.to(dev))
        P.append(torch.softmax(model.head(z), 1)[:, 1].cpu().numpy()); Z.append(z.cpu().numpy())
    return np.concatenate(P), np.concatenate(Z)


def train_one(cfg, data, pair, kind, seed, dev, d, heads, out=None):
    fc = cfg['fusion']; seed_everything(seed)
    a, b = PAIRS[pair]
    model = Fusion(TOKEN_DIMS[a][1], TOKEN_DIMS[b][1], kind, d, heads, fc['mlp_hidden'], fc['dropout']).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=fc['lr'], weight_decay=fc['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=fc['epochs'])
    crit = nn.CrossEntropyLoss(label_smoothing=fc['label_smoothing'])
    ta, tb, itr = data['train']; va, vb, iva = data['val']
    ytr = itr.label.values.astype(np.int64); yva = iva.label.values.astype(np.int64)
    rng = np.random.RandomState(seed)
    best, best_state, best_ep, bad, log = -1, None, 0, 0, []
    t_start = time.time()
    for ep in range(1, fc['epochs'] + 1):
        model.train(); tl, n = 0.0, 0
        for xa, xb, yb, _ in batches(ta, tb, ytr, fc['batch_size'], True, rng):
            xa, xb, yb = xa.to(dev), xb.to(dev), yb.to(dev)
            loss = crit(model(xa, xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(yb); n += len(yb)
        sched.step()
        pva, _ = predict(model, va, vb, yva, dev)
        wf1 = weighted_f1(yva, (pva >= 0.5).astype(int)); m = binary_metrics(yva, pva)
        log.append(dict(epoch=ep, train_loss=tl / n, val_weighted_f1=wf1, val_macro_f1=m['macro_f1'], val_auroc=m['auroc']))
        if wf1 > best:
            best, best_ep, bad = wf1, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= fc['early_stop_patience']:
                break
    model.load_state_dict(best_state)
    info = dict(best_val_weighted_f1=best, best_epoch=best_ep, train_time_s=time.time() - t_start,
                trainable_params=trainable_params(model))
    if out is not None:
        torch.save(best_state, out / 'best.pt')
        pd.DataFrame(log).to_csv(out / 'training_log.csv', index=False)
    return model, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    ap.add_argument('--pair', required=True, choices=list(PAIRS))
    ap.add_argument('--fusion', required=True, choices=FUSIONS + ['all'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--d', type=int); ap.add_argument('--heads', type=int)
    ap.add_argument('--sweep', action='store_true', help='validation sweep over fusion.sweep_d x sweep_heads')
    ap.add_argument('--device', default='auto')
    a = ap.parse_args()
    cfg = load_config(a.config); fc = cfg['fusion']; dev = get_device(a.device)
    data = load_cache(cfg, a.pair, a.seed)

    if a.sweep:
        rows = []
        for d in fc['sweep_d']:
            for h in fc['sweep_heads']:
                if d % h:
                    continue
                _, info = train_one(cfg, data, a.pair, 'attention', a.seed, dev, d, h)
                rows.append(dict(d=d, heads=h, **info)); print(rows[-1], flush=True)
        out = wd(cfg, 'fusion', a.pair, 'sweep'); pd.DataFrame(rows).to_csv(out / f'sweep_seed{a.seed}.csv', index=False)
        print(f'Wrote {out}/sweep_seed{a.seed}.csv. Put the chosen d and heads in config.yaml (fusion.d, fusion.heads).')
        return

    d = a.d or fc['d']; h = a.heads or fc['heads']
    for kind in (FUSIONS if a.fusion == 'all' else [a.fusion]):
        out = wd(cfg, 'fusion', a.pair, kind, f'seed{a.seed}')
        model, info = train_one(cfg, data, a.pair, kind, a.seed, dev, d, h, out)
        meta = dict(stage='fusion', model=a.pair, fusion=kind, classifier='mlp', seed=a.seed, d=d,
                    heads=h, mlp_hidden=fc['mlp_hidden'], dropout=fc['dropout'], device=str(dev), **info)
        for split in ('train', 'val', 'test'):
            ta, tb, idx = data[split]
            p, z = predict(model, ta, tb, idx.label.values, dev)
            np.save(out / f'z_{split}.npy', z.astype(np.float32))
            if split != 'train':
                write_preds(out / f'{split}_preds.csv', idx, p)
        save_json(meta, out / 'meta.json')
        print(f'{a.pair} {kind} seed {a.seed}: best val weighted F1 {info["best_val_weighted_f1"]:.4f} '
              f'(epoch {info["best_epoch"]}), trainable params {info["trainable_params"]}')


if __name__ == '__main__':
    main()
