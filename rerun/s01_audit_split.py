"""Step 1b. Duplicate audit and the frozen, stratified, group-aware 70/15/15 split.

Pass 1 (inspect):   python s01_audit_split.py --inspect
    computes ImageNet EfficientNetV2-M embeddings (cached), finds the closest pairs and writes
    runs/audit/closest_pairs.html. Open it and decide where "same photo" ends.
Pass 2 (split):     python s01_audit_split.py --phash-threshold T --cosine-threshold S
    (or set audit.phash_threshold / cosine_threshold in config.yaml)
    exact duplicates (pixel SHA-256) + near duplicates (pHash Hamming <= T AND cosine >= S)
    -> union-find clusters -> each cluster goes whole into one partition, strata = group x label.
Writes runs/split/split.csv, clusters.csv, audit_report.csv. Never regenerate split.csv after
training has started.
"""
import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd

from common import build_transforms, get_device, load_config, make_dataset, make_loader, wd


def phash_bits(hexes):
    v = np.array([int(h, 16) for h in hexes], dtype=np.uint64)
    return v


def popcount64(x):
    x = x.astype(np.uint64)
    b = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1)
    return b.sum(1)


def near_pairs(ph, max_dist):
    """All pairs (i<j) with Hamming distance <= max_dist, in blocks to limit memory."""
    n = len(ph); out = []
    for s in range(0, n, 512):
        blk = ph[s:s + 512]
        x = blk[:, None] ^ ph[None, :]
        d = popcount64(x.ravel()).reshape(x.shape)
        ii, jj = np.nonzero(d <= max_dist)
        ii = ii + s
        keep = jj > ii
        out.append(np.stack([ii[keep], jj[keep], d[ii[keep] - s, jj[keep]]], 1).astype(np.int64))
    return np.concatenate(out) if out else np.zeros((0, 3), np.int64)


def embeddings(cfg, df, cache):
    if Path(cache).exists():
        e = np.load(cache)
        if len(e) == len(df):
            return e
    import torch
    from models import Backbone
    dev = get_device()
    m = Backbone('effnetv2m', pretrained=not cfg.get('_no_pretrained')).to(dev).eval()
    ds = make_dataset(df, cfg['data_root'], build_transforms(cfg['image_size'], False))
    dl = make_loader(ds, 32, False, 0, cfg['backbone']['num_workers'])
    outs = []
    with torch.no_grad():
        for x, _, _ in dl:
            outs.append(m.pooled(x.to(dev)).float().cpu().numpy())
    e = np.concatenate(outs)
    e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-12
    np.save(cache, e)
    return e


def inspect(cfg, df, emb, out_dir, max_dist=16, per_bin=12):
    ph = phash_bits(df.phash)
    pairs = near_pairs(ph, max_dist)
    root = Path(cfg['data_root'])
    rows = []
    for i, j, d in pairs:
        cos = float(emb[i] @ emb[j]) if emb is not None else float('nan')
        rows.append((int(i), int(j), int(d), cos))
    pr = pd.DataFrame(rows, columns=['i', 'j', 'hamming', 'cosine'])
    pr['path_i'] = df.path.values[pr.i]; pr['path_j'] = df.path.values[pr.j]
    pr['label_i'] = df.label.values[pr.i]; pr['label_j'] = df.label.values[pr.j]
    pr.to_csv(out_dir / 'candidate_pairs.csv', index=False)
    print('Pairs per Hamming distance:')
    print(pr.hamming.value_counts().sort_index().to_string())
    parts = ['<html><head><meta charset="utf-8"><style>img{height:150px;margin:2px}'
             'td{vertical-align:top;font:12px sans-serif}</style></head><body>',
             '<h2>Closest image pairs by pHash distance</h2><p>Decide the largest distance at which pairs are still '
             'the same photograph (or the same item from almost the same view). Use the cosine values to set the '
             'second threshold.</p>']
    for d in sorted(pr.hamming.unique()):
        sub = pr[pr.hamming == d].sort_values('cosine').head(per_bin)
        parts.append(f'<h3>Hamming {d}: {int((pr.hamming == d).sum())} pairs (showing lowest-cosine {len(sub)})</h3><table>')
        for _, r in sub.iterrows():
            a = (root / r.path_i).resolve().as_uri(); b = (root / r.path_j).resolve().as_uri()
            parts.append(f'<tr><td><img src="{a}"><img src="{b}"></td><td>cos={r.cosine:.3f}<br>'
                         f'labels {r.label_i}/{r.label_j}<br>{html.escape(r.path_i)}<br>{html.escape(r.path_j)}</td></tr>')
        parts.append('</table>')
    (out_dir / 'closest_pairs.html').write_text('\n'.join(parts) + '</body></html>')
    print(f'\nOpen {out_dir / "closest_pairs.html"} and choose the thresholds.')


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def f(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x

    def u(self, a, b):
        a, b = self.f(a), self.f(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def split(cfg, df, emb, t, s, policy, out_dir):
    n = len(df); uf = UF(n)
    # exact duplicates
    n_exact = 0
    for _, idx in df.groupby('sha256').indices.items():
        idx = list(idx)
        for k in idx[1:]:
            uf.u(idx[0], k); n_exact += 1
    pairs = near_pairs(phash_bits(df.phash), t)
    if emb is not None and s is not None:
        cos = np.einsum('ij,ij->i', emb[pairs[:, 0]], emb[pairs[:, 1]]) if len(pairs) else np.zeros(0)
        keep = cos >= s
    else:
        keep = np.ones(len(pairs), bool)
    near = pairs[keep]
    for i, j, _ in near:
        uf.u(int(i), int(j))
    df = df.copy()
    df['cluster_id'] = [uf.f(i) for i in range(n)]
    sizes = df.cluster_id.map(df.cluster_id.value_counts())
    mixed = df.groupby('cluster_id').label.nunique()
    mixed_ids = set(mixed[mixed > 1].index)
    df['cluster_size'] = sizes
    df['mixed_label'] = df.cluster_id.isin(mixed_ids)
    clusters = df[df.cluster_size > 1].sort_values(['cluster_id', 'path'])
    clusters.to_csv(out_dir / 'clusters.csv', index=False)
    dropped = 0
    if policy == 'drop' and mixed_ids:
        dropped = int(df.mixed_label.sum())
        df[df.mixed_label].to_csv(out_dir / 'dropped_mixed_label.csv', index=False)
        df = df[~df.mixed_label].reset_index(drop=True)

    # stratified group assignment: stratum of a cluster = its most common (group, label)
    rng = np.random.RandomState(cfg['split']['split_seed'])
    frac = {k: cfg['split'][k] for k in ('train', 'val', 'test')}
    cl = df.groupby('cluster_id').agg(size=('path', 'size'),
                                      stratum=('group', lambda g: g.mode().iat[0]))
    cl['stratum'] = cl['stratum'] + '|' + df.groupby('cluster_id').label.agg(lambda v: str(v.mode().iat[0]))
    assign = {}
    for st, cdf in cl.groupby('stratum'):
        ids = cdf.index.values.copy(); rng.shuffle(ids)
        ids = sorted(ids, key=lambda c: -cdf.loc[c, 'size'])  # big clusters first, ties random
        total = cdf['size'].sum(); got = {k: 0 for k in frac}
        for c in ids:
            deficit = {k: frac[k] * total - got[k] for k in frac}
            k = max(deficit, key=deficit.get)
            assign[c] = k; got[k] += cdf.loc[c, 'size']
    df['partition'] = df.cluster_id.map(assign)
    assert df.groupby('cluster_id').partition.nunique().max() == 1
    df.to_csv(out_dir / 'split.csv', index=False)

    # report
    rep = []
    for g, gdf in df.groupby('group'):
        c = gdf[gdf.cluster_size > 1]
        rep.append(dict(group=g, images=len(gdf), good=int(gdf.label.sum()), bad=int((gdf.label == 0).sum()),
                        train=int((gdf.partition == 'train').sum()), val=int((gdf.partition == 'val').sum()),
                        test=int((gdf.partition == 'test').sum()),
                        exact_duplicate_images=int(gdf.duplicated('sha256', keep='first').sum()),
                        images_in_clusters=len(c), clusters_ge2=c.cluster_id.nunique(),
                        mixed_label_clusters=c[c.mixed_label].cluster_id.nunique()))
    rep = pd.DataFrame(rep)
    tot = rep.drop(columns='group').sum(); tot['group'] = 'Total'
    rep = pd.concat([rep, tot.to_frame().T], ignore_index=True)
    rep.to_csv(out_dir / 'audit_report.csv', index=False)
    summary = dict(phash_threshold=t, cosine_threshold=s, mixed_label_policy=policy,
                   exact_duplicate_links=n_exact, near_pairs_hamming=int(len(pairs)),
                   near_pairs_confirmed=int(len(near)), clusters_ge2=int(clusters.cluster_id.nunique()),
                   largest_cluster=int(df.cluster_size.max()), mixed_label_clusters=len(mixed_ids),
                   images_dropped=dropped, n_images=len(df))
    pd.Series(summary).to_csv(out_dir / 'audit_summary.csv', header=['value'])
    print(rep.to_string(index=False))
    print(pd.Series(summary).to_string())
    print(f'\nWrote {out_dir / "split.csv"}. Copy it to every machine (same path under runs/split/).')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    ap.add_argument('--inspect', action='store_true')
    ap.add_argument('--phash-threshold', type=int)
    ap.add_argument('--cosine-threshold', type=float)
    ap.add_argument('--no-embed', action='store_true', help='skip the embedding check')
    ap.add_argument('--no-pretrained', action='store_true', help='smoke tests only')
    ap.add_argument('--force', action='store_true', help='overwrite an existing split.csv')
    a = ap.parse_args()
    cfg = load_config(a.config); cfg['_no_pretrained'] = a.no_pretrained
    sd = wd(cfg, 'split'); ad = wd(cfg, 'audit')
    df = pd.read_csv(sd / 'manifest.csv', keep_default_na=False)
    emb = None if a.no_embed else embeddings(cfg, df, ad / 'embeddings_effnetv2m_imagenet.npy')
    if a.inspect:
        inspect(cfg, df, emb, ad)
        return
    t = a.phash_threshold if a.phash_threshold is not None else cfg['audit']['phash_threshold']
    s = a.cosine_threshold if a.cosine_threshold is not None else cfg['audit']['cosine_threshold']
    if t is None or (s is None and emb is not None):
        raise SystemExit('Set both thresholds (run --inspect first, then pass --phash-threshold/--cosine-threshold).')
    if (sd / 'split.csv').exists() and not a.force:
        raise SystemExit('split.csv already exists. It must not change once training has started. Use --force only before training.')
    split(cfg, df, emb, t, s, cfg['audit']['mixed_label_policy'], sd)


if __name__ == '__main__':
    main()
