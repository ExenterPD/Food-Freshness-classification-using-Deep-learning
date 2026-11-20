"""Step 5b. Paired statistics over the five seeds.

    python s07_stats.py [--metric macro_f1]

For each comparison: per-seed differences, mean, SD, 95% t-interval, Cohen's d_z (undefined when
SD = 0), paired t-test p, two-sided Wilcoxon signed-rank p (cannot go below 0.0625 with n = 5),
and McNemar's test on the paired test predictions of every seed (exact when discordant pairs < 25)
with Holm correction across comparisons within a seed.
Writes runs/results/stats.csv and table_stats.csv.
"""
import os; os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')  # torch 2.0 on Apple GPUs: run missing ops on CPU (must precede `import torch`)
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import binary_metrics, load_config, wd


def comparisons(fm):
    c = [('HM1 vs HM2 (final models)', f'cnn_tr|attention|xgb|{fm}', f'tr_tr|attention|xgb|{fm}')]
    for p, t in (('cnn_tr', 'HM1'), ('tr_tr', 'HM2')):
        c += [(f'Attention vs concatenation, {t}', f'{p}|attention|xgb|default', f'{p}|concat|xgb|default'),
              (f'Attention vs gated, {t}', f'{p}|attention|xgb|default', f'{p}|gated|xgb|default'),
              (f'XGBoost vs MLP head, {t}', f'{p}|attention|xgb|default', f'{p}|attention|mlp|-'),
              (f'Optuna-TPE vs BOHB, {t}', f'{p}|attention|xgb|tpe', f'{p}|attention|xgb|bohb')]
    return c


def mcnemar_p(ya, pa, pb):
    from statsmodels.stats.contingency_tables import mcnemar
    ca, cb = (pa == ya), (pb == ya)
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    if b + c == 0:
        return 1.0, b, c
    r = mcnemar([[0, b], [c, 0]], exact=(b + c) < 25, correction=True)
    return float(r.pvalue), b, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    ap.add_argument('--metric', default='macro_f1')
    a = ap.parse_args()
    cfg = load_config(a.config); out = wd(cfg, 'results')
    res = pd.read_csv(out / 'consolidated_metrics.csv')
    res = res[(res.opt_seed.isna()) | (res.opt_seed == res.seed)]
    files = {(r.key, r.seed): r.file for r in res.itertuples()}
    from statsmodels.stats.multitest import multipletests
    rows, mc = [], {}
    comps = comparisons(cfg['xgb']['final_method'])
    for name, ka, kb in comps:
        seeds = sorted(set(s for k, s in files if k == ka) & set(s for k, s in files if k == kb))
        if not seeds:
            continue
        diffs = []
        for s in seeds:
            da = pd.read_csv(files[(ka, s)], keep_default_na=False); db = pd.read_csv(files[(kb, s)], keep_default_na=False)
            m = da.merge(db, on='path', suffixes=('_a', '_b'))
            assert len(m) == len(da) == len(db), 'test sets differ between configurations'
            va = binary_metrics(m.label_a, m.prob_fresh_a)[a.metric]; vb = binary_metrics(m.label_b, m.prob_fresh_b)[a.metric]
            diffs.append(va - vb)
            mc[(name, s)] = mcnemar_p(m.label_a.values, m.pred_a.values, m.pred_b.values)
        d = np.array(diffs); n = len(d); mean = d.mean(); sd = d.std(ddof=1) if n > 1 else float('nan')
        if n > 1 and sd > 0:
            half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n); dz = mean / sd
            tp = stats.ttest_rel(d, np.zeros(n)).pvalue
        else:
            half, dz, tp = (0.0 if n > 1 else float('nan')), float('nan'), float('nan')
        try:
            wp = stats.wilcoxon(d).pvalue if np.any(d != 0) else float('nan')
        except ValueError:
            wp = float('nan')
        rows.append(dict(comparison=name, a=ka, b=kb, metric=a.metric, n=n, mean_diff=mean, sd_diff=sd,
                         ci_low=mean - half, ci_high=mean + half, cohens_dz=dz, t_p=tp, wilcoxon_p=wp,
                         per_seed_diffs=';'.join(f'{x:.6f}' for x in d)))
    st = pd.DataFrame(rows)
    # McNemar with Holm across comparisons, per seed
    sig = {r['comparison']: 0 for r in rows}; detail = []
    for s in sorted(set(k[1] for k in mc)):
        names = [n for (n, ss) in mc if ss == s]; ps = [mc[(n, s)][0] for n in names]
        adj = multipletests(ps, method='holm')[1]
        for n, p, q in zip(names, ps, adj):
            sig[n] += int(q < 0.05)
            detail.append(dict(comparison=n, seed=s, mcnemar_p=p, holm_p=q, b=mc[(n, s)][1], c=mc[(n, s)][2]))
    st['mcnemar_seeds_sig'] = st.comparison.map(sig)
    st.to_csv(out / 'stats.csv', index=False); pd.DataFrame(detail).to_csv(out / 'mcnemar_detail.csv', index=False)
    tab = pd.DataFrame({'Comparison': st.comparison,
                        'Mean Δ (points)': (st.mean_diff * 100).round(2),
                        '95% CI': [f'[{l * 100:.2f}, {h * 100:.2f}]' for l, h in zip(st.ci_low, st.ci_high)],
                        "Cohen's d_z": [('undefined' if np.isnan(x) else f'{x:.2f}') for x in st.cohens_dz],
                        'Wilcoxon p': st.wilcoxon_p.round(4),
                        'McNemar: seeds with p < 0.05 (Holm)': [f'{x}/{n}' for x, n in zip(st.mcnemar_seeds_sig, st.n)]})
    tab.to_csv(out / 'table_stats.csv', index=False)
    print(tab.to_string(index=False))


if __name__ == '__main__':
    main()
