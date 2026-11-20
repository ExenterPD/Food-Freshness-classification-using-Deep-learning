"""Step 5a. Every reported metric, recomputed from saved per-image predictions.

    python s06_metrics.py [--no-bootstrap]

Fresh (label 1) is the positive class. Reads every test_preds.csv under runs/ and writes
runs/results/:
  consolidated_metrics.csv  one row per run (model, fusion, classifier, method, seed, opt_seed)
  summary_metrics.csv       mean and SD over the five data seeds for every configuration
  category_metrics.csv      per food group, per run; category_summary.csv: mean and SD
  table_backbones.csv, table_ablation.csv, table_final.csv, table_category.csv   (paper-ready text)
"""
import os; os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')  # torch 2.0 on Apple GPUs: run missing ops on CPU (must precede `import torch`)
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import PAIRS, PAIR_NAMES, binary_metrics, load_config, load_json, wd

METRICS = ['accuracy', 'precision_fresh', 'recall_fresh', 'specificity', 'fpr', 'f1_fresh', 'f1_spoiled',
           'macro_f1', 'weighted_f1', 'auroc']


def discover(cfg):
    w = Path(cfg['work_dir']); runs = []
    for f in w.glob('backbones/*/seed*/test_preds.csv'):
        m = load_json(f.parent / 'meta.json')
        runs.append(dict(file=f, family='backbone', model=m['model'], fusion='-', classifier='linear', method='-',
                         seed=m['seed'], opt_seed=None, tag=m.get('tag', ''), meta=m))
    for f in w.glob('fusion/*/*/seed*/test_preds.csv'):
        m = load_json(f.parent / 'meta.json')
        runs.append(dict(file=f, family='fusion', model=m['model'], fusion=m['fusion'], classifier='mlp', method='-',
                         seed=m['seed'], opt_seed=None, tag='', meta=m))
    for f in w.glob('xgb/*/*/seed*/*/test_preds.csv'):
        m = load_json(f.parent / 'meta.json')
        runs.append(dict(file=f, family='xgb', model=m['model'], fusion=m['fusion'], classifier='xgb',
                         method=m['method'], seed=m['seed'], opt_seed=m.get('opt_seed'), tag='', meta=m))
    return runs


def key(r):
    return f"{r['model']}|{r['fusion']}|{r['classifier']}|{r['method']}{r['tag']}"


def bootstrap_ci(df, B, rng):
    y, p = df.label.values, df.prob_fresh.values
    acc, mf1 = [], []
    for _ in range(B):
        j = rng.randint(0, len(y), len(y))
        m = binary_metrics(y[j], p[j]); acc.append(m['accuracy']); mf1.append(m['macro_f1'])
    q = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
    return q(acc), q(mf1)


def fmt(mean, sd, pct=True, nd=2):
    if np.isnan(mean):
        return '—'
    k = 100 if pct else 1
    return f'{mean * k:.{nd}f} ± {sd * k:.{nd}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    ap.add_argument('--no-bootstrap', action='store_true')
    a = ap.parse_args()
    cfg = load_config(a.config); out = wd(cfg, 'results')
    runs = discover(cfg)
    if not runs:
        raise SystemExit('No predictions found under runs/.')
    rng = np.random.RandomState(0); rows, cats = [], []
    for r in runs:
        df = pd.read_csv(r['file'], keep_default_na=False)
        m = binary_metrics(df.label, df.prob_fresh)
        row = dict(key=key(r), family=r['family'], model=r['model'], fusion=r['fusion'], classifier=r['classifier'],
                   method=r['method'], tag=r['tag'], seed=r['seed'], opt_seed=r['opt_seed'], file=str(r['file']), **m)
        if not a.no_bootstrap and cfg['metrics']['bootstrap']:
            (al, ah), (fl, fh) = bootstrap_ci(df, cfg['metrics']['bootstrap'], rng)
            row.update(accuracy_ci_low=al, accuracy_ci_high=ah, macro_f1_ci_low=fl, macro_f1_ci_high=fh)
        for x in ('best_epoch', 'mean_epoch_time_s', 'train_time_s', 'device_name'):
            if x in r['meta']:
                row[x] = r['meta'][x]
        rows.append(row)
        for g, gdf in df.groupby('group'):
            cats.append(dict(key=row['key'], model=r['model'], fusion=r['fusion'], classifier=r['classifier'],
                             method=r['method'], seed=r['seed'], opt_seed=r['opt_seed'], group=g,
                             n_good=int(gdf.label.sum()), n_bad=int((gdf.label == 0).sum()),
                             **{k: v for k, v in binary_metrics(gdf.label, gdf.prob_fresh).items() if k in METRICS}))
    res = pd.DataFrame(rows); res.to_csv(out / 'consolidated_metrics.csv', index=False)
    cat = pd.DataFrame(cats); cat.to_csv(out / 'category_metrics.csv', index=False)

    # main runs: tuned XGBoost uses opt_seed == data seed
    main_runs = res[(res.opt_seed.isna()) | (res.opt_seed == res.seed)]
    summ = main_runs.groupby('key')[METRICS].agg(['mean', 'std', 'count'])
    summ.columns = [f'{m}_{s}' for m, s in summ.columns]; summ.to_csv(out / 'summary_metrics.csv')
    mc = cat[(cat.opt_seed.isna()) | (cat.opt_seed == cat.seed)]
    csum = mc.groupby(['key', 'group'])[METRICS + ['n_good', 'n_bad']].agg(['mean', 'std'])
    csum.columns = [f'{m}_{s}' for m, s in csum.columns]; csum.to_csv(out / 'category_summary.csv')

    def cell(k, m, pct=True, nd=2):
        if f'{m}_mean' not in summ.columns or k not in summ.index:
            return '—'
        return fmt(summ.loc[k, f'{m}_mean'], summ.loc[k, f'{m}_std'], pct, nd)

    # Table: backbones
    bb = []
    for b in ['effnetv2m', 'mobilevit_s', 'deit_s']:
        k = f'{b}|-|linear|-'; sub = res[res.key == k]
        bb.append(dict(Backbone=b, Accuracy=cell(k, 'accuracy'), **{'Macro-F1': cell(k, 'macro_f1')},
                       AUROC=cell(k, 'auroc', False, 4), Specificity=cell(k, 'specificity'),
                       **{'Best epoch (median)': sub.best_epoch.median() if 'best_epoch' in sub and len(sub) else '—',
                          'Time per epoch (s)': round(sub.mean_epoch_time_s.mean(), 1) if 'mean_epoch_time_s' in sub and len(sub) else '—'}))
    pd.DataFrame(bb).to_csv(out / 'table_backbones.csv', index=False)

    # Table: ablation
    abl_rows = [('First backbone only', lambda p: f'{PAIRS[p][0]}|-|linear|-'),
                ('MobileViT-S only', lambda p: 'mobilevit_s|-|linear|-'),
                ('Concatenation + MLP', lambda p: f'{p}|concat|mlp|-'),
                ('Concatenation + XGBoost', lambda p: f'{p}|concat|xgb|default'),
                ('Gated fusion + XGBoost', lambda p: f'{p}|gated|xgb|default'),
                ('Attention + MLP', lambda p: f'{p}|attention|mlp|-'),
                ('Attention + XGBoost (default)', lambda p: f'{p}|attention|xgb|default'),
                ('Attention + XGBoost (Optuna-TPE)', lambda p: f'{p}|attention|xgb|tpe'),
                ('Attention + XGBoost (BOHB)', lambda p: f'{p}|attention|xgb|bohb')]
    ab = []
    for name, kf in abl_rows:
        row = {'Configuration': name}
        for p, tag in (('cnn_tr', 'HM1'), ('tr_tr', 'HM2')):
            k = kf(p)
            row[f'{tag} Accuracy'] = cell(k, 'accuracy'); row[f'{tag} Macro-F1'] = cell(k, 'macro_f1')
            row[f'{tag} AUROC'] = cell(k, 'auroc', False, 4)
        ab.append(row)
    pd.DataFrame(ab).to_csv(out / 'table_ablation.csv', index=False)

    # Table: final models
    fm = cfg['xgb']['final_method']; fin = []
    for p in PAIRS:
        k = f'{p}|attention|xgb|{fm}'
        fin.append({'Model': PAIR_NAMES[p], 'Accuracy': cell(k, 'accuracy'), 'Precision (Fresh)': cell(k, 'precision_fresh'),
                    'Recall (Fresh)': cell(k, 'recall_fresh'), 'Specificity': cell(k, 'specificity'),
                    'F1 (Fresh)': cell(k, 'f1_fresh'), 'Macro-F1': cell(k, 'macro_f1'), 'Weighted F1': cell(k, 'weighted_f1'),
                    'AUROC': cell(k, 'auroc', False, 4), 'Seeds': int(summ.loc[k, 'accuracy_count']) if k in summ.index else 0})
    pd.DataFrame(fin).to_csv(out / 'table_final.csv', index=False)

    # Table: by food group (final models)
    ct = []
    for p, tag in (('cnn_tr', 'HM1'), ('tr_tr', 'HM2')):
        k = f'{p}|attention|xgb|{fm}'
        if k not in csum.index.get_level_values(0):
            continue
        for g in csum.loc[k].index:
            r = csum.loc[(k, g)]
            ct.append({'Food group': g, 'Model': tag, 'Test images (Good/Bad)': f"{int(r['n_good_mean'])}/{int(r['n_bad_mean'])}",
                       **{c: fmt(r[f'{m}_mean'], r[f'{m}_std']) for c, m in
                          (('Accuracy', 'accuracy'), ('Precision', 'precision_fresh'), ('Recall', 'recall_fresh'),
                           ('Specificity', 'specificity'), ('Macro-F1', 'macro_f1'))}})
    pd.DataFrame(ct).to_csv(out / 'table_category.csv', index=False)
    hpo_summary(cfg, res, out)
    print(f'{len(res)} runs summarised. Tables in {out}')


def hpo_summary(cfg, res, out):
    """TPE vs BOHB at matched budget on the seed-0 embeddings, over the optimiser seeds."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    w = Path(cfg['work_dir']); rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (p, tag) in zip(axes, (('cnn_tr', 'HM1'), ('tr_tr', 'HM2'))):
        for method, colour in (('tpe', 'tab:blue'), ('bohb', 'tab:orange')):
            metas = []
            for d in sorted((w / 'xgb' / p / 'attention' / 'seed0').glob(f'{method}_opt*')):
                if not (d / 'meta.json').exists():
                    continue
                m = load_json(d / 'meta.json'); tr = pd.read_csv(d / 'trials.csv')
                te = binary_metrics(*pd.read_csv(d / 'test_preds.csv')[['label', 'prob_fresh']].values.T)
                metas.append(dict(best=m['best_val_weighted_f1'], test=te['macro_f1'], n=m['configs_evaluated'],
                                  rounds=m['rounds_used'], time=m['wall_time_s']))
                ax.plot(tr.rounds_used, tr.val_weighted_f1.cummax(), color=colour, alpha=0.6,
                        label=method.upper() if len(metas) == 1 else None)
            if metas:
                df = pd.DataFrame(metas)
                rows.append({'Model': tag, 'Method': 'Optuna-TPE' if method == 'tpe' else 'BOHB',
                             'Best validation weighted F1': fmt(df.best.mean(), df.best.std()),
                             'Test macro-F1': fmt(df.test.mean(), df.test.std()),
                             'Configurations evaluated': f'{df.n.mean():.1f} ± {df.n.std():.1f}',
                             'Boosting rounds used': f'{df.rounds.mean():.0f}', 'Wall-clock time (s)': f'{df.time.mean():.0f} ± {df.time.std():.0f}',
                             'Optimiser seeds': len(df)})
        ax.set_title(tag); ax.set_xlabel('cumulative boosting rounds'); ax.legend()
    axes[0].set_ylabel('best validation weighted F1 so far')
    fig.tight_layout(); fig.savefig(out / 'figure_hpo_convergence.png', dpi=300); plt.close(fig)
    pd.DataFrame(rows).to_csv(out / 'table_hpo.csv', index=False)


if __name__ == '__main__':
    main()
