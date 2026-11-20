"""Step 4b. XGBoost on fused embeddings: default, Optuna-TPE and BOHB with a matched budget.

    python s05_xgb_hpo.py --pair cnn_tr --fusion attention --seed 0 --method default tpe bohb
    python s05_xgb_hpo.py --pair cnn_tr --fusion attention --seed 0 --method tpe bohb --opt-seeds 0 1 2 3 4

Both searches use the same space (config.yaml: xgb.space), maximise validation weighted F1 and stop
when the boosting rounds summed over all evaluated configurations reach xgb.total_rounds.
TPE samples n_estimators in [b_min, b_max]; for BOHB n_estimators is the Hyperband budget.
The selected configuration is refit on train+val (xgb.refit_on_train_val) and the test set is
scored once. Default XGBoost = library defaults with the same seed.
For the final models the optimiser seed equals the data seed (default when --opt-seeds is omitted).

Outputs in runs/xgb/<pair>/<fusion>/seed<s>/<method>[_opt<k>]/:
  model.json, trials.csv, best_config.json, val_preds.csv, test_preds.csv, meta.json
"""
import os; os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')  # torch 2.0 on Apple GPUs: run missing ops on CPU (must precede `import torch`)
import argparse
import logging
import time

import numpy as np
import pandas as pd

from common import PAIRS, load_config, save_json, weighted_f1, write_preds, wd
from models import xgb_dir

logging.getLogger('hpbandster').setLevel(logging.WARNING)


def load_z(cfg, pair, kind, seed):
    d = wd(cfg, 'fusion', pair, kind, f'seed{seed}')
    out = {}
    for s in ('train', 'val', 'test'):
        z = np.load(d / f'z_{s}.npy')
        idx = pd.read_csv(wd(cfg, 'cache', PAIRS[pair][0], f'seed{seed}') / f'{s}_index.csv', keep_default_na=False)
        out[s] = (z, idx)
    return out


def make_model(params, seed):
    import xgboost as xgb
    return xgb.XGBClassifier(**params, random_state=seed, n_jobs=-1, eval_metric='logloss', tree_method='hist')


class Tracker:
    """Counts boosting rounds spent; refuses work once the matched budget is used up."""

    def __init__(self, total):
        self.total, self.used, self.rows, self.t0 = total, 0, [], time.time()

    def exhausted(self, nxt=0):
        return self.used + nxt > self.total

    def record(self, params, n_est, score):
        self.used += n_est
        self.rows.append(dict(**params, n_estimators=n_est, val_weighted_f1=score, rounds_used=self.used,
                              elapsed_s=time.time() - self.t0))


def evaluate(params, n_est, Z, seed):
    (ztr, itr), (zva, iva) = Z['train'], Z['val']
    m = make_model(dict(params, n_estimators=int(n_est)), seed)
    m.fit(ztr, itr.label.values)
    return weighted_f1(iva.label.values, m.predict(zva))


def run_tpe(cfg, Z, seed, opt_seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    xc, sp = cfg['xgb'], cfg['xgb']['space']; tr = Tracker(xc['total_rounds'])

    def objective(trial):
        p = dict(max_depth=trial.suggest_int('max_depth', *sp['max_depth']),
                 learning_rate=trial.suggest_float('learning_rate', *sp['learning_rate'], log=True),
                 subsample=trial.suggest_float('subsample', *sp['subsample']),
                 colsample_bytree=trial.suggest_float('colsample_bytree', *sp['colsample_bytree']),
                 reg_alpha=trial.suggest_float('reg_alpha', *sp['reg_alpha']),
                 reg_lambda=trial.suggest_float('reg_lambda', *sp['reg_lambda']))
        n = trial.suggest_int('n_estimators', xc['b_min'], xc['b_max'])
        if tr.exhausted(n):
            trial.study.stop(); raise optuna.TrialPruned()
        s = evaluate(p, n, Z, seed); tr.record(p, n, s)
        return s

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=opt_seed))
    while not tr.exhausted(xc['b_min']):
        study.optimize(objective, n_trials=1)
        if study.trials and study.trials[-1].state == optuna.trial.TrialState.PRUNED:
            break
    return tr


def run_bohb(cfg, Z, seed, opt_seed):
    import ConfigSpace as CS
    import hpbandster.core.nameserver as hpns
    from hpbandster.core.worker import Worker
    from hpbandster.optimizers import BOHB
    if int(CS.__version__.split('.')[0]) >= 1:
        raise SystemExit('hpbandster 0.7.4 hangs with ConfigSpace >= 1.0. Install ConfigSpace==0.7.1 (see requirements.txt).')
    xc, sp = cfg['xgb'], cfg['xgb']['space']; tr = Tracker(xc['total_rounds'])
    np.random.seed(opt_seed)
    cs = CS.ConfigurationSpace(seed=opt_seed)
    hps = [CS.UniformIntegerHyperparameter('max_depth', *sp['max_depth']),
           CS.UniformFloatHyperparameter('learning_rate', *sp['learning_rate'], log=True),
           CS.UniformFloatHyperparameter('subsample', *sp['subsample']),
           CS.UniformFloatHyperparameter('colsample_bytree', *sp['colsample_bytree']),
           CS.UniformFloatHyperparameter('reg_alpha', *sp['reg_alpha']),
           CS.UniformFloatHyperparameter('reg_lambda', *sp['reg_lambda'])]
    (cs.add if hasattr(cs, 'add') else cs.add_hyperparameters)(hps)

    class W(Worker):
        def compute(self, config, budget, **kw):
            n = int(round(budget))
            if tr.exhausted(n):
                return {'loss': 1.0, 'info': {'skipped': True}}
            p = {k: (int(v) if k == 'max_depth' else float(v)) for k, v in dict(config).items()}
            s = evaluate(p, n, Z, seed); tr.record(p, n, s)
            return {'loss': 1 - s, 'info': {'val_weighted_f1': s}}

    run_id = f'bohb_{seed}_{opt_seed}_{time.time_ns()}'
    ns = hpns.NameServer(run_id=run_id, host='127.0.0.1', port=None)
    host, port = ns.start()
    w = W(run_id=run_id, nameserver=host, nameserver_port=port)
    w.run(background=True)
    opt = BOHB(configspace=cs, run_id=run_id, nameserver=host, nameserver_port=port,
               min_budget=xc['b_min'], max_budget=xc['b_max'], eta=xc['eta'])
    try:
        while not tr.exhausted(xc['b_min']):
            before = tr.used
            opt.run(n_iterations=1, min_n_workers=1)
            if tr.used == before:
                break
    finally:
        opt.shutdown(shutdown_workers=True); ns.shutdown()
    return tr


def pick_best(tr, method):
    """TPE: best validation score over all trials. BOHB: incumbent = best score on the largest
    budget reached, which is how BOHB defines its result."""
    df = pd.DataFrame(tr.rows)
    if method == 'bohb':
        pool = df[df.n_estimators == df.n_estimators.max()]
        return pool.loc[pool.val_weighted_f1.idxmax()], df
    return df.loc[df.val_weighted_f1.idxmax()], df


def finish(cfg, Z, seed, method, opt_seed, params, out, extra):
    (ztr, itr), (zva, iva), (zte, ite) = Z['train'], Z['val'], Z['test']
    m = make_model(params, seed)
    m.fit(ztr, itr.label.values)
    p_val = m.predict_proba(zva)[:, 1]
    if cfg['xgb']['refit_on_train_val']:
        m = make_model(params, seed)
        m.fit(np.vstack([ztr, zva]), np.concatenate([itr.label.values, iva.label.values]))
    p_te = m.predict_proba(zte)[:, 1]
    m.save_model(str(out / 'model.json'))
    write_preds(out / 'val_preds.csv', iva, p_val)
    write_preds(out / 'test_preds.csv', ite, p_te)
    save_json(params, out / 'best_config.json')
    return dict(params=params, **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    ap.add_argument('--pair', required=True, choices=list(PAIRS))
    ap.add_argument('--fusion', required=True, nargs='+', choices=['concat', 'gated', 'attention'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--method', nargs='+', default=['default', 'tpe', 'bohb'], choices=['default', 'tpe', 'bohb'])
    ap.add_argument('--opt-seeds', nargs='*', type=int)
    a = ap.parse_args()
    cfg = load_config(a.config)
    for kind in a.fusion:
        Z = load_z(cfg, a.pair, kind, a.seed)
        for method in a.method:
            opt_seeds = [None] if method == 'default' else (a.opt_seeds if a.opt_seeds else [a.seed])
            for k in opt_seeds:
                out = xgb_dir(cfg, a.pair, kind, a.seed, method, k); out.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                meta = dict(stage='xgb', model=a.pair, fusion=kind, classifier='xgb', method=method,
                            seed=a.seed, opt_seed=k)
                if method == 'default':
                    info = finish(cfg, Z, a.seed, method, k, {}, out, {})
                else:
                    tr = run_tpe(cfg, Z, a.seed, k) if method == 'tpe' else run_bohb(cfg, Z, a.seed, k)
                    best, trials = pick_best(tr, method)
                    trials.to_csv(out / 'trials.csv', index=False)
                    params = {c: best[c] for c in ['max_depth', 'learning_rate', 'subsample', 'colsample_bytree',
                                                   'reg_alpha', 'reg_lambda', 'n_estimators']}
                    params['max_depth'] = int(params['max_depth']); params['n_estimators'] = int(params['n_estimators'])
                    info = finish(cfg, Z, a.seed, method, k, params, out,
                                  dict(best_val_weighted_f1=float(best.val_weighted_f1), configs_evaluated=len(trials),
                                       rounds_used=int(tr.used), budget=cfg['xgb']['total_rounds']))
                meta.update(info); meta['wall_time_s'] = time.time() - t0
                save_json(meta, out / 'meta.json')
                print(f'{a.pair} {kind} seed {a.seed} {method} opt{k}: '
                      f'{ {x: meta[x] for x in meta if x in ("best_val_weighted_f1", "configs_evaluated", "rounds_used")} } '
                      f'({meta["wall_time_s"]:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
