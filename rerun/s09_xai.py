"""Step 7. Explanations and their quantitative evaluation.

    python s09_xai.py --shap        # TreeSHAP on the deployed (final) XGBoost, every seed
    python s09_xai.py --maps        # Grad-CAM, fusion attention, deletion/insertion, mask energy, stability
    python s09_xai.py --figures     # example overlays (seed 0) with one fixed colour scale

SHAP stays at feature level: embedding dimensions are averages over tokens and are NOT mapped to
pixels. Spatial maps:
  * Grad-CAM on each backbone's last feature map (EfficientNetV2-M 7x7, MobileViT-S 7x7, DeiT-Small
    patch tokens as 14x14), computed through the fusion module and its MLP head, i.e. the
    differentiable counterpart of XGBoost on the same embedding; target = predicted class.
  * Fusion attention: mean attention each token receives (over heads and queries), per source.
Faithfulness (deletion/insertion) is scored on the deployed XGBoost pipeline.
Needs masks from s08_robustness.py --make-masks for the energy-in-mask score.
"""
import os; os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')  # torch 2.0 on Apple GPUs: run missing ops on CPU (must precede `import torch`)
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import PAIRS, build_transforms, get_device, load_config, load_rgb, load_split, wd
from models import GRID, Pipeline, load_xgb


# ----------------------------------------------------------------- SHAP
def run_shap(cfg, seeds):
    import shap
    from scipy.stats import spearmanr
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = wd(cfg, 'results', 'shap'); fm = cfg['xgb']['final_method']; rows = []
    rng = np.random.RandomState(0)
    for p in PAIRS:
        for s in seeds:
            try:
                model, d = load_xgb(cfg, p, 'attention', s, fm)
            except Exception as e:
                print(f'skip {p} seed {s}: {e}'); continue
            z = np.load(Path(cfg['work_dir']) / 'fusion' / p / 'attention' / f'seed{s}' / 'z_test.npy')
            sv = shap.TreeExplainer(model).shap_values(z)
            sv = sv[1] if isinstance(sv, list) else sv
            np.save(out / f'shap_{p}_seed{s}.npy', sv)
            imp = np.abs(sv).mean(0); order = np.argsort(-imp)
            cum = np.cumsum(imp[order]) / imp.sum()
            gain = model.get_booster().get_score(importance_type='gain')
            g = np.array([gain.get(f'f{i}', 0.0) for i in range(z.shape[1])])
            rho = spearmanr(imp, g).correlation
            boot = []
            for _ in range(100):
                j = rng.randint(0, len(z), len(z))
                boot.append(spearmanr(np.abs(sv[j]).mean(0), imp).correlation)
            rows.append(dict(model=p, seed=s, dims=z.shape[1], dims_for_80pct=int(np.searchsorted(cum, 0.8) + 1),
                             dims_for_90pct=int(np.searchsorted(cum, 0.9) + 1), spearman_shap_vs_gain=rho,
                             bootstrap_rank_stability=float(np.mean(boot)), top10=';'.join(f'F{i}' for i in order[:10])))
            if s == min(seeds):
                names = [f'F{i}' for i in range(z.shape[1])]
                for kind in ('dot', 'bar'):
                    plt.figure()
                    shap.summary_plot(sv, z, feature_names=names, plot_type=kind, show=False, max_display=20)
                    plt.tight_layout(); plt.savefig(out / f'figure_shap_{kind}_{p}_seed{s}.png', dpi=300); plt.close('all')
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(wd(cfg, 'results') / 'shap_summary.csv', index=False)


# ----------------------------------------------------------------- spatial maps
def sample_test(cfg):
    te = load_split(cfg)['test']; n = cfg['xai']['n_per_stratum']
    return te.groupby(['group', 'label'], group_keys=False).apply(
        lambda d: d.sample(min(len(d), n), random_state=0)).reset_index(drop=True)


def norm01(m):
    m = m - m.amin(dim=(-2, -1), keepdim=True)
    return m / (m.amax(dim=(-2, -1), keepdim=True) + 1e-12)


def maps_for(pipe, x, size):
    """Returns dict name -> (B, size, size) maps in [0,1], plus predicted class."""
    x = x.clone().requires_grad_(True)
    logits = pipe.logits_mlp(x, keep=True)
    c = logits.argmax(1)
    logits.gather(1, c[:, None]).sum().backward()
    out = {}
    for tag, bb in (('a', pipe.a), ('b', pipe.b)):
        A, G = bb.last_map, bb.last_map.grad
        w = G.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * A).sum(1, keepdim=True))
        out[f'gradcam_{tag}'] = norm01(F.interpolate(cam, size=size, mode='bilinear', align_corners=False)[:, 0]).detach()
    att = pipe.fusion.last_attn.detach()                     # (B, heads, N, N)
    recv = att.mean(1).mean(1)                               # (B, N): mean attention received
    na = GRID[pipe.a.name] ** 2
    parts = []
    for tag, bb, r in (('a', pipe.a, recv[:, :na]), ('b', pipe.b, recv[:, na:])):
        g = GRID[bb.name]
        mm = norm01(F.interpolate(r.reshape(-1, 1, g, g), size=size, mode='bilinear', align_corners=False)[:, 0])
        out[f'attention_{tag}'] = mm; parts.append(mm)
    out['attention'] = norm01((parts[0] + parts[1]) / 2)
    return out, c.detach()


def del_ins(pipe, x, m, cls, steps):
    """Deletion (to mean colour) and insertion (from a blurred copy) AUCs for one image."""
    import torchvision.transforms.functional as TF
    _, h, w = x.shape
    order = torch.argsort(m.flatten(), descending=True)
    blurred = TF.gaussian_blur(x[None], kernel_size=51, sigma=15.0)[0]
    dels, ins = [], []
    for k in range(steps + 1):
        n = int(round(k / steps * h * w))
        keep = torch.ones(h * w, device=x.device); keep[order[:n]] = 0
        keep = keep.reshape(1, h, w)
        dels.append(x * keep)                                  # deleted pixels -> 0 = ImageNet mean
        ins.append(blurred * keep + x * (1 - keep))
    pd_ = pipe.prob_fresh(torch.stack(dels)); pi_ = pipe.prob_fresh(torch.stack(ins))
    if int(cls) == 0:
        pd_, pi_ = 1 - pd_, 1 - pi_
    trap = lambda v: float(np.trapz(v, dx=1.0 / steps))
    return trap(pd_), trap(pi_)


def run_maps(cfg, seeds, dev):
    from scipy.stats import spearmanr
    df = sample_test(cfg); tf = build_transforms(cfg['image_size'], False); size = cfg['image_size']
    steps = cfg['xai']['deletion_steps']; out = wd(cfg, 'results', 'xai'); rows = []
    for p in PAIRS:
        store = {}
        for s in seeds:
            try:
                pipe = Pipeline(cfg, p, 'attention', s, dev)
            except FileNotFoundError as e:
                print(f'skip {p} seed {s}: {e}'); continue
            gen = torch.Generator().manual_seed(s)
            for i, r in df.iterrows():
                x = tf(load_rgb(Path(cfg['data_root']) / r.path))[None].to(dev)
                maps, cls = maps_for(pipe, x, (size, size))
                maps['random'] = torch.rand((1, size, size), generator=gen).to(dev)
                mp = Path(cfg['work_dir']) / 'masks' / Path(r.path).with_suffix('.png')
                mask = None
                if mp.exists():
                    from PIL import Image
                    mask = torch.from_numpy((np.asarray(Image.open(mp).convert('L').resize((size, size))) > 127).astype(np.float32)).to(dev)
                for name, m in maps.items():
                    if name in ('attention_a', 'attention_b'):
                        continue
                    d_auc, i_auc = del_ins(pipe, x[0], m[0], cls[0], steps)
                    e = float((m[0] * mask).sum() / (m[0].sum() + 1e-12)) if mask is not None else float('nan')
                    rows.append(dict(model=p, seed=s, path=r.path, group=r.group, label=r.label, pred=int(cls[0]),
                                     map=name, deletion_auc=d_auc, insertion_auc=i_auc, energy_in_mask=e))
                    if name != 'random':
                        store[(s, r.path, name)] = m[0].flatten().cpu().numpy()
                if i % 20 == 0:
                    print(f'{p} seed {s}: {i}/{len(df)}', flush=True)
            pd.DataFrame(rows).to_csv(out / 'xai_per_image.csv', index=False)
        # stability across seeds
        stab = []
        for name in ('gradcam_a', 'gradcam_b', 'attention'):
            for path in df.path:
                vs = [store[(s, path, name)] for s in seeds if (s, path, name) in store]
                for i in range(len(vs)):
                    for j in range(i + 1, len(vs)):
                        stab.append(dict(model=p, map=name, rho=spearmanr(vs[i], vs[j]).correlation))
        pd.DataFrame(stab).to_csv(out / f'stability_{p}.csv', index=False)
    res = pd.DataFrame(rows)
    names = {'gradcam_a': 'Grad-CAM, first backbone', 'gradcam_b': 'Grad-CAM, MobileViT-S', 'attention': 'Fusion attention',
             'random': 'Random ordering (reference)'}
    tab = []
    for p, t in (('cnn_tr', 'HM1'), ('tr_tr', 'HM2')):
        st = pd.read_csv(out / f'stability_{p}.csv') if (out / f'stability_{p}.csv').exists() else pd.DataFrame(columns=['map', 'rho'])
        for mname, label in names.items():
            d = res[(res.model == p) & (res.map == mname)]
            if not len(d):
                continue
            per_seed = d.groupby('seed')[['deletion_auc', 'insertion_auc', 'energy_in_mask']].mean()
            f = lambda c: f'{per_seed[c].mean():.3f} ± {per_seed[c].std():.3f}'
            rho = st[st['map'] == mname].rho
            tab.append({'Model': t, 'Map': label.replace('first backbone', 'EfficientNetV2-M' if p == 'cnn_tr' else 'DeiT-Small'),
                        'Deletion AUC (lower is better)': f('deletion_auc'), 'Insertion AUC (higher is better)': f('insertion_auc'),
                        'Energy inside food mask': f('energy_in_mask'),
                        'Seed-to-seed Spearman ρ': f'{rho.mean():.3f} ± {rho.std():.3f}' if len(rho) else '—'})
    pd.DataFrame(tab).to_csv(wd(cfg, 'results') / 'table_xai.csv', index=False)
    grp = res[res.map != 'random'].groupby(['model', 'map', 'group', 'label']).energy_in_mask.mean().reset_index()
    grp.to_csv(wd(cfg, 'results') / 'xai_energy_by_group_label.csv', index=False)
    print(pd.DataFrame(tab).to_string(index=False))


def run_figures(cfg, dev):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df = sample_test(cfg).groupby(['group', 'label'], group_keys=False).head(1)
    tf = build_transforms(cfg['image_size'], False); size = cfg['image_size']; cmap = cfg['xai']['colormap']
    s = min(cfg['seeds']); out = wd(cfg, 'results', 'xai')
    for p in PAIRS:
        pipe = Pipeline(cfg, p, 'attention', s, dev)
        cols = ['image', 'gradcam_a', 'gradcam_b', 'attention']
        fig, ax = plt.subplots(len(df), 4, figsize=(10, 2.5 * len(df)), squeeze=False)
        for i, r in enumerate(df.itertuples()):
            img = load_rgb(Path(cfg['data_root']) / r.path).resize((size, size))
            maps, cls = maps_for(pipe, tf(img)[None].to(dev), (size, size))
            for j, c in enumerate(cols):
                ax[i, j].imshow(img)
                if c != 'image':
                    im = ax[i, j].imshow(maps[c][0].cpu().numpy(), cmap=cmap, vmin=0, vmax=1, alpha=0.5)
                ax[i, j].axis('off')
                if i == 0:
                    ax[i, j].set_title({'image': 'Image', 'gradcam_a': f'Grad-CAM {pipe.a.name}',
                                        'gradcam_b': f'Grad-CAM {pipe.b.name}', 'attention': 'Fusion attention'}[c], fontsize=9)
            ax[i, 0].text(-5, size / 2, f"{r.group}\n{'Good' if r.label else 'Bad'} / pred {'Good' if int(cls[0]) else 'Bad'}",
                          ha='right', va='center', fontsize=8)
        cb = fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.3); cb.set_label('normalised importance (0–1)')
        fig.savefig(out / f'figure_maps_{p}_seed{s}.png', dpi=300, bbox_inches='tight'); plt.close(fig)
        print(f'Wrote {out}/figure_maps_{p}_seed{s}.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    ap.add_argument('--shap', action='store_true'); ap.add_argument('--maps', action='store_true')
    ap.add_argument('--figures', action='store_true')
    ap.add_argument('--seeds', nargs='*', type=int); ap.add_argument('--device', default='auto')
    a = ap.parse_args()
    cfg = load_config(a.config); dev = get_device(a.device); seeds = a.seeds if a.seeds is not None else cfg['seeds']
    if a.shap:
        run_shap(cfg, seeds)
    if a.maps:
        run_maps(cfg, seeds, dev)
    if a.figures:
        run_figures(cfg, dev)


if __name__ == '__main__':
    main()
