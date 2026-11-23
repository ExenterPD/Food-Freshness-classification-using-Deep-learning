"""Step 6. Food masks, shortcut tests and photometric robustness.

    python s08_robustness.py --make-masks            # all images; writes runs/masks/ + review sheet
    python s08_robustness.py --evaluate              # final models, every seed, all test conditions
    python s08_robustness.py --probe                 # background-only probe (frozen ImageNet features)

Masks: rembg (U^2-Net salient-object model, `pip install rembg`) or OpenCV GrabCut as a fallback.
Check a sample by hand in runs/masks/review.html; correct a mask by editing its PNG (white = food)
and record it in runs/masks/review.csv (columns checked, corrected).
Food-only = background replaced; background-only = food replaced; fill = ImageNet mean colour or blur.
Writes runs/results/table_robustness.csv and robustness_runs.csv.
"""
import os; os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')  # torch 2.0 on Apple GPUs: run missing ops on CPU (must precede `import torch`)
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from common import (PAIRS, PAIR_NAMES, binary_metrics, build_transforms, get_device, load_config, load_rgb,
                    load_split, make_dataset, make_loader, wd)

MEAN_RGB = (124, 116, 104)


def mask_path(cfg, rel):
    return Path(cfg['work_dir']) / 'masks' / (Path(rel).with_suffix('.png'))


def make_masks(cfg, method, limit=None):
    sp = load_split(cfg); df = pd.concat(sp.values())
    if limit:
        df = df.head(limit)
    if method == 'rembg':
        try:
            from rembg import new_session, remove
            sess = new_session('u2net')
        except ImportError:
            raise SystemExit('rembg is not installed: pip install rembg  (or use --mask-method grabcut)')
    import cv2
    for i, rel in enumerate(df.path):
        out = mask_path(cfg, rel)
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        img = load_rgb(Path(cfg['data_root']) / rel)
        if method == 'rembg':
            m = np.asarray(remove(img, session=sess, only_mask=True))
            m = (m > 127).astype(np.uint8) * 255
        else:
            a = np.asarray(img)[:, :, ::-1].copy(); h, w = a.shape[:2]
            mk = np.zeros((h, w), np.uint8); bg, fg = np.zeros((1, 65)), np.zeros((1, 65))
            rect = (int(0.05 * w), int(0.05 * h), int(0.9 * w), int(0.9 * h))
            cv2.grabCut(a, mk, rect, bg, fg, 5, cv2.GC_INIT_WITH_RECT)
            m = np.where((mk == 1) | (mk == 3), 255, 0).astype(np.uint8)
        Image.fromarray(m).save(out)
        if i % 200 == 0:
            print(f'{i}/{len(df)} masks', flush=True)
    # review sheet: stratified sample of test images
    te = sp['test']
    sample = te.groupby(['group', 'label'], group_keys=False).apply(lambda d: d.sample(min(len(d), 10), random_state=0))
    rv = Path(cfg['work_dir']) / 'masks' / 'review.csv'
    if not rv.exists():
        sample[['path', 'group', 'label']].assign(checked=0, corrected=0).to_csv(rv, index=False)
    parts = ['<html><body style="font:12px sans-serif"><h2>Mask review: tick checked/corrected in review.csv</h2>']
    for rel in pd.read_csv(rv).path:
        if not mask_path(cfg, rel).exists():  # e.g. a --limit test run
            continue
        img = load_rgb(Path(cfg['data_root']) / rel).resize((224, 224))
        m = Image.open(mask_path(cfg, rel)).convert('L').resize((224, 224))
        ov = Image.composite(img, Image.new('RGB', img.size, (255, 0, 255)), m)
        p = Path(cfg['work_dir']) / 'masks' / 'review_png' / Path(rel).with_suffix('.png').name
        p.parent.mkdir(parents=True, exist_ok=True); ov.save(p)
        parts.append(f'<div style="display:inline-block;margin:3px"><img src="review_png/{p.name}"><br>{rel}</div>')
    (Path(cfg['work_dir']) / 'masks' / 'review.html').write_text('\n'.join(parts) + '</body></html>')
    print('Masks done. Review runs/masks/review.html and fill review.csv.')


def apply_mask(img, mask, keep, fill):
    m = mask.convert('L').resize(img.size)
    if fill == 'blur':
        filler = img.filter(ImageFilter.GaussianBlur(radius=max(img.size) / 20))
    else:
        filler = Image.new('RGB', img.size, MEAN_RGB)
    return Image.composite(img, filler, m) if keep == 'food' else Image.composite(filler, img, m)


def conditions(cfg):
    rc = cfg['robustness']; c = [('Original', None), ('Food-only (background replaced)', ('food',)),
                                 ('Background-only (food masked)', ('background',))]
    c += [(f'Brightness ×{f}', ('bright', f)) for f in rc['brightness']]
    c += [(f'Contrast ×{f}', ('contrast', f)) for f in rc['contrast']]
    c += [(f'Gaussian blur σ = {s}', ('blur', s)) for s in rc['blur_sigma']]
    return c


def pil_fn_for(cfg, cond):
    fill = cfg['robustness']['fill']
    if cond is None:
        return None

    def fn(img, rel):
        if cond[0] in ('food', 'background'):
            return apply_mask(img, Image.open(mask_path(cfg, rel)), cond[0], fill)
        if cond[0] == 'bright':
            return ImageEnhance.Brightness(img).enhance(cond[1])
        if cond[0] == 'contrast':
            return ImageEnhance.Contrast(img).enhance(cond[1])
        return img.filter(ImageFilter.GaussianBlur(radius=cond[1]))
    return fn


def evaluate(cfg, dev, seeds):
    from models import Pipeline
    sp = load_split(cfg); te = sp['test']; tf = build_transforms(cfg['image_size'], False); rows = []
    for p in PAIRS:
        for s in seeds:
            try:
                pipe = Pipeline(cfg, p, 'attention', s, dev)
            except FileNotFoundError as e:
                print(f'skip {p} seed {s}: {e}'); continue
            for name, cond in conditions(cfg):
                dl = make_loader(make_dataset(te, cfg['data_root'], tf, pil_fn_for(cfg, cond)), 32, False, 0)
                prob = np.concatenate([pipe.prob_fresh(x) for x, _, _ in dl])
                m = binary_metrics(te.label.values, prob)
                rows.append(dict(model=p, seed=s, condition=name, accuracy=m['accuracy'], macro_f1=m['macro_f1'], auroc=m['auroc']))
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows); out = wd(cfg, 'results'); f = out / 'robustness_runs.csv'
    if f.exists() and len(df):  # keep earlier rows for seeds not run this time, so seeds can be added later
        old = pd.read_csv(f)
        old = old[~old.set_index(['model', 'seed']).index.isin(df.set_index(['model', 'seed']).index)]
        df = pd.concat([old, df], ignore_index=True).sort_values(['model', 'seed'], kind='stable')
    df.to_csv(f, index=False)
    return df


def probe(cfg, dev):
    """Logistic regression on frozen ImageNet EfficientNetV2-M features of background-only images."""
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from models import Backbone
    sp = load_split(cfg); tf = build_transforms(cfg['image_size'], False)
    m = Backbone('effnetv2m', pretrained=not cfg.get('_no_pretrained')).to(dev).eval()
    feats = {}
    for k in ('train', 'test'):
        dl = make_loader(make_dataset(sp[k], cfg['data_root'], tf, pil_fn_for(cfg, ('background',))), 32, False, 0)
        with torch.no_grad():
            feats[k] = np.concatenate([m.pooled(x.to(dev)).float().cpu().numpy() for x, _, _ in dl])
    sc = StandardScaler().fit(feats['train'])
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(feats['train']), sp['train'].label.values)
    prob = clf.predict_proba(sc.transform(feats['test']))[:, 1]
    r = binary_metrics(sp['test'].label.values, prob)
    base = max(sp['test'].label.mean(), 1 - sp['test'].label.mean())
    out = wd(cfg, 'results')
    pd.DataFrame([dict(condition='Background-only probe (separate model)', accuracy=r['accuracy'], macro_f1=r['macro_f1'],
                       auroc=r['auroc'], majority_class_rate=base)]).to_csv(out / 'background_probe.csv', index=False)
    print(f'Background-only probe: accuracy {r["accuracy"]:.4f}, macro-F1 {r["macro_f1"]:.4f}; majority-class rate {base:.4f}')


def table(cfg):
    out = wd(cfg, 'results'); f = out / 'robustness_runs.csv'
    if not f.exists():
        return
    df = pd.read_csv(f); rows = []
    for cond in df.condition.unique():
        row = {'Test condition': cond}
        for p, t in (('cnn_tr', 'HM1'), ('tr_tr', 'HM2')):
            d = df[(df.condition == cond) & (df.model == p)]
            for mcol, lab in (('accuracy', 'Accuracy'), ('macro_f1', 'Macro-F1')):
                row[f'{t} {lab}'] = f'{d[mcol].mean() * 100:.2f} ± {d[mcol].std() * 100:.2f}' if len(d) else '—'
        rows.append(row)
    if (out / 'background_probe.csv').exists():
        b = pd.read_csv(out / 'background_probe.csv').iloc[0]
        rows.append({'Test condition': b.condition, 'HM1 Accuracy': f'{b.accuracy * 100:.2f}', 'HM1 Macro-F1': f'{b.macro_f1 * 100:.2f}',
                     'HM2 Accuracy': '(same probe)', 'HM2 Macro-F1': ''})
    pd.DataFrame(rows).to_csv(out / 'table_robustness.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    ap.add_argument('--make-masks', action='store_true')
    ap.add_argument('--mask-method')
    ap.add_argument('--evaluate', action='store_true')
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--seeds', nargs='*', type=int)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--no-pretrained', action='store_true')
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    cfg = load_config(a.config); cfg['_no_pretrained'] = a.no_pretrained; dev = get_device(a.device)
    if a.make_masks:
        make_masks(cfg, a.mask_method or cfg['robustness']['mask_method'], a.limit)
    if a.evaluate:
        evaluate(cfg, dev, a.seeds if a.seeds is not None else cfg['seeds'])
    if a.probe:
        probe(cfg, dev)
    table(cfg)


if __name__ == '__main__':
    main()
