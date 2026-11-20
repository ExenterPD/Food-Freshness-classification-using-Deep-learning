"""Shared helpers: config, paths, seeding, devices, image loading, datasets, metrics."""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}

PAIRS = {
    'cnn_tr': ('effnetv2m', 'mobilevit_s'),   # Hybrid Model 1
    'tr_tr': ('deit_s', 'mobilevit_s'),       # Hybrid Model 2
}
PAIR_NAMES = {'cnn_tr': 'Hybrid Model 1', 'tr_tr': 'Hybrid Model 2'}
BACKBONES = ['effnetv2m', 'mobilevit_s', 'deit_s']

try:  # optional HEIC support
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pass


# ----------------------------------------------------------------- config / paths
def load_config(path=None):
    path = Path(path) if path else HERE / 'config.yaml'
    with open(path) as f:
        cfg = yaml.safe_load(f)
    local = path.resolve().parent / 'config.local.yaml'   # per-machine overrides (not in git)
    if local.exists():
        with open(local) as f:
            over = yaml.safe_load(f) or {}
        for k, v in over.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    base = path.resolve().parent
    cfg['data_root'] = (base / cfg['data_root']).resolve()
    cfg['work_dir'] = (base / cfg['work_dir']).resolve()
    cfg['work_dir'].mkdir(parents=True, exist_ok=True)
    return cfg


def wd(cfg, *parts):
    p = Path(cfg['work_dir']).joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------- seeding / device
def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(name='auto'):
    import torch
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def sync(device):
    import torch
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps' and hasattr(torch, 'mps'):
        torch.mps.synchronize()


def device_name(device):
    import platform
    import torch
    if device.type == 'cuda':
        return torch.cuda.get_device_name(0)
    if device.type == 'mps':
        return 'Apple ' + platform.processor() + ' (MPS) ' + platform.platform()
    return platform.processor() or platform.machine()


# ----------------------------------------------------------------- images
def load_rgb(path):
    with Image.open(path) as im:
        return im.convert('RGB')


def build_transforms(size, train):
    import torchvision.transforms as T
    if train:
        return T.Compose([
            T.Resize((size, size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def make_dataset(df, root, transform, pil_fn=None):
    """A separate Dataset per split (fixes the shared-transform bug of the old scripts)."""
    import torch
    from torch.utils.data import Dataset

    class _DS(Dataset):
        def __init__(self):
            self.paths = df['path'].tolist()
            self.labels = df['label'].astype(int).tolist()

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            img = load_rgb(Path(root) / self.paths[i])
            if pil_fn is not None:
                img = pil_fn(img, self.paths[i])
            return transform(img), torch.tensor(self.labels[i]), i

    return _DS()


def make_loader(ds, batch_size, shuffle, seed, num_workers=0):
    import torch
    from torch.utils.data import DataLoader
    g = torch.Generator()
    g.manual_seed(seed)

    def _wif(worker_id):
        s = seed * 1000 + worker_id
        np.random.seed(s)
        random.seed(s)

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g,
                      num_workers=num_workers, worker_init_fn=_wif, drop_last=False)


def load_split(cfg):
    p = Path(cfg['work_dir']) / 'split' / 'split.csv'
    if not p.exists():
        raise SystemExit('split.csv not found. Run 01_audit_split.py first.')
    df = pd.read_csv(p)
    return {k: df[df.partition == k].reset_index(drop=True) for k in ('train', 'val', 'test')}


# ----------------------------------------------------------------- metrics
def binary_metrics(y, p_fresh, pred=None):
    """All metrics with Fresh (label 1) as the positive class."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    p_fresh = np.asarray(p_fresh, dtype=float)
    pred = (p_fresh >= 0.5).astype(int) if pred is None else np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    n = len(y)

    def div(a, b):
        return a / b if b else float('nan')

    prec_f, rec_f = div(tp, tp + fp), div(tp, tp + fn)
    prec_s, rec_s = div(tn, tn + fn), div(tn, tn + fp)
    f1_f = div(2 * prec_f * rec_f, prec_f + rec_f) if not np.isnan(prec_f + rec_f) else float('nan')
    f1_s = div(2 * prec_s * rec_s, prec_s + rec_s) if not np.isnan(prec_s + rec_s) else float('nan')
    n_f, n_s = int((y == 1).sum()), int((y == 0).sum())
    macro = np.nanmean([f1_f, f1_s])
    weighted = div(np.nansum([f1_f * n_f, f1_s * n_s]), n)
    try:
        auroc = roc_auc_score(y, p_fresh) if len(np.unique(y)) == 2 else float('nan')
    except ValueError:
        auroc = float('nan')
    return dict(n=n, n_fresh=n_f, n_spoiled=n_s, tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=div(tp + tn, n), precision_fresh=prec_f, recall_fresh=rec_f,
                specificity=rec_s, fpr=div(fp, fp + tn), precision_spoiled=prec_s,
                f1_fresh=f1_f, f1_spoiled=f1_s, macro_f1=macro, weighted_f1=weighted, auroc=auroc)


def weighted_f1(y, pred):
    from sklearn.metrics import f1_score
    return f1_score(y, pred, average='weighted', zero_division=0)


def write_preds(path, df_split, p_fresh, meta=None):
    out = df_split[['path', 'group', 'subcategory', 'label']].copy()
    out['prob_fresh'] = np.asarray(p_fresh, dtype=float)
    out['pred'] = (out['prob_fresh'] >= 0.5).astype(int)
    out.to_csv(path, index=False)
    if meta is not None:
        save_json(meta, Path(path).parent / 'meta.json')
    return out


class Timer:
    def __enter__(self):
        self.t0 = time.time(); return self

    def __exit__(self, *a):
        self.s = time.time() - self.t0
