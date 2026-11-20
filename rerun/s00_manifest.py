"""Step 1a. Walk the dataset, hash every image, write manifest.csv and print the category tree.

Folder layout handled: <group>/[<sub>/...]/Good|Bad/<image>. The label comes from the folder
named Good or Bad (any case); the group is the first folder; anything between is the subcategory.

    python s00_manifest.py [--config config.yaml] [--workers 4]
"""
import os; os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')  # torch 2.0 on Apple GPUs: run missing ops on CPU (must precede `import torch`)
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from common import IMAGE_EXTS, load_config, load_rgb, wd


def scan(root):
    rows = []
    for p in sorted(Path(root).rglob('*')):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS or p.name.startswith('._'):
            continue
        rel = p.relative_to(root)
        parts = [s for s in rel.parts[:-1]]
        lab_idx = [i for i, s in enumerate(parts) if s.lower() in ('good', 'bad')]
        if not lab_idx:
            continue
        li = lab_idx[-1]
        label = 1 if parts[li].lower() == 'good' else 0
        group = parts[0] if li > 0 else '(root)'
        sub = '/'.join(parts[1:li]) if li > 1 else ''
        rows.append(dict(path=rel.as_posix(), group=group, subcategory=sub, label=label, ext=p.suffix.lower()))
    return rows


def clean(v):
    """EXIF strings can carry NUL bytes and other control characters that CSV cannot store."""
    if isinstance(v, bytes):
        v = v.decode('utf-8', 'ignore')
    return ''.join(ch for ch in str(v) if ch.isprintable()).strip()


def hash_one(args):
    root, rel = args
    import imagehash
    try:
        img = load_rgb(Path(root) / rel)
        arr = np.asarray(img)
        sha = hashlib.sha256(arr.tobytes() + str(arr.shape).encode()).hexdigest()
        ph = str(imagehash.phash(img, hash_size=8))
        w, h = img.size
        exif_time, exif_model = '', ''
        try:
            from PIL import Image
            with Image.open(Path(root) / rel) as im:
                ex = im.getexif()
                exif_time = clean(ex.get(36867) or ex.get(306) or '')
                exif_model = clean(ex.get(272) or '')
        except Exception:
            pass
        return dict(path=rel, width=w, height=h, sha256=sha, phash=ph, exif_time=exif_time,
                    exif_model=exif_model, error='')
    except Exception as e:  # unreadable file
        return dict(path=rel, width=0, height=0, sha256='', phash='', exif_time='', exif_model='', error=str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    cfg = load_config(a.config)
    root = cfg['data_root']
    if not Path(root).exists():
        raise SystemExit(f'data_root not found: {root}  (edit config.yaml)')
    rows = scan(root)
    print(f'Found {len(rows)} labelled images under {root}')
    with ProcessPoolExecutor(a.workers) as ex:
        hashed = list(ex.map(hash_one, [(str(root), r['path']) for r in rows], chunksize=16))
    df = pd.DataFrame(rows).merge(pd.DataFrame(hashed), on='path')
    bad = df[df.error != '']
    if len(bad):
        print(f'WARNING: {len(bad)} unreadable files, excluded (listed in unreadable.csv)')
        bad.to_csv(wd(cfg, 'split') / 'unreadable.csv', index=False)
    df = df[df.error == ''].drop(columns='error')
    for c in df.columns:  # belt and braces: no control characters anywhere in the CSV
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: clean(v) if isinstance(v, str) else v)
    out = wd(cfg, 'split') / 'manifest.csv'
    df.to_csv(out, index=False)

    print('\nCategory tree (Good / Bad / total):')
    for g, gdf in df.groupby('group'):
        print(f'  {g}: {int(gdf.label.sum())} / {int((gdf.label == 0).sum())} / {len(gdf)}')
        for s, sdf in gdf.groupby('subcategory'):
            if s:
                print(f'      {s}: {int(sdf.label.sum())} / {int((sdf.label == 0).sum())} / {len(sdf)}')
    print(f'TOTAL: {int(df.label.sum())} Good / {int((df.label == 0).sum())} Bad / {len(df)}')
    print('File types:', df.ext.value_counts().to_dict())
    print('Images with EXIF capture time:', int((df.exif_time != '').sum()))
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
