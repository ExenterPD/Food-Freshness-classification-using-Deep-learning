"""Step 2. Fine-tune ONE backbone with a linear Good/Bad head (all layers trainable).

    python s02_train_backbone.py --backbone effnetv2m --seed 0 [--amp] [--batch 8 --accum 2]
    python s02_train_backbone.py --backbone deit_s --seed 0 --epochs 20 --tag _long   # convergence run

AdamW (lr 1e-4, wd 1e-5), cosine schedule, cross-entropy with label smoothing 0.1, batch 16,
best checkpoint by validation weighted F1, early stopping. Training images are augmented;
validation and test images are not (separate dataset objects).

Outputs in runs/backbones/<backbone>/seed<s><tag>/:
  best.pt, training_log.csv, curves.png, val_preds.csv, test_preds.csv, meta.json
"""
import os; os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')  # torch 2.0 on Apple GPUs: run missing ops on CPU (must precede `import torch`)
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import (BACKBONES, binary_metrics, build_transforms, device_name, get_device, load_config,
                    load_split, make_dataset, make_loader, save_json, seed_everything, sync, wd,
                    weighted_f1, write_preds)
from models import BackboneClassifier, total_params


def run_epoch(model, dl, dev, opt=None, crit=None, scaler=None, accum=1, amp=False):
    train = opt is not None
    model.train(train)
    probs, ys, losses = [], [], []
    if train:
        opt.zero_grad(set_to_none=True)
    for step, (x, y, _) in enumerate(dl):
        x, y = x.to(dev), y.to(dev)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type='cuda', enabled=amp and dev.type == 'cuda'):
                out = model(x)
                loss = crit(out.float(), y)
            if train:
                if scaler is not None:
                    scaler.scale(loss / accum).backward()
                else:
                    (loss / accum).backward()
                if (step + 1) % accum == 0 or step + 1 == len(dl):
                    if scaler is not None:
                        scaler.step(opt); scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)
        losses.append(loss.item() * len(y))
        probs.append(torch.softmax(out.detach().float(), 1)[:, 1].cpu().numpy())
        ys.append(y.cpu().numpy())
    p = np.concatenate(probs); y = np.concatenate(ys)
    return sum(losses) / len(y), y, p


def plot(log, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(log.epoch, log.train_loss, label='train'); ax[0].plot(log.epoch, log.val_loss, label='val')
    ax[0].set_xlabel('epoch'); ax[0].set_ylabel('loss'); ax[0].legend()
    ax[1].plot(log.epoch, log.val_macro_f1, label='val macro-F1'); ax[1].plot(log.epoch, log.val_accuracy, label='val accuracy')
    ax[1].set_xlabel('epoch'); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config')
    ap.add_argument('--backbone', required=True, choices=BACKBONES)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--tag', default='')
    ap.add_argument('--batch', type=int)
    ap.add_argument('--accum', type=int, default=1)
    ap.add_argument('--amp', action='store_true', help='mixed precision (CUDA only)')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--no-pretrained', action='store_true', help='smoke tests only')
    ap.add_argument('--limit', type=int, help='smoke tests: use only N images per split')
    a = ap.parse_args()
    cfg = load_config(a.config); bc = cfg['backbone']
    epochs = a.epochs or bc['epochs']; bs = a.batch or bc['batch_size']
    seed_everything(a.seed)
    dev = get_device(a.device)
    out = wd(cfg, 'backbones', a.backbone, f'seed{a.seed}{a.tag}')
    sp = load_split(cfg)
    if a.limit:
        sp = {k: v.groupby('label', group_keys=False).apply(lambda d: d.head(a.limit // 2)).reset_index(drop=True)
              for k, v in sp.items()}
    root, size = cfg['data_root'], cfg['image_size']
    dl_tr = make_loader(make_dataset(sp['train'], root, build_transforms(size, True)), bs, True, a.seed, bc['num_workers'])
    dl_va = make_loader(make_dataset(sp['val'], root, build_transforms(size, False)), bs, False, a.seed, bc['num_workers'])
    dl_te = make_loader(make_dataset(sp['test'], root, build_transforms(size, False)), bs, False, a.seed, bc['num_workers'])

    model = BackboneClassifier(a.backbone, pretrained=not a.no_pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=bc['lr'], weight_decay=bc['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=bc['label_smoothing'])
    scaler = torch.cuda.amp.GradScaler() if (a.amp and dev.type == 'cuda') else None

    print(f'{a.backbone} seed {a.seed} on {dev} ({device_name(dev)}); train/val/test = '
          f'{len(sp["train"])}/{len(sp["val"])}/{len(sp["test"])}; epochs {epochs}; batch {bs}x{a.accum}')
    log, best, best_ep, bad = [], -1.0, 0, 0
    t_start = time.time()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, ytr, ptr = run_epoch(model, dl_tr, dev, opt, crit, scaler, a.accum, a.amp)
        va_loss, yva, pva = run_epoch(model, dl_va, dev, crit=crit, amp=a.amp)
        sync(dev)
        lr = opt.param_groups[0]['lr']; sched.step()
        m = binary_metrics(yva, pva)
        wf1 = weighted_f1(yva, (pva >= 0.5).astype(int))
        row = dict(epoch=ep, train_loss=tr_loss, val_loss=va_loss,
                   train_accuracy=float(((ptr >= 0.5) == ytr).mean()), val_accuracy=m['accuracy'],
                   val_weighted_f1=wf1, val_macro_f1=m['macro_f1'], val_auroc=m['auroc'], lr=lr,
                   time_s=time.time() - t0)
        log.append(row)
        print(' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()), flush=True)
        if wf1 > best:
            best, best_ep, bad = wf1, ep, 0
            torch.save({'model': model.state_dict(), 'epoch': ep, 'val_weighted_f1': wf1}, out / 'best.pt')
        else:
            bad += 1
            if bad >= bc['early_stop_patience']:
                print(f'Early stop at epoch {ep}'); break
        pd.DataFrame(log).to_csv(out / 'training_log.csv', index=False)
    total_time = time.time() - t_start
    log = pd.DataFrame(log); log.to_csv(out / 'training_log.csv', index=False); plot(log, out / 'curves.png')

    model.load_state_dict(torch.load(out / 'best.pt', map_location='cpu')['model'])
    _, _, pva = run_epoch(model, dl_va, dev, crit=crit)
    _, _, pte = run_epoch(model, dl_te, dev, crit=crit)
    meta = dict(stage='backbone', model=a.backbone, classifier='linear', seed=a.seed, tag=a.tag,
                best_epoch=best_ep, epochs_run=int(log.epoch.max()), epochs_planned=epochs,
                best_val_weighted_f1=best, train_time_s=total_time, mean_epoch_time_s=float(log.time_s.mean()),
                device=str(dev), device_name=device_name(dev), batch=bs, accum=a.accum, amp=bool(scaler),
                params=total_params(model), pretrained=not a.no_pretrained, torch=torch.__version__)
    write_preds(out / 'val_preds.csv', sp['val'], pva)
    write_preds(out / 'test_preds.csv', sp['test'], pte, meta)
    save_json(meta, out / 'meta.json')
    if best_ep == int(log.epoch.max()):
        print('NOTE: best validation epoch is the last epoch; the run may not have converged.')
    print(f'Done. Best epoch {best_ep}, val weighted F1 {best:.4f}. Outputs in {out}')


if __name__ == '__main__':
    main()
