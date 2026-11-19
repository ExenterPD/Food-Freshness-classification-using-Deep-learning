"""Backbones (with token access), fusion operators and the end-to-end inference pipeline."""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import PAIRS, load_json

TOKEN_DIMS = {'effnetv2m': (49, 1280), 'mobilevit_s': (49, 640), 'deit_s': (196, 384)}
GRID = {'effnetv2m': 7, 'mobilevit_s': 7, 'deit_s': 14}


class Backbone(nn.Module):
    """Wraps a pretrained network and exposes last-stage tokens (B, N, C).

    effnetv2m   : torchvision EfficientNetV2-M, final 1x1 conv map 7x7x1280 -> 49 tokens
    mobilevit_s : timm MobileViT-S, final 1x1 conv map 7x7x640 -> 49 tokens
    deit_s      : timm deit_small_patch16_224 (non-distilled), 196 patch tokens of width 384
    """

    def __init__(self, name, pretrained=True):
        super().__init__()
        self.name = name
        if name == 'effnetv2m':
            from torchvision.models import efficientnet_v2_m
            try:
                from torchvision.models import EfficientNet_V2_M_Weights
                w = EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
                net = efficientnet_v2_m(weights=w)
            except ImportError:  # very old torchvision
                net = efficientnet_v2_m(pretrained=pretrained)
            self.body = net.features
            self.dim = 1280
        elif name == 'mobilevit_s':
            import timm
            self.body = timm.create_model('mobilevit_s', pretrained=pretrained, num_classes=0)
            self.dim = self.body.num_features
        elif name == 'deit_s':
            import timm
            self.body = timm.create_model('deit_small_patch16_224', pretrained=pretrained, num_classes=0)
            self.dim = self.body.num_features
        else:
            raise ValueError(name)
        self.last_map = None  # spatial map kept for Grad-CAM

    def feature_map(self, x):
        """Returns a spatial map (B, C, H, W) for every backbone."""
        if self.name == 'effnetv2m':
            return self.body(x)
        if self.name == 'mobilevit_s':
            return self.body.forward_features(x)
        tok = self.body.forward_features(x)          # (B, 1+196, C)
        patches = tok[:, self.body.num_prefix_tokens:]
        b, n, c = patches.shape
        s = int(round(n ** 0.5))
        return patches.transpose(1, 2).reshape(b, c, s, s)

    def tokens(self, x, keep_map=False):
        if self.name == 'deit_s':
            tok = self.body.forward_features(x)
            cls = tok[:, 0]
            patches = tok[:, self.body.num_prefix_tokens:]
            if keep_map:
                b, n, c = patches.shape
                s = int(round(n ** 0.5))
                m = patches.transpose(1, 2).reshape(b, c, s, s)
                if m.requires_grad:
                    m.retain_grad()
                self.last_map = m
                patches = m.flatten(2).transpose(1, 2)
            return patches, cls
        m = self.feature_map(x)
        if keep_map:
            if m.requires_grad:
                m.retain_grad()
            self.last_map = m
        tok = m.flatten(2).transpose(1, 2)            # (B, HW, C)
        return tok, tok.mean(1)

    def pooled(self, x):
        return self.tokens(x)[1]


class BackboneClassifier(nn.Module):
    """Stage 1: one backbone + linear Good/Bad head."""

    def __init__(self, name, pretrained=True):
        super().__init__()
        self.backbone = Backbone(name, pretrained)
        self.head = nn.Linear(self.backbone.dim, 2)

    def forward(self, x):
        return self.head(self.backbone.pooled(x))


class Fusion(nn.Module):
    """Stage 3: projections + one of {concat, gated, attention} + MLP head.

    attention (Eqs. 4-9 in the revised paper): tokens of both backbones are projected to width d,
    a modality embedding is added, multi-head self-attention runs over ALL N_a + N_b tokens,
    then residual + LayerNorm, then per-source mean pooling -> z in R^{2d}.
    concat: z = [mean(T_a); mean(T_b)] in R^{2d}.   gated: z = g*p_a + (1-g)*p_b in R^d.
    """

    def __init__(self, dim_a, dim_b, kind, d=256, heads=8, hidden=256, dropout=0.2):
        super().__init__()
        self.kind = kind
        self.proj_a = nn.Linear(dim_a, d)
        self.proj_b = nn.Linear(dim_b, d)
        self.emb_a = nn.Parameter(torch.zeros(d))
        self.emb_b = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.emb_a, std=0.02); nn.init.normal_(self.emb_b, std=0.02)
        if kind == 'attention':
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.norm = nn.LayerNorm(d)
            out = 2 * d
        elif kind == 'gated':
            self.gate = nn.Linear(2 * d, d)
            out = d
        elif kind == 'concat':
            out = 2 * d
        else:
            raise ValueError(kind)
        self.out_dim = out
        self.head = nn.Sequential(nn.Linear(out, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))
        self.last_attn = None

    def embed(self, ta, tb, keep_attn=False):
        a = self.proj_a(ta) + self.emb_a
        b = self.proj_b(tb) + self.emb_b
        na = a.shape[1]
        if self.kind == 'attention':
            t = torch.cat([a, b], dim=1)                      # (B, Na+Nb, d)
            h, w = self.attn(t, t, t, need_weights=keep_attn, average_attn_weights=False)
            if keep_attn:
                self.last_attn = w                            # (B, heads, N, N)
            z = self.norm(t + h)
            return torch.cat([z[:, :na].mean(1), z[:, na:].mean(1)], dim=1)
        pa, pb = a.mean(1), b.mean(1)
        if self.kind == 'concat':
            return torch.cat([pa, pb], dim=1)
        g = torch.sigmoid(self.gate(torch.cat([pa, pb], dim=1)))
        return g * pa + (1 - g) * pb

    def forward(self, ta, tb):
        return self.head(self.embed(ta, tb))


def trainable_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def total_params(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------- loading trained parts
def backbone_ckpt(cfg, name, seed, tag=''):
    return Path(cfg['work_dir']) / 'backbones' / name / f'seed{seed}{tag}' / 'best.pt'


def load_backbone(cfg, name, seed, device, pretrained=False):
    m = BackboneClassifier(name, pretrained=pretrained)
    sd = torch.load(backbone_ckpt(cfg, name, seed), map_location='cpu')
    m.load_state_dict(sd['model'])
    return m.to(device).eval()


def fusion_dir(cfg, pair, kind, seed):
    return Path(cfg['work_dir']) / 'fusion' / pair / kind / f'seed{seed}'


def load_fusion(cfg, pair, kind, seed, device):
    d = fusion_dir(cfg, pair, kind, seed)
    meta = load_json(d / 'meta.json')
    a, b = PAIRS[pair]
    f = Fusion(TOKEN_DIMS[a][1], TOKEN_DIMS[b][1], kind, meta['d'], meta['heads'], meta['mlp_hidden'], meta['dropout'])
    f.load_state_dict(torch.load(d / 'best.pt', map_location='cpu'))
    return f.to(device).eval(), meta


def xgb_dir(cfg, pair, kind, seed, method, opt_seed=None):
    opt = f'_opt{opt_seed}' if opt_seed is not None else ''
    return Path(cfg['work_dir']) / 'xgb' / pair / kind / f'seed{seed}' / f'{method}{opt}'


def load_xgb(cfg, pair, kind, seed, method):
    import xgboost as xgb
    opt = None if method == 'default' else seed
    d = xgb_dir(cfg, pair, kind, seed, method, opt)
    m = xgb.XGBClassifier()
    m.load_model(str(d / 'model.json'))
    return m, d


class Pipeline(nn.Module):
    """End-to-end deployed model: two frozen backbones -> fusion -> XGBoost (or MLP head)."""

    def __init__(self, cfg, pair, kind, seed, device, classifier='xgb', method=None):
        super().__init__()
        a, b = PAIRS[pair]
        self.a = load_backbone(cfg, a, seed, device).backbone
        self.b = load_backbone(cfg, b, seed, device).backbone
        self.fusion, _ = load_fusion(cfg, pair, kind, seed, device)
        self.device = device
        self.classifier = classifier
        self.xgb = None
        if classifier == 'xgb':
            self.xgb, _ = load_xgb(cfg, pair, kind, seed, method or cfg['xgb']['final_method'])
        for p in self.parameters():
            p.requires_grad_(False)

    def embed(self, x, keep=False):
        ta, _ = self.a.tokens(x, keep_map=keep)
        tb, _ = self.b.tokens(x, keep_map=keep)
        return self.fusion.embed(ta, tb, keep_attn=keep)

    def logits_mlp(self, x, keep=False):
        return self.fusion.head(self.embed(x, keep))

    @torch.no_grad()
    def prob_fresh(self, x):
        z = self.embed(x.to(self.device))
        if self.classifier == 'xgb':
            return self.xgb.predict_proba(z.float().cpu().numpy())[:, 1]
        return F.softmax(self.fusion.head(z), 1)[:, 1].cpu().numpy()
