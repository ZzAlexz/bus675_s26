"""
experiment_runner.py  —  Architecture & Hyperparameter Search for Lab 3 Part A
===============================================================================
Runs 26 controlled experiments, each varying ONE design axis at a time, so the
results directly answer the seven lab questions:

  Q1  How many conv blocks? How wide? Skip connections?
  Q2  All five list fields, or a subset? What vocabulary size?
  Q3  Mean-pool or max-pool across list-field tokens?
  Q4  How to combine numeric and embedding features?
  Q5  Fusion head size and depth?
  Q6  How much dropout? Where?
  Q7  Learning rate, batch size, LR schedule?

Each experiment trains for EPOCHS_PER_EXP epochs (default 12) with early
stopping.  Results are saved to experiments/results.csv as they complete, so
a crash never loses progress.  A ranked summary table is printed at the end
with per-question recommendations.

Usage
-----
  python experiment_runner.py              # full run (~1-2 hrs on GPU)
  python experiment_runner.py --fast       # 6 epochs each (~30-45 min)
  python experiment_runner.py --resume     # skip already-logged experiments
"""

# ─── stdlib ──────────────────────────────────────────────────────────────────
import argparse
import csv
import json
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 0.  GLOBAL PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR        = Path("../../data/movie_posters")
IMAGE_DIR       = DATA_DIR / "images"
OUTPUT_DIR      = Path("experiments")
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_CSV     = OUTPUT_DIR / "results.csv"

GENRES          = ["Animation", "Comedy", "Documentary", "Horror", "Romance", "Sci-Fi"]
NUM_CLASSES     = len(GENRES)
GENRE_TO_IDX    = {g: i for i, g in enumerate(GENRES)}

NUMERIC_COLS        = ["runtime", "vote_average", "vote_count",
                        "release_year", "popularity", "budget", "revenue"]
ALL_LIST_FIELDS     = ("cast", "directors", "writers", "production_companies")
SINGLE_CAT_FIELDS   = ("mpaa_rating",)

IMAGE_SIZE      = 128
MAX_LIST_LEN    = 20
IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]

EPOCHS_PER_EXP_FULL = 12
EPOCHS_PER_EXP_FAST = 6

# ─────────────────────────────────────────────────────────────────────────────
# 1.  EXPERIMENT CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExpConfig:
    """
    One experiment = one set of answers to the lab's design questions.
    Every field has a sensible baseline default so only the axis being probed
    needs to change.
    """
    name:   str
    group:  str   = "misc"     # which question this experiment answers

    # ── Q1  Image branch ─────────────────────────────────────────────────────
    n_conv_blocks:  int   = 3          # 2 / 3 / 4
    base_channels:  int   = 32         # first block channels; doubles each block
    use_skip:       bool  = False      # projection skip connection on each block
    img_out_dim:    int   = 256        # linear projection after global avg pool
    img_dropout:    float = 0.4

    # ── Q2 & Q3  Tabular branch ───────────────────────────────────────────────
    active_list_fields: Tuple[str, ...] = ALL_LIST_FIELDS
    top_n_vocab:        int   = 50     # vocabulary cap per list field
    pool_mode:          str   = "max"  # "max" or "mean"
    embed_dim:          int   = 32
    tab_out_dim:        int   = 256
    tab_dropout:        float = 0.3

    # ── Q4  Numeric + embedding combine strategy ──────────────────────────────
    # "two_branch" : separate MLPs → concat → projection  (baseline)
    # "flat"       : raw concat of numeric + pooled embeds → one MLP
    combine_mode: str = "two_branch"

    # ── Q5  Fusion head ───────────────────────────────────────────────────────
    fusion_hidden:  Tuple[int, ...] = (256,)   # hidden layer widths
    fusion_dropout: float = 0.3

    # ── Q6  Regularisation ────────────────────────────────────────────────────
    weight_decay: float = 1e-3

    # ── Q7  Optimiser / schedule ─────────────────────────────────────────────
    lr:         float = 3e-4
    batch_size: int   = 64
    scheduler:  str   = "none"   # "none" / "cosine" / "step"

    # ── Experiment control ────────────────────────────────────────────────────
    num_epochs: int   = EPOCHS_PER_EXP_FULL
    patience:   int   = 4


# ─────────────────────────────────────────────────────────────────────────────
# 2.  EXPERIMENT SUITE  — one entry per design axis being probed
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENTS: List[ExpConfig] = [

    # ── Baseline (reference for all comparisons) ──────────────────────────────
    ExpConfig(name="baseline",        group="reference"),

    # ── Q1a: Number of conv blocks ────────────────────────────────────────────
    ExpConfig(name="img_2blocks",     group="q1_blocks",   n_conv_blocks=2),
    ExpConfig(name="img_4blocks",     group="q1_blocks",   n_conv_blocks=4),

    # ── Q1b: Channel width (base_channels; doubles each block) ────────────────
    ExpConfig(name="img_narrow",      group="q1_width",    base_channels=16),
    ExpConfig(name="img_wide",        group="q1_width",    base_channels=64),

    # ── Q1c: Skip connections ─────────────────────────────────────────────────
    ExpConfig(name="img_skip",        group="q1_skip",     use_skip=True),

    # ── Q2a: Vocabulary size ──────────────────────────────────────────────────
    ExpConfig(name="vocab_25",        group="q2_vocab",    top_n_vocab=25),
    ExpConfig(name="vocab_100",       group="q2_vocab",    top_n_vocab=100),
    ExpConfig(name="vocab_200",       group="q2_vocab",    top_n_vocab=200),

    # ── Q2b: Field subset ─────────────────────────────────────────────────────
    # All 4 list fields is the baseline; probe subsets
    ExpConfig(name="fields_3",        group="q2_fields",
              active_list_fields=("cast", "directors", "production_companies")),
    ExpConfig(name="fields_2",        group="q2_fields",
              active_list_fields=("cast", "directors")),
    ExpConfig(name="fields_cast_only",group="q2_fields",
              active_list_fields=("cast",)),

    # ── Q3: Pooling mode across list-field tokens ─────────────────────────────
    ExpConfig(name="pool_mean",       group="q3_pooling",  pool_mode="mean"),

    # ── Q4: Combine numeric + embedding inside tabular branch ─────────────────
    ExpConfig(name="combine_flat",    group="q4_combine",  combine_mode="flat"),

    # ── Q5a: Fusion head depth ────────────────────────────────────────────────
    ExpConfig(name="fusion_tiny",     group="q5_fusion",   fusion_hidden=(128,)),
    ExpConfig(name="fusion_wide",     group="q5_fusion",   fusion_hidden=(512,)),
    ExpConfig(name="fusion_deep",     group="q5_fusion",   fusion_hidden=(512, 256)),
    ExpConfig(name="fusion_deeper",   group="q5_fusion",   fusion_hidden=(512, 256, 128)),

    # ── Q6: Dropout ───────────────────────────────────────────────────────────
    ExpConfig(name="drop_low",        group="q6_dropout",
              img_dropout=0.2, tab_dropout=0.15, fusion_dropout=0.15),
    ExpConfig(name="drop_high",       group="q6_dropout",
              img_dropout=0.5, tab_dropout=0.4,  fusion_dropout=0.4),

    # ── Q6: Weight decay ──────────────────────────────────────────────────────
    ExpConfig(name="wd_low",          group="q6_wd",       weight_decay=1e-4),
    ExpConfig(name="wd_high",         group="q6_wd",       weight_decay=1e-2),

    # ── Q7: Learning rate ─────────────────────────────────────────────────────
    ExpConfig(name="lr_1e3",          group="q7_lr",       lr=1e-3),
    ExpConfig(name="lr_1e4",          group="q7_lr",       lr=1e-4),

    # ── Q7: LR schedule ───────────────────────────────────────────────────────
    ExpConfig(name="sched_cosine",    group="q7_schedule", scheduler="cosine"),
    ExpConfig(name="sched_step",      group="q7_schedule", scheduler="step"),

    # ── Q7: Batch size ────────────────────────────────────────────────────────
    ExpConfig(name="batch_32",        group="q7_batch",    batch_size=32),
    ExpConfig(name="batch_128",       group="q7_batch",    batch_size=128),
]

# ─────────────────────────────────────────────────────────────────────────────
# 3.  PROVIDED INFRASTRUCTURE  (VocabBuilder + NumericScaler)
# ─────────────────────────────────────────────────────────────────────────────

class VocabBuilder:
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, top_n: int = 50):
        self.top_n  = top_n
        self.vocabs: Dict[str, Dict[str, int]] = {}
        self.sizes:  Dict[str, int]            = {}

    def fit(self, df: pd.DataFrame) -> "VocabBuilder":
        for field in ALL_LIST_FIELDS:
            if field not in df.columns:
                continue
            counts = Counter()
            for val in df[field].dropna():
                if val:
                    counts.update(v.strip() for v in str(val).split("|") if v.strip())
            top = [tok for tok, _ in counts.most_common(self.top_n)]
            vocab = {tok: idx + 2 for idx, tok in enumerate(top)}
            self.vocabs[field] = vocab
            self.sizes[field]  = len(vocab) + 2

        for field in SINGLE_CAT_FIELDS:
            if field not in df.columns:
                continue
            unique = [v for v in df[field].unique() if isinstance(v, str) and v.strip()]
            vocab  = {v: idx + 2 for idx, v in enumerate(sorted(unique))}
            self.vocabs[field] = vocab
            self.sizes[field]  = len(vocab) + 2
        return self

    def encode_list(self, val: str, field: str, max_len: int = MAX_LIST_LEN) -> List[int]:
        vocab = self.vocabs.get(field, {})
        if not isinstance(val, str) or not val.strip():
            return [self.PAD_IDX] * max_len
        tokens = [v.strip() for v in val.split("|") if v.strip()]
        ids    = [vocab.get(t, self.UNK_IDX) for t in tokens]
        ids    = ids[:max_len] + [self.PAD_IDX] * (max_len - len(ids))
        return ids

    def encode_single(self, val: str, field: str) -> int:
        vocab = self.vocabs.get(field, {})
        if not isinstance(val, str) or not val.strip():
            return self.PAD_IDX
        return vocab.get(val.strip(), self.UNK_IDX)


class NumericScaler:
    def __init__(self):
        self.means: Dict[str, float] = {}
        self.stds:  Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "NumericScaler":
        for col in NUMERIC_COLS:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                self.means[col] = float(vals.mean())
                self.stds[col]  = max(float(vals.std()), 1e-8)
        return self

    def transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        out = {}
        for col in NUMERIC_COLS:
            vals = pd.to_numeric(df[col], errors="coerce") if col in df.columns \
                   else pd.Series([float("nan")] * len(df))
            vals = vals.fillna(self.means.get(col, 0.0))
            out[col] = ((vals - self.means.get(col, 0.0)) /
                        self.stds.get(col, 1.0)).values.astype(np.float32)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DATASET
# ─────────────────────────────────────────────────────────────────────────────

class MoviePosterDataset(Dataset):
    def __init__(self, df, data_dir, vocab_builder, scaler, transform):
        self.df        = df.reset_index(drop=True)
        self.data_dir  = Path(data_dir)
        self.transform = transform

        self.labels = torch.tensor(
            [GENRE_TO_IDX[g] for g in df["label"].values], dtype=torch.long
        )

        scaled = scaler.transform(df)
        self.numeric = np.stack(
            [scaled[col] for col in NUMERIC_COLS], axis=1
        ).astype(np.float32)

        self.cat_list = {}
        for f in ALL_LIST_FIELDS:
            col = df[f].fillna("").values if f in df.columns else [""] * len(df)
            self.cat_list[f] = np.array(
                [vocab_builder.encode_list(str(v), f) for v in col], dtype=np.int64
            )

        self.cat_single = {}
        for f in SINGLE_CAT_FIELDS:
            col = df[f].fillna("").values if f in df.columns else [""] * len(df)
            self.cat_single[f] = np.array(
                [vocab_builder.encode_single(str(v), f) for v in col], dtype=np.int64
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self.data_dir / row["image_path"]
        try:
            image = Image.open(str(img_path)).convert("RGB")
        except Exception:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(128, 128, 128))
        image = self.transform(image)

        numeric = torch.tensor(self.numeric[idx], dtype=torch.float32)

        cat_ids: Dict[str, torch.Tensor] = {}
        for f in ALL_LIST_FIELDS:
            cat_ids[f] = torch.tensor(self.cat_list[f][idx],      dtype=torch.long)
        for f in SINGLE_CAT_FIELDS:
            cat_ids[f] = torch.tensor(int(self.cat_single[f][idx]), dtype=torch.long)

        return image, numeric, cat_ids, self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FLEXIBLE MODEL COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

class FlexConvBlock(nn.Module):
    """
    [Conv → BN → Pool] block with an optional projection skip connection.

    The skip path uses a 1×1 conv + AvgPool to match (out_ch, H/2, W/2),
    following the projection-skip pattern from Lecture 11.  ReLU is applied
    after the optional addition so gradients always flow through the shortcut.
    """

    def __init__(self, in_ch: int, out_ch: int, use_skip: bool = False):
        super().__init__()
        self.conv  = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn    = nn.BatchNorm2d(out_ch)
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU(inplace=True)

        self.use_skip = use_skip
        if use_skip:
            # Project to (out_ch, H/2, W/2) so skip can be added to main path
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.AvgPool2d(2, 2),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pool(self.bn(self.conv(x)))   # no ReLU yet
        if self.use_skip:
            out = out + self.skip(x)
        return self.relu(out)


class FlexImageBranch(nn.Module):
    """
    Configurable convolutional encoder.

    Channels:  3 → base → base×2 → … (doubles each block)
    Spatial:   IMAGE_SIZE → /2 → /4 → … (halved by MaxPool each block)
    After blocks: AdaptiveAvgPool2d(1,1) → Flatten → Dropout → Linear → ReLU
    """

    def __init__(self, n_blocks: int, base_channels: int, use_skip: bool,
                 out_dim: int, dropout: float):
        super().__init__()
        blocks, in_ch, out_ch = [], 3, base_channels
        for _ in range(n_blocks):
            blocks.append(FlexConvBlock(in_ch, out_ch, use_skip=use_skip))
            in_ch, out_ch = out_ch, out_ch * 2
        self.blocks      = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        final_ch = base_channels * (2 ** (n_blocks - 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(final_ch, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.global_pool(self.blocks(x)))


class FlexTabularBranch(nn.Module):
    """
    Tabular encoder supporting:
      - Configurable subset of list fields
      - Max or mean pooling across tokens
      - "two_branch": separate numeric MLP + embedding MLP → concat → project
      - "flat": raw numeric + pooled embeddings concatenated → one MLP
    """

    def __init__(self, vocab_sizes: Dict[str, int],
                 active_list_fields: Tuple[str, ...],
                 pool_mode: str, combine_mode: str,
                 embed_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.pool_mode    = pool_mode
        self.combine_mode = combine_mode

        # Build embedding tables in a fixed, deterministic order
        all_cat = list(active_list_fields) + list(SINGLE_CAT_FIELDS)
        self.embeddings         = nn.ModuleDict()
        self.embed_field_order  = []
        self.list_field_set     = set(active_list_fields)
        for f in all_cat:
            if f in vocab_sizes:
                self.embeddings[f] = nn.Embedding(
                    vocab_sizes[f], embed_dim, padding_idx=0
                )
                self.embed_field_order.append(f)

        n_emb_fields = len(self.embed_field_order)
        embed_total  = n_emb_fields * embed_dim
        n_numeric    = len(NUMERIC_COLS)

        if combine_mode == "two_branch":
            # Numeric sub-branch
            self.numeric_branch = nn.Sequential(
                nn.Linear(n_numeric, 64), nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, 64),        nn.ReLU(inplace=True),
            )
            # Embedding sub-branch
            self.embed_branch = nn.Sequential(
                nn.Linear(embed_total, 128), nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            merge_in = 64 + 128
        else:  # "flat" — raw concat straight to one MLP
            merge_in = n_numeric + embed_total

        self.merge = nn.Sequential(
            nn.Linear(merge_in, out_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    # ── pooling helpers ───────────────────────────────────────────────────────
    def _max_pool(self, ids: torch.Tensor, emb: nn.Embedding) -> torch.Tensor:
        out = emb(ids).masked_fill((ids == 0).unsqueeze(-1), float("-inf"))
        pooled = out.max(dim=1).values
        all_pad = (ids == 0).all(dim=1, keepdim=True)
        return torch.where(all_pad, torch.zeros_like(pooled), pooled)

    def _mean_pool(self, ids: torch.Tensor, emb: nn.Embedding) -> torch.Tensor:
        out  = emb(ids)
        mask = (ids != 0).float().unsqueeze(-1)
        n    = mask.sum(dim=1).clamp(min=1)
        return (out * mask).sum(dim=1) / n

    def _pool(self, ids: torch.Tensor, emb: nn.Embedding) -> torch.Tensor:
        return self._max_pool(ids, emb) if self.pool_mode == "max" \
               else self._mean_pool(ids, emb)

    def forward(self, numeric: torch.Tensor,
                cat_ids: Dict[str, torch.Tensor]) -> torch.Tensor:

        # Collect embedding vectors in the fixed order established in __init__
        embed_vecs = []
        for f in self.embed_field_order:
            if f not in cat_ids:
                continue
            emb_layer = self.embeddings[f]
            ids = cat_ids[f]
            if f in self.list_field_set:
                embed_vecs.append(self._pool(ids, emb_layer))      # list field
            else:
                embed_vecs.append(emb_layer(ids))                   # single-cat

        embed_cat = torch.cat(embed_vecs, dim=1) if embed_vecs \
                    else torch.zeros(numeric.size(0), 0, device=numeric.device)

        if self.combine_mode == "two_branch":
            num_out   = self.numeric_branch(numeric)
            embed_out = self.embed_branch(embed_cat)
            combined  = torch.cat([num_out, embed_out], dim=1)
        else:
            combined = torch.cat([numeric, embed_cat], dim=1)

        return self.merge(combined)


def _build_fusion_head(in_dim: int, hidden_dims: Tuple[int, ...],
                       n_classes: int, dropout: float) -> nn.Sequential:
    """Build a variable-depth classification head from a tuple of hidden widths."""
    layers, prev = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    return nn.Sequential(*layers)


class FlexMultimodalClassifier(nn.Module):
    """Full multimodal model built from an ExpConfig."""

    def __init__(self, vocab_sizes: Dict[str, int], cfg: ExpConfig):
        super().__init__()
        self.image_branch   = FlexImageBranch(
            n_blocks=cfg.n_conv_blocks, base_channels=cfg.base_channels,
            use_skip=cfg.use_skip, out_dim=cfg.img_out_dim, dropout=cfg.img_dropout,
        )
        self.tabular_branch = FlexTabularBranch(
            vocab_sizes=vocab_sizes,
            active_list_fields=cfg.active_list_fields,
            pool_mode=cfg.pool_mode, combine_mode=cfg.combine_mode,
            embed_dim=cfg.embed_dim, out_dim=cfg.tab_out_dim, dropout=cfg.tab_dropout,
        )
        fusion_in = cfg.img_out_dim + cfg.tab_out_dim
        self.fusion_head = _build_fusion_head(
            fusion_in, cfg.fusion_hidden, NUM_CLASSES, cfg.fusion_dropout
        )

    def forward(self, images, numeric, cat_ids):
        img = self.image_branch(images)
        tab = self.tabular_branch(numeric, cat_ids)
        return self.fusion_head(torch.cat([img, tab], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def _to_device(images, numeric, cat_ids, labels, device):
    return (images.to(device),
            numeric.to(device),
            {k: v.to(device) for k, v in cat_ids.items()},
            labels.to(device))


def _make_scheduler(optimizer, cfg: ExpConfig, steps_per_epoch: int):
    if cfg.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.num_epochs
        )
    if cfg.scheduler == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, cfg.num_epochs // 3), gamma=0.5
        )
    return None


def run_experiment(cfg: ExpConfig,
                   train_loader: DataLoader,
                   val_loader:   DataLoader,
                   vocab_sizes:  Dict[str, int],
                   device:       torch.device) -> dict:
    """
    Train one model configuration.
    Returns a result dict with accuracy / loss / param counts.
    """
    model = FlexMultimodalClassifier(vocab_sizes, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = _make_scheduler(optimizer, cfg, len(train_loader))

    best_val_acc    = 0.0
    best_train_acc  = 0.0
    best_epoch      = 0
    patience_ctr    = 0
    train_history   = []
    val_history     = []

    for epoch in range(1, cfg.num_epochs + 1):

        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, num, cat, lbl in train_loader:
            imgs, num, cat, lbl = _to_device(imgs, num, cat, lbl, device)
            optimizer.zero_grad()
            logits = model(imgs, num, cat)
            loss   = criterion(logits, lbl)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += (logits.argmax(1) == lbl).sum().item()
            total   += lbl.size(0)
        train_acc  = 100.0 * correct / total
        train_loss = running_loss / len(train_loader)
        train_history.append(train_acc)

        if scheduler:
            scheduler.step()

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, num, cat, lbl in val_loader:
                imgs, num, cat, lbl = _to_device(imgs, num, cat, lbl, device)
                correct += (model(imgs, num, cat).argmax(1) == lbl).sum().item()
                total   += lbl.size(0)
        val_acc = 100.0 * correct / total
        val_history.append(val_acc)

        # ── early stopping ────────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_train_acc = train_acc
            best_epoch     = epoch
            patience_ctr   = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                break

    return {
        "name":            cfg.name,
        "group":           cfg.group,
        "best_val_acc":    round(best_val_acc,  2),
        "best_train_acc":  round(best_train_acc, 2),
        "best_epoch":      best_epoch,
        "epochs_run":      epoch,
        "n_params":        n_params,
        # config echoes (for CSV readability)
        "n_conv_blocks":   cfg.n_conv_blocks,
        "base_channels":   cfg.base_channels,
        "use_skip":        cfg.use_skip,
        "active_fields":   "+".join(cfg.active_list_fields),
        "top_n_vocab":     cfg.top_n_vocab,
        "pool_mode":       cfg.pool_mode,
        "combine_mode":    cfg.combine_mode,
        "fusion_hidden":   str(cfg.fusion_hidden),
        "img_dropout":     cfg.img_dropout,
        "tab_dropout":     cfg.tab_dropout,
        "fusion_dropout":  cfg.fusion_dropout,
        "weight_decay":    cfg.weight_decay,
        "lr":              cfg.lr,
        "batch_size":      cfg.batch_size,
        "scheduler":       cfg.scheduler,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  RESULTS ANALYSIS  —  answers to the seven lab questions
# ─────────────────────────────────────────────────────────────────────────────

def _best_in_group(results: List[dict], group: str) -> dict:
    """Return the result with highest val accuracy in a given group."""
    group_results = [r for r in results if r["group"] == group]
    return max(group_results, key=lambda r: r["best_val_acc"]) if group_results else {}


def _fmt(r: dict) -> str:
    if not r:
        return "n/a"
    return f"{r['name']} ({r['best_val_acc']:.1f}%)"


def print_summary(results: List[dict]) -> None:
    """
    Print a ranked table and per-question recommendations derived from the data.
    """
    if not results:
        print("No results to summarise.")
        return

    ranked = sorted(results, key=lambda r: r["best_val_acc"], reverse=True)
    baseline = next((r for r in results if r["name"] == "baseline"), None)
    base_acc = baseline["best_val_acc"] if baseline else 0.0

    sep = "═" * 90

    # ── Ranked table ─────────────────────────────────────────────────────────
    print("\n" + sep)
    print("EXPERIMENT RESULTS  —  ranked by best validation accuracy")
    print(sep)
    header = f"{'#':>3}  {'Name':<22} {'Group':<18} {'Best Val':>9} {'Train':>8} {'Gap':>7} {'Params':>8} {'Epoch':>6}"
    print(header)
    print("─" * 90)
    for rank, r in enumerate(ranked, 1):
        gap = r["best_train_acc"] - r["best_val_acc"]
        delta = r["best_val_acc"] - base_acc
        marker = " ▲" if delta > 0.5 else (" ▼" if delta < -0.5 else "")
        print(f"{rank:>3}  {r['name']:<22} {r['group']:<18} "
              f"{r['best_val_acc']:>8.1f}%{marker:<2} "
              f"{r['best_train_acc']:>7.1f}% "
              f"{gap:>6.1f}% "
              f"{r['n_params']:>7,} "
              f"{r['best_epoch']:>6}")
    print("─" * 90)
    if baseline:
        print(f"  Baseline: {base_acc:.1f}%  (reference for ▲▼ deltas > 0.5 pp)")

    # ── Per-question answers ──────────────────────────────────────────────────
    print("\n" + sep)
    print("ANSWERS TO THE SEVEN DESIGN QUESTIONS  (from experimental data)")
    print(sep)

    def group_table(groups, label):
        """Print a mini comparison table for a set of groups + the baseline."""
        relevant = [r for r in results if r["group"] in groups or r["name"] == "baseline"]
        relevant = sorted(relevant, key=lambda r: r["best_val_acc"], reverse=True)
        print(f"\n  {label}")
        print(f"  {'Name':<22} {'Val Acc':>9} {'Gap (overfit)':>14} {'Params':>10}")
        print("  " + "─" * 60)
        for r in relevant:
            gap = r["best_train_acc"] - r["best_val_acc"]
            star = " ←best" if r == relevant[0] else ""
            print(f"  {r['name']:<22} {r['best_val_acc']:>8.1f}%  "
                  f"{gap:>12.1f}%  {r['n_params']:>9,}{star}")

    group_table(["q1_blocks"],  "Q1a  How many conv blocks?")
    group_table(["q1_width"],   "Q1b  How wide? (base_channels)")
    group_table(["q1_skip"],    "Q1c  Skip connections ON vs OFF?")
    group_table(["q2_vocab"],   "Q2a  Vocabulary size (top_n_vocab)")
    group_table(["q2_fields"],  "Q2b  Field subset vs all 4 list fields")
    group_table(["q3_pooling"], "Q3   Mean-pool vs max-pool")
    group_table(["q4_combine"], "Q4   Combine strategy (two_branch vs flat)")
    group_table(["q5_fusion"],  "Q5   Fusion head width / depth")
    group_table(["q6_dropout"], "Q6a  Dropout level")
    group_table(["q6_wd"],      "Q6b  Weight decay")
    group_table(["q7_lr"],      "Q7a  Learning rate")
    group_table(["q7_schedule"],"Q7b  LR schedule")
    group_table(["q7_batch"],   "Q7c  Batch size")

    # ── Synthesised recommendation ────────────────────────────────────────────
    print("\n" + sep)
    print("RECOMMENDED CONFIGURATION FOR movie_genre_classifier.py")
    print(sep)

    picks = {}
    for grp, axis, key in [
        ("q1_blocks",   "conv blocks",    "n_conv_blocks"),
        ("q1_width",    "base_channels",  "base_channels"),
        ("q1_skip",     "use_skip",       "use_skip"),
        ("q2_vocab",    "vocab_size",     "top_n_vocab"),
        ("q2_fields",   "active_fields",  "active_fields"),
        ("q3_pooling",  "pool_mode",      "pool_mode"),
        ("q4_combine",  "combine_mode",   "combine_mode"),
        ("q5_fusion",   "fusion_hidden",  "fusion_hidden"),
        ("q6_dropout",  "img_dropout",    "img_dropout"),
        ("q6_wd",       "weight_decay",   "weight_decay"),
        ("q7_lr",       "lr",             "lr"),
        ("q7_schedule", "scheduler",      "scheduler"),
        ("q7_batch",    "batch_size",     "batch_size"),
    ]:
        best = _best_in_group(results, grp)
        # Only recommend a change from baseline if it actually beat the baseline
        if best and baseline and best["best_val_acc"] > base_acc + 0.3:
            picks[axis] = best.get(key, "?")
            tag = f"▲ +{best['best_val_acc'] - base_acc:.1f} pp over baseline"
        else:
            val = baseline.get(key, "baseline") if baseline else "?"
            picks[axis] = val
            tag = "= keep baseline"
        print(f"  {axis:<20} {str(picks[axis]):<25}  {tag}")

    print(f"\n  → Overfitting (train−val gap) was the main risk for "
          f"{sum(1 for r in results if r['best_train_acc'] - r['best_val_acc'] > 15)}"
          f"/{len(results)} configs.")
    print(f"  → Best single experiment: {ranked[0]['name']}  "
          f"({ranked[0]['best_val_acc']:.1f}% val)")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  RESULTS CSV  (incremental — safe to interrupt)
# ─────────────────────────────────────────────────────────────────────────────

RESULT_FIELDS = [
    "name", "group", "best_val_acc", "best_train_acc", "best_epoch",
    "epochs_run", "n_params",
    "n_conv_blocks", "base_channels", "use_skip", "active_fields",
    "top_n_vocab", "pool_mode", "combine_mode", "fusion_hidden",
    "img_dropout", "tab_dropout", "fusion_dropout", "weight_decay",
    "lr", "batch_size", "scheduler",
]


def _load_existing_results() -> List[dict]:
    if not RESULTS_CSV.exists():
        return []
    with open(RESULTS_CSV) as f:
        return list(csv.DictReader(f))


def _append_result(row: dict) -> None:
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  DATASET CACHE  (avoid re-encoding when vocab size is shared)
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(top_n: int,
                   train_df: pd.DataFrame,
                   val_df:   pd.DataFrame) -> Tuple[Dataset, Dataset, Dict[str, int]]:
    """Build (train_ds, val_ds, vocab_sizes) for a given vocabulary cap."""
    vb = VocabBuilder(top_n=top_n).fit(train_df)
    sc = NumericScaler().fit(train_df)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_ds = MoviePosterDataset(train_df, DATA_DIR, vb, sc, train_transform)
    val_ds   = MoviePosterDataset(val_df,   DATA_DIR, vb, sc, eval_transform)
    return train_ds, val_ds, vb.sizes


# ─────────────────────────────────────────────────────────────────────────────
# 10.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Lab 3 experiment runner")
    parser.add_argument("--fast",   action="store_true",
                        help=f"Use {EPOCHS_PER_EXP_FAST} epochs per experiment instead of {EPOCHS_PER_EXP_FULL}")
    parser.add_argument("--resume", action="store_true",
                        help="Skip experiments already saved to results.csv")
    args = parser.parse_args()

    epochs = EPOCHS_PER_EXP_FAST if args.fast else EPOCHS_PER_EXP_FULL

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Device : {device}")
    print(f"Epochs : {epochs} per experiment  (patience=4)")
    print(f"Output : {RESULTS_CSV}\n")

    # ── Load manifests once ───────────────────────────────────────────────────
    print("Loading manifests …")
    train_df = pd.read_csv(DATA_DIR / "train_manifest.csv")
    val_df   = pd.read_csv(DATA_DIR / "val_manifest.csv")
    print(f"Train {len(train_df):,}  |  Val {len(val_df):,}\n")

    # ── Resume: find already-completed experiment names ───────────────────────
    done_names: set = set()
    if args.resume:
        done_names = {r["name"] for r in _load_existing_results()}
        if done_names:
            print(f"Resuming — skipping {len(done_names)} already-logged experiments.\n")

    # ── Dataset cache keyed by vocab size ─────────────────────────────────────
    ds_cache: Dict[int, Tuple[Dataset, Dataset, Dict[str, int]]] = {}

    def get_datasets(top_n: int):
        if top_n not in ds_cache:
            print(f"  [cache miss] building datasets for top_n_vocab={top_n} …")
            ds_cache[top_n] = build_datasets(top_n, train_df, val_df)
        return ds_cache[top_n]

    # ── Run experiments ───────────────────────────────────────────────────────
    results: List[dict] = []

    for i, cfg in enumerate(EXPERIMENTS, 1):
        cfg.num_epochs = epochs   # honour --fast flag

        if cfg.name in done_names:
            print(f"[{i:>2}/{len(EXPERIMENTS)}] {cfg.name:<22} — SKIPPED (already done)")
            continue

        print(f"[{i:>2}/{len(EXPERIMENTS)}] {cfg.name:<22}  "
              f"({cfg.group})  …")

        try:
            train_ds, val_ds, vocab_sizes = get_datasets(cfg.top_n_vocab)

            train_loader = DataLoader(
                train_ds, batch_size=cfg.batch_size, shuffle=True,
                num_workers=2, pin_memory=(device.type == "cuda"),
            )
            val_loader = DataLoader(
                val_ds, batch_size=cfg.batch_size, shuffle=False,
                num_workers=2, pin_memory=(device.type == "cuda"),
            )

            t0     = time.time()
            result = run_experiment(cfg, train_loader, val_loader,
                                    vocab_sizes, device)
            elapsed = time.time() - t0

            result["time_s"] = round(elapsed, 1)
            results.append(result)
            _append_result(result)

            print(f"         val={result['best_val_acc']:.1f}%  "
                  f"train={result['best_train_acc']:.1f}%  "
                  f"params={result['n_params']:,}  "
                  f"epoch={result['best_epoch']}/{result['epochs_run']}  "
                  f"({elapsed:.0f}s)")

        except Exception:
            print(f"         !! FAILED !!")
            traceback.print_exc()

    # ── If resuming, merge previous results so summary is complete ────────────
    if args.resume:
        prev = _load_existing_results()
        done_new = {r["name"] for r in results}
        for r in prev:
            if r["name"] not in done_new:
                # Cast numeric fields back to float/int
                for k in ("best_val_acc", "best_train_acc", "lr",
                           "img_dropout", "tab_dropout", "fusion_dropout", "weight_decay"):
                    if k in r:
                        try:
                            r[k] = float(r[k])
                        except (ValueError, TypeError):
                            pass
                for k in ("best_epoch", "epochs_run", "n_params",
                           "n_conv_blocks", "base_channels", "top_n_vocab",
                           "batch_size"):
                    if k in r:
                        try:
                            r[k] = int(r[k])
                        except (ValueError, TypeError):
                            pass
                results.append(r)

    print_summary(results)


if __name__ == "__main__":
    main()
