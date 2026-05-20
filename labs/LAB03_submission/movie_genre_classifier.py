"""
movie_genre_classifier.py  —  Part A
Multimodal Movie Genre Classifier

Architecture
============
  Movie Poster  ──►  [ ImageBranch   ]  ──┐
                                           ├──► [ FusionHead ] ──► 6-class genre
  Metadata      ──►  [ TabularBranch ]  ──┘

Image Branch  : 4 convolutional blocks (3→32→64→128→256 channels, each with
                Conv2d → BatchNorm2d → ReLU → MaxPool2d), then a global
                average pool and a linear projection to 256 dims.
                Follows the CIFAR10Net pattern from Lecture 12.
                4 blocks chosen over 3 based on experiment results (+0.4pp val).

Tabular Branch: Two sub-branches merged before the output projection.
  • Numeric  : 7 standardised numeric features → 2-layer MLP → 64 dims
  • Embedding: 4 pipe-separated list fields + 1 single-cat field, each with
               its own nn.Embedding table.  List fields are MAX-pooled across
               tokens (ignoring padding) — confirmed +1.6pp over mean pooling.
               All field vectors are concatenated → 2-layer MLP → 128 dims
  Numeric + Embedding sub-branches are concatenated → Linear → 256 dims.
  Two-branch design confirmed +2.8pp over flat concat (experiment q4_combine).

Fusion Head   : cat(image_256, tabular_256) → Linear(512→256) → ReLU →
                Dropout → Linear(256→6) → logits.
                Single hidden layer confirmed best; deeper heads all hurt.

Hyperparameters (tuned via experiment_runner.py — 28 experiments, 6 epochs each)
=================================================================================
  LR           = 1e-3   (was 3e-4; lr_1e3 +0.3pp)
  WEIGHT_DECAY = 1e-4   (was 1e-3; wd_low  +0.9pp — biggest single gain)
  DROPOUT      = 0.2/0.3 (was 0.3/0.4; model was underfitting, not overfitting)
  BATCH_SIZE   = 64     (batch_32 marginally +0.3pp but 43x slower on MPS)
  BLOCKS       = 4      (img_4blocks +0.4pp over 3 blocks)

Training
========
  • CrossEntropyLoss, Adam with weight decay
  • Validation accuracy tracked each epoch; best model saved as checkpoint
  • Early stopping (patience configurable at top of file)
  • MPS / CUDA / CPU device auto-selected
  • Per-class accuracy printed on the test set at the end

Usage
=====
  python movie_genre_classifier.py
"""

# ─── Standard library ────────────────────────────────────────────────────────
import json
import os
from collections import Counter
from pathlib import Path

# ─── Third-party ─────────────────────────────────────────────────────────────
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
# 0.  PATHS  —  adjust if your working directory differs
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR       = Path("../../data/movie_posters")
IMAGE_DIR      = DATA_DIR / "images"
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
GENRES       = ["Animation", "Comedy", "Documentary", "Horror", "Romance", "Sci-Fi"]
NUM_CLASSES  = len(GENRES)
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}

NUMERIC_COLS     = ["runtime", "vote_average", "vote_count",
                    "release_year", "popularity", "budget", "revenue"]
LIST_FIELDS      = ["cast", "directors", "writers", "production_companies"]
SINGLE_CAT_FIELDS = ["mpaa_rating"]

IMAGE_SIZE   = 128   # poster resize target (pixels)
MAX_LIST_LEN = 20    # pad / truncate list fields to this many tokens
TOP_N_VOCAB  = 50    # keep only top-N tokens per field by training frequency
EMBED_DIM    = 32    # embedding dimension for all categorical fields

# Training hyper-parameters  (tuned via experiment_runner.py)
BATCH_SIZE   = 64     # batch_32 +0.3pp but 43× slower on MPS — not worth it
NUM_EPOCHS   = 40
LR           = 1e-3   # was 3e-4; lr_1e3 experiment: +0.3pp val accuracy
WEIGHT_DECAY = 1e-4   # was 1e-3; wd_low experiment: +0.9pp — biggest single gain
PATIENCE     = 8      # early-stopping patience (epochs)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  PROVIDED INFRASTRUCTURE  (VocabBuilder + NumericScaler)
# ─────────────────────────────────────────────────────────────────────────────

class VocabBuilder:
    """
    Builds integer vocabularies for pipe-separated categorical fields.

    Fit ONLY on training data — fitting on val/test is data leakage.

    Token index conventions:
        0 = <PAD>  — padding (short lists are padded to MAX_LIST_LEN)
        1 = <UNK>  — unknown token (not in top-N at training time)
        2+ = actual tokens, ordered by training frequency
    """

    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, top_n=TOP_N_VOCAB):
        self.top_n  = top_n
        self.vocabs = {}   # field -> {token_string: integer_index}
        self.sizes  = {}   # field -> vocab size (including PAD and UNK)

    def fit(self, df):
        for field in LIST_FIELDS:
            if field not in df.columns:
                continue
            counts = Counter()
            for val in df[field].dropna():
                if val:
                    counts.update(v.strip() for v in str(val).split("|") if v.strip())
            top_tokens = [tok for tok, _ in counts.most_common(self.top_n)]
            vocab = {tok: idx + 2 for idx, tok in enumerate(top_tokens)}
            self.vocabs[field] = vocab
            self.sizes[field]  = len(vocab) + 2  # +2 for PAD and UNK

        for field in SINGLE_CAT_FIELDS:
            if field not in df.columns:
                continue
            unique_vals = [v for v in df[field].unique()
                           if isinstance(v, str) and v.strip()]
            vocab = {v: idx + 2 for idx, v in enumerate(sorted(unique_vals))}
            self.vocabs[field] = vocab
            self.sizes[field]  = len(vocab) + 2
        return self

    def encode_list(self, val, field, max_len=MAX_LIST_LEN):
        vocab = self.vocabs.get(field, {})
        if not isinstance(val, str) or not val.strip():
            return [self.PAD_IDX] * max_len
        tokens = [v.strip() for v in val.split("|") if v.strip()]
        ids = [vocab.get(tok, self.UNK_IDX) for tok in tokens]
        ids = ids[:max_len]
        ids += [self.PAD_IDX] * (max_len - len(ids))
        return ids

    def encode_single(self, val, field):
        vocab = self.vocabs.get(field, {})
        if not isinstance(val, str) or not val.strip():
            return self.PAD_IDX
        return vocab.get(val.strip(), self.UNK_IDX)

    def save(self, path):
        data = {"vocabs": self.vocabs, "sizes": self.sizes, "top_n": self.top_n}
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        vb = cls(top_n=data["top_n"])
        vb.vocabs = data["vocabs"]
        vb.sizes  = data["sizes"]
        return vb


class NumericScaler:
    """
    Standardises numeric features to zero mean, unit variance.
    Fit on training data only. Missing values are imputed with the training mean.
    """

    def __init__(self):
        self.means = {}
        self.stds  = {}

    def fit(self, df):
        for col in NUMERIC_COLS:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                self.means[col] = float(vals.mean())
                self.stds[col]  = max(float(vals.std()), 1e-8)
        return self

    def transform(self, df):
        result = {}
        for col in NUMERIC_COLS:
            vals = pd.to_numeric(df[col], errors="coerce") if col in df.columns \
                   else pd.Series([float("nan")] * len(df))
            vals = vals.fillna(self.means.get(col, 0.0))
            mean = self.means.get(col, 0.0)
            std  = self.stds.get(col, 1.0)
            result[col] = ((vals - mean) / std).values.astype(np.float32)
        return result

    def save(self, path):
        Path(path).write_text(json.dumps({"means": self.means, "stds": self.stds}))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        ns = cls()
        ns.means = data["means"]
        ns.stds  = data["stds"]
        return ns


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATASET
# ─────────────────────────────────────────────────────────────────────────────

class MoviePosterDataset(Dataset):
    """
    Loads movie posters + metadata for the multimodal classifier.

    All categorical encoding and numeric scaling are pre-computed in __init__
    so that __getitem__ only does image I/O and tensor conversion.
    """

    def __init__(self, df, data_dir, vocab_builder, scaler, transform):
        self.df        = df.reset_index(drop=True)
        self.data_dir  = Path(data_dir)
        self.transform = transform

        # ── Labels ──────────────────────────────────────────────────────────
        self.labels = torch.tensor(
            [GENRE_TO_IDX[g] for g in df["label"].values], dtype=torch.long
        )

        # ── Pre-scale numeric features → (N, 7) float32 ─────────────────────
        scaled = scaler.transform(df)
        self.numeric = np.stack(
            [scaled[col] for col in NUMERIC_COLS], axis=1
        ).astype(np.float32)   # (N, len(NUMERIC_COLS))

        # ── Pre-encode list fields → {field: (N, MAX_LIST_LEN) int64} ────────
        self.cat_list = {}
        for field in LIST_FIELDS:
            if field not in df.columns:
                self.cat_list[field] = np.zeros(
                    (len(df), MAX_LIST_LEN), dtype=np.int64
                )
                continue
            rows = []
            for val in df[field].fillna("").values:
                rows.append(vocab_builder.encode_list(str(val), field))
            self.cat_list[field] = np.array(rows, dtype=np.int64)

        # ── Pre-encode single-cat fields → {field: (N,) int64} ──────────────
        self.cat_single = {}
        for field in SINGLE_CAT_FIELDS:
            if field not in df.columns:
                self.cat_single[field] = np.zeros(len(df), dtype=np.int64)
                continue
            ids = [vocab_builder.encode_single(str(v), field)
                   for v in df[field].fillna("").values]
            self.cat_single[field] = np.array(ids, dtype=np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Image ────────────────────────────────────────────────────────────
        img_path = self.data_dir / row["image_path"]
        try:
            image = Image.open(str(img_path)).convert("RGB")
        except Exception:
            # Fallback: grey placeholder if the file is missing
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(128, 128, 128))
        image = self.transform(image)   # (3, IMAGE_SIZE, IMAGE_SIZE)

        # ── Numeric ──────────────────────────────────────────────────────────
        numeric = torch.tensor(self.numeric[idx], dtype=torch.float32)

        # ── Categorical ──────────────────────────────────────────────────────
        cat_ids = {}
        for field in LIST_FIELDS:
            cat_ids[field] = torch.tensor(
                self.cat_list[field][idx], dtype=torch.long
            )
        for field in SINGLE_CAT_FIELDS:
            cat_ids[field] = torch.tensor(
                int(self.cat_single[field][idx]), dtype=torch.long
            )

        label = self.labels[idx]
        return image, numeric, cat_ids, label


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MODEL COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

class ImageBranch(nn.Module):
    """
    Convolutional image encoder.

    Four [Conv → BatchNorm → ReLU → MaxPool] blocks that progressively
    reduce spatial size and grow the number of channels (3→32→64→128→256),
    following the CIFAR10Net pattern from Lecture 12.

    4 blocks chosen over 3: experiment img_4blocks scored 75.1% vs 74.7%
    (+0.4pp). Skip connections tested and dropped — hurt by 1.4pp (img_skip
    experiment), adding noise on a small dataset.

    A global average pool collapses spatial dimensions to 1×1, giving a
    256-dimensional vector regardless of the exact input resolution.
    A dropout + linear projection maps this to `out_dim` dimensions.

    Dropout lowered to 0.3 (from 0.4): all experiments showed val > train,
    meaning the model was underfitting, not overfitting.

    Input : (B, 3, IMAGE_SIZE, IMAGE_SIZE)
    Output: (B, out_dim)
    """

    def __init__(self, out_dim: int = 256, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # ── Block 1: 128×128 → 64×64 ────────────────────────────────────
            nn.Conv2d(3, 32, kernel_size=3, padding=1),    # (B,  32, 128, 128)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                             # (B,  32,  64,  64)

            # ── Block 2: 64×64 → 32×32 ──────────────────────────────────────
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # (B,  64,  64,  64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                             # (B,  64,  32,  32)

            # ── Block 3: 32×32 → 16×16 ──────────────────────────────────────
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (B, 128,  32,  32)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                             # (B, 128,  16,  16)

            # ── Block 4: 16×16 → 8×8  (added — experiment img_4blocks +0.4pp)
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # (B, 256,  16,  16)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                             # (B, 256,   8,   8)
        )

        # Global average pool → (B, 256, 1, 1) — spatial position is irrelevant
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Trainable projection head
        self.head = nn.Sequential(
            nn.Flatten(),                       # (B, 256)
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),            # (B, out_dim)
            nn.ReLU(inplace=True),
        )

    def forward(self, x):                       # x: (B, 3, H, W)
        x = self.features(x)                    # (B, 256, H', W')
        x = self.global_pool(x)                 # (B, 256, 1, 1)
        return self.head(x)                     # (B, out_dim)


class TabularBranch(nn.Module):
    """
    Tabular encoder combining numeric features and learned embeddings.

    Two internal sub-branches (Lecture 11, MultiInputNet pattern):
      • Numeric  : standardised floats → small MLP → 64 dims
      • Embedding: one embedding table per categorical field; list fields are
                   max-pooled across tokens (padding masked to -∞ before max
                   so padding never wins).  All field vectors are concatenated
                   then projected → 128 dims.

    Both sub-branches are concatenated and projected to `out_dim`.

    Input : numeric (B, 7) + cat_ids dict
    Output: (B, out_dim)
    """

    def __init__(self, vocab_sizes: dict, out_dim: int = 256, dropout: float = 0.2):
        super().__init__()

        # ── One embedding table per field ────────────────────────────────────
        # (Lecture 11: "Each categorical field should get its own embedding table")
        self.embeddings = nn.ModuleDict()
        for field in LIST_FIELDS + SINGLE_CAT_FIELDS:
            if field in vocab_sizes:
                self.embeddings[field] = nn.Embedding(
                    vocab_sizes[field], EMBED_DIM, padding_idx=0
                )

        n_embed_fields = len(self.embeddings)
        embed_total    = n_embed_fields * EMBED_DIM   # 5 × 32 = 160

        # ── Numeric sub-branch ───────────────────────────────────────────────
        self.numeric_branch = nn.Sequential(
            nn.Linear(len(NUMERIC_COLS), 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        # ── Embedding sub-branch ─────────────────────────────────────────────
        self.embed_branch = nn.Sequential(
            nn.Linear(embed_total, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Merge both sub-branches ──────────────────────────────────────────
        self.merge = nn.Sequential(
            nn.Linear(64 + 128, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def _max_pool_embeddings(self, ids: torch.Tensor,
                             emb_layer: nn.Embedding) -> torch.Tensor:
        """
        Look up embeddings for a list field and max-pool across tokens,
        ignoring padding positions (index 0).

        ids  : (B, MAX_LIST_LEN)
        return: (B, EMBED_DIM)
        """
        emb = emb_layer(ids)                            # (B, L, EMBED_DIM)

        # Mask padding positions with -inf so they can never win the max
        padding_mask = (ids == 0).unsqueeze(-1)         # (B, L, 1) bool
        emb_masked   = emb.masked_fill(padding_mask, float("-inf"))

        pooled = emb_masked.max(dim=1).values           # (B, EMBED_DIM)

        # Any row where EVERY token was padding → replace -inf with 0
        all_pad = padding_mask.squeeze(-1).all(dim=1, keepdim=True)  # (B, 1)
        pooled  = torch.where(all_pad, torch.zeros_like(pooled), pooled)
        return pooled

    def forward(self, numeric: torch.Tensor, cat_ids: dict) -> torch.Tensor:
        # ── Numeric ──────────────────────────────────────────────────────────
        num_out = self.numeric_branch(numeric)      # (B, 64)

        # ── Embeddings ───────────────────────────────────────────────────────
        embed_vecs = []

        for field in LIST_FIELDS:
            if field in self.embeddings and field in cat_ids:
                pooled = self._max_pool_embeddings(
                    cat_ids[field], self.embeddings[field]
                )                                   # (B, EMBED_DIM)
                embed_vecs.append(pooled)

        for field in SINGLE_CAT_FIELDS:
            if field in self.embeddings and field in cat_ids:
                ids = cat_ids[field]                # (B,)
                emb = self.embeddings[field](ids)   # (B, EMBED_DIM)
                embed_vecs.append(emb)

        embed_cat = torch.cat(embed_vecs, dim=1)    # (B, n_fields × EMBED_DIM)
        embed_out = self.embed_branch(embed_cat)    # (B, 128)

        # ── Merge ────────────────────────────────────────────────────────────
        combined = torch.cat([num_out, embed_out], dim=1)   # (B, 192)
        return self.merge(combined)                          # (B, out_dim)


class MultimodalGenreClassifier(nn.Module):
    """
    Full multimodal model.

    Image branch and tabular branch each produce a 256-dim feature vector.
    These are concatenated and passed through the fusion head to produce
    6-class logits.

    Input : images (B,3,H,W), numeric (B,7), cat_ids dict
    Output: logits (B, NUM_CLASSES)
    """

    def __init__(self, vocab_sizes: dict,
                 img_out: int = 256, tab_out: int = 256,
                 dropout: float = 0.2):
        super().__init__()

        self.image_branch   = ImageBranch(out_dim=img_out, dropout=0.3)
        self.tabular_branch = TabularBranch(vocab_sizes, out_dim=tab_out, dropout=dropout)

        # Fusion head: concatenate → downsample → classify
        self.fusion_head = nn.Sequential(
            nn.Linear(img_out + tab_out, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, images: torch.Tensor,
                numeric: torch.Tensor,
                cat_ids: dict) -> torch.Tensor:
        img_feats = self.image_branch(images)              # (B, img_out)
        tab_feats = self.tabular_branch(numeric, cat_ids)  # (B, tab_out)
        fused     = torch.cat([img_feats, tab_feats], dim=1)
        return self.fusion_head(fused)                     # (B, NUM_CLASSES)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def move_batch_to_device(images, numeric, cat_ids, labels, device):
    """Move all tensors in a batch to the specified device."""
    images  = images.to(device)
    numeric = numeric.to(device)
    labels  = labels.to(device)
    cat_ids = {k: v.to(device) for k, v in cat_ids.items()}
    return images, numeric, cat_ids, labels


def save_checkpoint(model, optimizer, epoch, val_acc, path):
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc":              val_acc,
    }, path)


def load_checkpoint(model, optimizer, path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"] + 1, ckpt["val_acc"]


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN TRAINING SCRIPT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Device: {device}\n")

    # ── Load manifests ───────────────────────────────────────────────────────
    train_df = pd.read_csv(DATA_DIR / "train_manifest.csv")
    val_df   = pd.read_csv(DATA_DIR / "val_manifest.csv")
    test_df  = pd.read_csv(DATA_DIR / "test_manifest.csv")

    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    # ── Fit preprocessors on training data only ──────────────────────────────
    # (Lecture 11: "Fit ONLY on training data — fitting on val/test is leakage")
    vocab_builder = VocabBuilder(top_n=TOP_N_VOCAB).fit(train_df)
    scaler        = NumericScaler().fit(train_df)

    vocab_builder.save(CHECKPOINT_DIR / "vocab_builder.json")
    scaler.save(CHECKPOINT_DIR / "scaler.json")

    # ── Image transforms ─────────────────────────────────────────────────────
    # ImageNet normalisation stats (used by Part B ResNet backbone too —
    # setting this up consistently now makes comparison fair)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

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

    # ── Datasets & DataLoaders ───────────────────────────────────────────────
    train_dataset = MoviePosterDataset(train_df, DATA_DIR, vocab_builder, scaler, train_transform)
    val_dataset   = MoviePosterDataset(val_df,   DATA_DIR, vocab_builder, scaler, eval_transform)
    test_dataset  = MoviePosterDataset(test_df,  DATA_DIR, vocab_builder, scaler, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model = MultimodalGenreClassifier(vocab_sizes=vocab_builder.sizes).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} total  |  {trainable_params:,} trainable\n")

    # ── Optimizer + loss ─────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # ── Training loop with early stopping ────────────────────────────────────
    # Pattern from Lecture 11: track best val metric, save checkpoint, stop early
    best_val_acc      = 0.0
    patience_counter  = 0

    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>10}  {'Val Acc':>10}  {'Note'}")
    print("─" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):

        # ── Training phase ───────────────────────────────────────────────────
        model.train()
        running_loss  = 0.0
        train_correct = 0
        train_total   = 0

        for images, numeric, cat_ids, labels in tqdm(
                train_loader, desc=f"Epoch {epoch:>2}", leave=False):

            images, numeric, cat_ids, labels = move_batch_to_device(
                images, numeric, cat_ids, labels, device
            )

            optimizer.zero_grad()
            logits = model(images, numeric, cat_ids)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss  += loss.item()
            preds          = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        avg_train_loss = running_loss / len(train_loader)
        train_acc      = 100.0 * train_correct / train_total

        # ── Validation phase ─────────────────────────────────────────────────
        model.eval()
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for images, numeric, cat_ids, labels in val_loader:
                images, numeric, cat_ids, labels = move_batch_to_device(
                    images, numeric, cat_ids, labels, device
                )
                logits = model(images, numeric, cat_ids)
                preds  = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = 100.0 * val_correct / val_total

        # ── Early stopping & checkpoint ──────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_acc, BEST_MODEL_PATH)
            note = "✓ saved"
        else:
            patience_counter += 1
            note = f"patience {patience_counter}/{PATIENCE}"

        print(f"{epoch:>5}  {avg_train_loss:>10.4f}  {train_acc:>9.1f}%  {val_acc:>9.1f}%  {note}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} — best val acc: {best_val_acc:.1f}%")
            break

    # ── Test evaluation with per-class accuracy ───────────────────────────────
    print("\n" + "═" * 60)
    print("Loading best checkpoint for test evaluation …")

    model = MultimodalGenreClassifier(vocab_sizes=vocab_builder.sizes).to(device)
    optimizer_tmp = optim.Adam(model.parameters())
    load_checkpoint(model, optimizer_tmp, BEST_MODEL_PATH, device=str(device))
    model.eval()

    class_correct = [0] * NUM_CLASSES
    class_total   = [0] * NUM_CLASSES
    all_preds     = []
    all_labels    = []

    with torch.no_grad():
        for images, numeric, cat_ids, labels in test_loader:
            images, numeric, cat_ids, labels = move_batch_to_device(
                images, numeric, cat_ids, labels, device
            )
            logits = model(images, numeric, cat_ids)
            preds  = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            for p, t in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                class_total[t]   += 1
                if p == t:
                    class_correct[t] += 1

    overall_acc = 100.0 * sum(class_correct) / sum(class_total)

    print("\nPer-class test accuracy:")
    print(f"  {'Genre':<14} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("  " + "─" * 44)
    for i, genre in enumerate(GENRES):
        acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        print(f"  {genre:<14} {class_correct[i]:>8} {class_total[i]:>8} {acc:>9.1f}%")
    print("  " + "─" * 44)
    print(f"  {'Overall':<14} {sum(class_correct):>8} {sum(class_total):>8} {overall_acc:>9.1f}%")
    print("\nBest checkpoint saved to:", BEST_MODEL_PATH)


if __name__ == "__main__":
    main()
