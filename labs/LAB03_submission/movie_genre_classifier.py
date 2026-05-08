"""
Lab 3 — Part A: Multimodal Movie Genre Classifier
==================================================
Complete this file to build and train your multimodal neural network.
How you structure the training script (entry point, argument handling, etc.)
is up to you.
"""

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# =============================================================================
# Constants — adjust these to control model complexity
# =============================================================================

GENRES = ["Animation", "Comedy", "Documentary", "Horror", "Romance", "Sci-Fi"]

NUMERIC_COLS = ["runtime", "vote_average", "vote_count",
                "release_year", "popularity", "budget", "revenue"]

# Pipe-separated list fields — each gets its own embedding vocabulary
LIST_FIELDS = ["cast", "directors", "writers", "production_companies"]

# Single-value categorical fields
SINGLE_CAT_FIELDS = ["mpaa_rating"]

IMAGE_SIZE   = 128   # poster resize target (pixels)
MAX_LIST_LEN = 20    # pad/truncate list fields to this many tokens
TOP_N_VOCAB  = 50    # keep only top-N tokens per field by training frequency
EMBED_DIM    = 32    # embedding dimension for all categorical fields


# =============================================================================
# PROVIDED: VocabBuilder
# =============================================================================

class VocabBuilder:
    """
    Builds integer vocabularies for pipe-separated categorical fields.
    Fit ONLY on training data — fitting on val/test is data leakage.

    Token index conventions:
        0 = <PAD>  — padding
        1 = <UNK>  — unknown token (not in top-N at training time)
        2+ = actual tokens, ordered by training frequency
    """

    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, top_n=TOP_N_VOCAB):
        self.top_n  = top_n
        self.vocabs = {}
        self.sizes  = {}

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
            self.sizes[field]  = len(vocab) + 2

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


# =============================================================================
# PROVIDED: NumericScaler
# =============================================================================

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


# =============================================================================
# YOUR CODE: Dataset
# =============================================================================

class MoviePosterDataset(Dataset):
    """
    Loads a split (train / val / test) and returns one sample per film.

    Each sample should contain:
      - The poster image as a tensor
      - The numeric features as a tensor
      - Encoded list-field tensors (one per field in LIST_FIELDS)
      - The MPAA rating as an integer index
      - The genre label as an integer index
    """

    def __init__(self, df, image_dir, vocab_builder, numeric_scaler,
                 transform=None):
        # YOUR CODE HERE
        # Hint: call numeric_scaler.transform(df) once here and store the result
        # so you're not recomputing it on every __getitem__ call.
        raise NotImplementedError

    def __len__(self):
        # YOUR CODE HERE
        raise NotImplementedError

    def __getitem__(self, idx):
        # YOUR CODE HERE
        # Return a dict or tuple containing image tensor, numeric tensor,
        # categorical tensors, and the integer label.
        raise NotImplementedError


# =============================================================================
# YOUR CODE: Image Branch
# =============================================================================

class ImageBranch(nn.Module):
    """
    Takes a (batch, 3, H, W) poster tensor and produces a feature vector.
    Must use convolutional layers — not just a flattened image into FC.
    """

    def __init__(self, out_dim=256):
        super().__init__()
        # YOUR CODE HERE
        # Suggested structure: stack of Conv2d blocks that reduce spatial size,
        # followed by global average (or max) pooling, then a linear projection.
        raise NotImplementedError

    def forward(self, x):
        # YOUR CODE HERE
        raise NotImplementedError


# =============================================================================
# YOUR CODE: Tabular Branch
# =============================================================================

class TabularBranch(nn.Module):
    """
    Takes numeric features and categorical embeddings and produces a feature vector.

    Consider two sub-branches:
      - Numeric: FC layers over the standardised numeric features
      - Embedding: one nn.Embedding table per field, pool tokens -> concat -> FC

    Then merge the two sub-branches into a single output vector.
    """

    def __init__(self, vocab_sizes, out_dim=256):
        super().__init__()
        # YOUR CODE HERE
        # vocab_sizes is a dict: {field_name: int} from vocab_builder.sizes
        raise NotImplementedError

    def forward(self, numeric, cat_fields):
        # YOUR CODE HERE
        # numeric:    (batch, len(NUMERIC_COLS)) float tensor
        # cat_fields: dict of {field_name: (batch, MAX_LIST_LEN) int tensor}
        #             plus mpaa_rating as a (batch,) int tensor
        raise NotImplementedError


# =============================================================================
# YOUR CODE: Fusion Head
# =============================================================================

class FusionHead(nn.Module):
    """
    Concatenates image and tabular feature vectors and predicts genre.
    Output: (batch, num_classes) logits (no softmax — use CrossEntropyLoss).
    """

    def __init__(self, image_dim, tabular_dim, num_classes=len(GENRES)):
        super().__init__()
        # YOUR CODE HERE
        raise NotImplementedError

    def forward(self, image_features, tabular_features):
        # YOUR CODE HERE
        raise NotImplementedError


# =============================================================================
# YOUR CODE: Full Model
# =============================================================================

class MultimodalGenreClassifier(nn.Module):
    """Wires ImageBranch, TabularBranch, and FusionHead together."""

    def __init__(self, vocab_sizes):
        super().__init__()
        # YOUR CODE HERE
        raise NotImplementedError

    def forward(self, image, numeric, cat_fields):
        # YOUR CODE HERE
        raise NotImplementedError


# =============================================================================
# YOUR CODE: Training
# =============================================================================
# How you structure training is up to you.
# Your script must:
#   - Load the three manifest CSVs
#   - Fit VocabBuilder and NumericScaler on the training set only
#   - Build Datasets and DataLoaders for each split
#   - Train for multiple epochs, reporting validation accuracy each epoch
#   - Save the best model checkpoint
#   - Print per-class accuracy on the test set at the end
#
# Device setup (works locally and on Colab GPU):
#   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
