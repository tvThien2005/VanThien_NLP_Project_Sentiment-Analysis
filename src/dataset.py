"""
dataset.py — PyTorch Dataset + DataLoader factory
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from vocabulary import Vocabulary
from config import (
    DATA_PATH, MAX_SEQ_LEN, BATCH_SIZE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    RANDOM_SEED, NUM_CLASSES, DEVICE, USE_CLASS_WEIGHTS
)


# ── Dataset ───────────────────────────────────────────────────────────────────
class SentimentDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: Vocabulary):
        self.texts  = texts
        self.labels = labels
        self.vocab  = vocab

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        ids = self.vocab.encode(self.texts[idx], MAX_SEQ_LEN)
        return {
            "input_ids":       torch.tensor(ids,              dtype=torch.long),
            "attention_mask":  torch.tensor([1 if x != 0 else 0 for x in ids], dtype=torch.bool),
            "label":           torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Data Loading ──────────────────────────────────────────────────────────────
def load_raw_data() -> tuple[list[str], list[int]]:
    """Đọc CSV, trả về (texts, labels). Lọc các text rỗng sau tokenize."""
    from vocabulary import basic_tokenize
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Comment", "Sentiment"])
    texts  = df["Comment"].astype(str).tolist()
    labels = df["Sentiment"].astype(int).tolist()

    # Lọc text rỗng (sau khi tokenize vẫn là empty)
    filtered = [(t, l) for t, l in zip(texts, labels) if len(basic_tokenize(t)) > 0]
    texts, labels = zip(*filtered)
    print(f"Removed {len(df) - len(texts):,} empty samples after tokenization.")
    return list(texts), list(labels)


def make_dataloaders(
    vocab: Vocabulary | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, Vocabulary, torch.Tensor | None]:
    """
    Pipeline hoàn chỉnh: load → split → build vocab → tạo DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, vocab, class_weights
    """
    texts, labels = load_raw_data()
    print(f"Total samples: {len(texts):,}")

    # ── Train / Val / Test split ──────────────────────────────────────────────
    # Bước 1: tách test ra trước
    test_size  = TEST_RATIO
    val_size   = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)   # relative to remainder

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        texts, labels,
        test_size=test_size,
        stratify=labels,         # giữ nguyên phân phối class
        random_state=RANDOM_SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_size,
        stratify=y_trainval,
        random_state=RANDOM_SEED,
    )

    print(f"  Train : {len(X_train):,}")
    print(f"  Val   : {len(X_val):,}")
    print(f"  Test  : {len(X_test):,}")

    # ── Build vocabulary (chỉ từ train) ──────────────────────────────────────
    if vocab is None:
        vocab = Vocabulary()
        vocab.build(X_train)
        vocab.save()

    # ── Compute class weights ─────────────────────────────────────────────────
    class_weights = None
    if USE_CLASS_WEIGHTS:
        cw = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(NUM_CLASSES),
            y=y_train,
        )
        class_weights = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
        print(f"  Class weights: {cw.round(3)}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = SentimentDataset(X_train, y_train, vocab)
    val_ds   = SentimentDataset(X_val,   y_val,   vocab)
    test_ds  = SentimentDataset(X_test,  y_test,  vocab)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2,
        shuffle=False, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE * 2,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, vocab, class_weights
