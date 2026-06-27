"""
main.py — Entry point: chạy toàn bộ pipeline

Usage:
    python main.py              # train + evaluate
    python main.py --eval-only  # chỉ evaluate từ checkpoint đã có
"""
import argparse
import os
import sys
import json
import torch

# Thêm thư mục hiện tại vào path
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DEVICE, SAVE_DIR, EMBED_DIM, NUM_HEADS, NUM_LAYERS,
    FFN_DIM, DROPOUT, MAX_POSITION, NUM_CLASSES, RANDOM_SEED
)
from dataset   import make_dataloaders
from model     import SentimentTransformer
from trainer   import Trainer
from evaluator import evaluate, plot_training_curves, plot_confusion_matrix
from vocabulary import Vocabulary

import random
import numpy as np


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def build_model(vocab_size: int) -> SentimentTransformer:
    model = SentimentTransformer(
        vocab_size  = vocab_size,
        num_classes = NUM_CLASSES,
        embed_dim   = EMBED_DIM,
        num_heads   = NUM_HEADS,
        num_layers  = NUM_LAYERS,
        ffn_dim     = FFN_DIM,
        dropout     = DROPOUT,
        max_position= MAX_POSITION,
        pad_idx     = 0,
    )
    return model.to(DEVICE)


def print_banner(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def run_pipeline(eval_only: bool = False) -> None:
    set_seed()

    # ── Step 1: Data ──────────────────────────────────────────────────────────
    print_banner("Step 1/4 — Loading Data & Building Vocabulary")

    # Nếu eval_only và vocab đã có: load vocab từ file
    vocab = None
    if eval_only:
        vocab = Vocabulary()
        vocab.load()  # load từ SAVE_DIR/vocab.json

    train_loader, val_loader, test_loader, vocab, class_weights = make_dataloaders(vocab)

    # ── Step 2: Model ─────────────────────────────────────────────────────────
    print_banner("Step 2/4 — Building Transformer Model")
    model = build_model(vocab_size=len(vocab))

    print(f"\nModel Architecture:")
    print(f"  Vocab size   : {len(vocab):,}")
    print(f"  Embed dim    : {EMBED_DIM}")
    print(f"  Heads        : {NUM_HEADS}")
    print(f"  Layers       : {NUM_LAYERS}")
    print(f"  FFN dim      : {FFN_DIM}")
    print(f"  Total params : {model.count_parameters():,}")
    print(f"  Device       : {DEVICE}")

    # ── Step 3: Training ──────────────────────────────────────────────────────
    history = None

    if not eval_only:
        print_banner("Step 3/4 — Training")
        trainer = Trainer(model, train_loader, val_loader, class_weights)
        history = trainer.train()
        trainer.load_best()   # reload best checkpoint
    else:
        print_banner("Step 3/4 — Skipping Training (eval-only mode)")
        ckpt_path = os.path.join(SAVE_DIR, "best_model.pt")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

        # Load history nếu có
        hist_path = os.path.join(SAVE_DIR, "..", "logs", "history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)

    # ── Step 4: Evaluation ────────────────────────────────────────────────────
    print_banner("Step 4/4 — Evaluation & Visualization")

    # Val set
    val_results  = evaluate(model, val_loader,  split="Val")
    # Test set
    test_results = evaluate(model, test_loader, split="Test")

    # Plots
    if history:
        plot_training_curves(history)
    plot_confusion_matrix(val_results["confusion_matrix"],  split="Val")
    plot_confusion_matrix(test_results["confusion_matrix"], split="Test")

    print_banner("Pipeline Complete!")
    print(f"\n  Test Accuracy   : {test_results['accuracy']*100:.2f}%")
    print(f"  Test F1 (weighted): {test_results['f1_weighted']:.4f}")
    print(f"\n  Outputs saved in: /mnt/user-data/outputs/")


# ── Inference helper ──────────────────────────────────────────────────────────
def predict_single(text: str, model: SentimentTransformer, vocab: Vocabulary) -> dict:
    """
    Dự đoán sentiment cho một câu văn bản.
    """
    from config import MAX_SEQ_LEN
    import torch.nn.functional as F

    model.eval()
    ids  = vocab.encode(text, MAX_SEQ_LEN)
    mask = [1 if x != 0 else 0 for x in ids]

    input_ids = torch.tensor([ids],  dtype=torch.long).to(DEVICE)
    attn_mask = torch.tensor([mask], dtype=torch.bool).to(DEVICE)

    with torch.no_grad():
        out    = model(input_ids, attn_mask)
        probs  = F.softmax(out["logits"], dim=-1)[0]
        pred   = probs.argmax().item()

    label_names = ["Negative", "Neutral", "Positive"]
    return {
        "text":       text,
        "prediction": label_names[pred],
        "confidence": f"{probs[pred].item()*100:.1f}%",
        "probs": {
            "Negative": f"{probs[0].item()*100:.1f}%",
            "Neutral":  f"{probs[1].item()*100:.1f}%",
            "Positive": f"{probs[2].item()*100:.1f}%",
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment Transformer Pipeline")
    parser.add_argument("--eval-only", action="store_true",
                        help="Chỉ evaluate từ checkpoint đã có, bỏ qua training")
    args = parser.parse_args()

    run_pipeline(eval_only=args.eval_only)
