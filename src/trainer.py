"""
trainer.py — Training loop với:
  - Linear Warmup + Cosine Annealing LR Scheduler
  - Gradient Clipping
  - Early Stopping
  - Checkpoint (best model theo val_loss)
  - Live logging per epoch
"""
import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math

from config import (
    LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    GRAD_CLIP, WARMUP_STEPS, EARLY_STOP_PAT,
    SAVE_DIR, LOG_DIR, DEVICE
)


# ── LR Scheduler: Warmup + Cosine Decay ──────────────────────────────────────
def get_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """
    - Bước 0 → warmup_steps : LR tăng tuyến tính 0 → LR_max
    - Sau warmup             : cosine decay → LR_min (10% LR_max)
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ── Metric helpers ────────────────────────────────────────────────────────────
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


# ── Single epoch ──────────────────────────────────────────────────────────────
def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: LambdaLR | None,
    phase: str,          # "train" | "val" | "test"
) -> tuple[float, float]:
    """Returns (avg_loss, avg_accuracy)."""
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            input_ids  = batch["input_ids"].to(DEVICE)
            attn_mask  = batch["attention_mask"].to(DEVICE)
            labels     = batch["label"].to(DEVICE)

            out    = model(input_ids, attn_mask)
            logits = out["logits"]
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
            total_acc  += accuracy(logits, labels)
            n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


# ── Main Trainer ──────────────────────────────────────────────────────────────
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor | None = None,
    ):
        self.model        = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader   = val_loader

        # Loss with optional class weighting
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer: AdamW (decoupled weight decay)
        self.optimizer = AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        # Scheduler
        total_steps = len(train_loader) * NUM_EPOCHS
        self.scheduler = get_cosine_warmup_scheduler(
            self.optimizer, WARMUP_STEPS, total_steps
        )

        # State
        self.history: dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
            "lr":         [],
        }
        self.best_val_loss = float("inf")
        self.patience_cnt  = 0
        self.best_ckpt     = os.path.join(SAVE_DIR, "best_model.pt")

    def train(self) -> dict:
        print(f"\n{'='*60}")
        print(f"  Training on: {DEVICE}")
        print(f"  Params     : {self.model.count_parameters():,}")
        print(f"  Epochs     : {NUM_EPOCHS}  |  Batch: {self.train_loader.batch_size}")
        print(f"{'='*60}\n")

        for epoch in range(1, NUM_EPOCHS + 1):
            t0 = time.time()

            tr_loss, tr_acc = _run_epoch(
                self.model, self.train_loader, self.criterion,
                self.optimizer, self.scheduler, "train"
            )
            vl_loss, vl_acc = _run_epoch(
                self.model, self.val_loader, self.criterion,
                None, None, "val"
            )

            elapsed = time.time() - t0
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log
            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(vl_loss)
            self.history["val_acc"].append(vl_acc)
            self.history["lr"].append(current_lr)

            print(
                f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
                f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f} | "
                f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f} | "
                f"LR: {current_lr:.2e} | {elapsed:.1f}s"
            )

            # ── Checkpoint (best val_loss) ────────────────────────────────────
            if vl_loss < self.best_val_loss:
                self.best_val_loss = vl_loss
                self.patience_cnt  = 0
                torch.save({
                    "epoch":       epoch,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "val_loss":    vl_loss,
                    "val_acc":     vl_acc,
                }, self.best_ckpt)
                print(f"  ✓ Best model saved (val_loss={vl_loss:.4f})")
            else:
                self.patience_cnt += 1
                print(f"  · No improvement ({self.patience_cnt}/{EARLY_STOP_PAT})")
                if self.patience_cnt >= EARLY_STOP_PAT:
                    print(f"\n⚡ Early stopping triggered at epoch {epoch}!")
                    break

        # Save history
        hist_path = os.path.join(LOG_DIR, "history.json")
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\nTraining history saved → {hist_path}")
        print(f"Best val_loss: {self.best_val_loss:.4f}")

        return self.history

    def load_best(self) -> None:
        """Load best checkpoint vào model."""
        ckpt = torch.load(self.best_ckpt, map_location=DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"Best model loaded from epoch {ckpt['epoch']}  (val_loss={ckpt['val_loss']:.4f})")
