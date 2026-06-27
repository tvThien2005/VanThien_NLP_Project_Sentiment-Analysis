
import os

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH   = "../Data/sentiment_data_cleaned.csv"
SAVE_DIR    = "../Data/outputs/checkpoints"
LOG_DIR     = "../Data/outputs/logs"
PLOT_DIR    = "../Data/outputs/plots"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10
RANDOM_SEED = 42

# ── Vocabulary ───────────────────────────────────────────────────────────────
MAX_VOCAB_SIZE  = 40000   # top-K words by frequency
MAX_SEQ_LEN     = 64       # cắt/pad về độ dài này (phù hợp với text ngắn ~13 words)
MIN_FREQ        = 2        # bỏ qua từ xuất hiện < MIN_FREQ lần

# ── Model Architecture ───────────────────────────────────────────────────────
NUM_CLASSES     = 3        # Negative / Neutral / Positive
EMBED_DIM       = 128      # embedding dimension
NUM_HEADS       = 4        # multi-head attention heads  (EMBED_DIM % NUM_HEADS == 0)
NUM_LAYERS      = 3        # số Transformer Encoder layers
FFN_DIM         = 256      # feed-forward hidden dim (thường 2x EMBED_DIM)
DROPOUT         = 0.2      # dropout rate
MAX_POSITION    = 512      # max positional encoding length

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 256      # tăng nếu VRAM còn nhiều
NUM_EPOCHS      = 20
LEARNING_RATE   = 0.0001
WEIGHT_DECAY    = 1e-4
WARMUP_STEPS    = 500      # linear warmup cho scheduler
GRAD_CLIP       = 1.0      # gradient clipping để tránh exploding gradient
EARLY_STOP_PAT  = 5        # dừng nếu val_loss không cải thiện sau N epochs

# ── Class weights (xử lý imbalance) ─────────────────────────────────────────
# Neg: 22.9% | Neu: 34.4% | Pos: 42.8%  → inverse frequency weighting
USE_CLASS_WEIGHTS = True   # tự tính từ data nếu True

# ── Device ───────────────────────────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
