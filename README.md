# 🤖 Sentiment Analysis - Transformer From Scratch (PyTorch)

A full end-to-end NLP pipeline that implements a **Transformer Encoder from scratch** in PyTorch for 3-class sentiment classification — without relying on `nn.Transformer` or pretrained models.

---

## 📊 Dataset

The dataset is sourced from **Kaggle** and contains **241,145 user comments** for 3-class sentiment classification.

| Attribute               | Detail                                                                                                                |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Source:**             | [Kaggle - Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abdelmalekeladjelet/sentiment-analysis-dataset) |
| **File**                | `sentiment_data_cleaned.csv`                                                                                          |
| **Total samples**       | **241,145 rows**                                                                                                      |
| **Columns**             | `Comment` (text), `Sentiment` (label)                                                                                 |
| **Classes**             | `0` = Negative · `1` = Neutral · `2` = Positive                                                                       |
| **Domain**              | User comments about mobile payment services (e.g. Apple Pay)                                                          |
| **Avg. comment length** | ~20 words per comment                                                                                                 |

### Class Distribution

| Label     | Sentiment | Samples     | Percentage |
| --------- | --------- | ----------- | ---------- |
| **0**     | Negative  | 55,114      | 22.85%     |
| **1**     | Neutral   | 82,972      | 34.41%     |
| **2**     | Positive  | 103,059     | 42.75%     |
| **Total** | —         | **241,145** | **100%**   |

**Label mapping:**

- **0 — Negative:** complaints, dissatisfaction, limitations ("don't like the high fees", "doesn't take Apple Pay")
- **1 — Neutral:** factual or mixed observations ("required brand new iPhone to use")
- **2 — Positive:** satisfaction, convenience, praise ("convenient, secure, easy to use")

> ⚠️ Dataset is class-imbalanced — handled via **class weighting** during training.

---

## 🏗️ Architecture

```
Input text
    │
    ▼
Word-level Tokenizer & Vocabulary (40k vocab, freq ≥ 2)
    │
    ▼
Token Embedding (128-dim) + Sinusoidal Positional Encoding
    │
    ▼
┌─────────────────────────────────┐
│  Transformer Encoder Block × 3  │
│  ┌───────────────────────────┐  │
│  │  Pre-LayerNorm            │  │
│  │  Multi-Head Self-Attention│  │  (4 heads, 32-dim each)
│  │  + Residual connection    │  │
│  ├───────────────────────────┤  │
│  │  Pre-LayerNorm            │  │
│  │  Feed-Forward (256-dim)   │  │  GELU activation
│  │  + Residual connection    │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
    │
    ▼
Mean Pooling (non-PAD tokens only)
    │
    ▼
Classifier Head: Linear(128→64) → GELU → Dropout → Linear(64→3)
    │
    ▼
Softmax → {Negative, Neutral, Positive}
```

**Key design choices:**

- **Pre-LayerNorm** (before attention) for more stable training than Post-LN
- **Masked mean pooling** — averages only non-padding token representations
- **Xavier uniform init** for projection weights, Normal(0, 0.02) for embeddings
- Built entirely from `nn.Linear`, `nn.Embedding`, `nn.LayerNorm` — no `nn.Transformer`

---

## ⚙️ Configuration

| Hyperparameter      | Value                                    |
| ------------------- | ---------------------------------------- |
| Vocab size          | 40,000 (top-K by freq)                   |
| Min word frequency  | 2                                        |
| Max sequence length | 64                                       |
| Embedding dim       | 128                                      |
| Attention heads     | 4                                        |
| Encoder layers      | 3                                        |
| FFN hidden dim      | 256                                      |
| Dropout             | 0.3                                      |
| Batch size          | 512                                      |
| Max epochs          | 30 (early stopping patience = 5)         |
| Optimizer           | AdamW (lr=1e-4, weight decay=1e-4)       |
| Scheduler           | Linear warmup (500 steps) → Cosine decay |
| Gradient clipping   | 1.0                                      |

---

## 🚀 Training Pipeline

```
1. Load & clean data
      │
2. Stratified split → Train 70% / Val 20% / Test 10%
      │
3. Build vocabulary from train set only (no leakage)
      │
4. Compute class weights for imbalanced labels
      │
5. Train with AdamW + warmup-cosine scheduler + early stopping
      │
6. Load best checkpoint (lowest val_loss)
      │
7. Evaluate on test set → metrics + confusion matrix
      │
8. Inference on new sentences
```

---

## 📈 Results

| Metric                 | Score      |
| ---------------------- | ---------- |
| **Test Accuracy**      | **80.97%** |
| **Test F1 (weighted)** | **0.8100** |

**Per-class performance (Test Set):**

| Class    | Precision | Recall |
| -------- | --------- | ------ |
| Negative | ~80%      | 80.25% |
| Neutral  | ~75%      | 75.18% |
| Positive | ~86%      | 85.99% |

---

## 🛠️ Tech Stack

| Category        | Tools               |
| --------------- | ------------------- |
| Language        | Python 3            |
| Deep Learning   | PyTorch             |
| Data processing | pandas, NumPy       |
| Evaluation      | scikit-learn        |
| Visualization   | matplotlib, seaborn |
| Environment     | Kaggle (T4 GPU)     |

---

## 📂 Project Structure

```
├── sentiment_data_cleaned.csv       # Dataset
├── sentiment-transformer-kaggle.ipynb  # Full pipeline notebook
└── outputs/
    ├── checkpoints/
    │   ├── best_model.pt            # Best model weights
    │   └── vocab.json               # Saved vocabulary
    ├── plots/
    │   ├── training_curves.png      # Loss / Accuracy / LR curves
    │   ├── cm_val.png               # Confusion matrix (Val)
    │   └── cm_test.png              # Confusion matrix (Test)
    └── logs/
        └── history.json             # Training history
```

---

## 🔍 Inference Example

```python
predict_text("apple pay is absolutely amazing and so convenient to use")
# → Prediction: Positive (91.3% confidence)

predict_text("i don't like the new update it crashed my phone twice")
# → Prediction: Negative (85.7% confidence)

predict_text("the product arrived on time and packaging was okay")
# → Prediction: Neutral (72.1% confidence)
```

---

## 👤 Author

**Truong Van Thien**  
Third-year Computer Science student — Saigon University  
📧 truongthien1334@email.com  
🔗 [github.com/tvThien2005](https://github.com/tvThien2005)
