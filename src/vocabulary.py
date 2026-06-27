"""
vocabulary.py — Word-level tokenizer + Vocabulary builder
"""
import re
import json
import os
from collections import Counter
from config import MAX_VOCAB_SIZE, MIN_FREQ, SAVE_DIR


# ── Special tokens ────────────────────────────────────────────────────────────
PAD_TOKEN = "<PAD>"   # index 0 — padding
UNK_TOKEN = "<UNK>"   # index 1 — unknown words
BOS_TOKEN = "<BOS>"   # index 2 — beginning of sequence  (reserved, optional)
EOS_TOKEN = "<EOS>"   # index 3 — end of sequence        (reserved, optional)

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
PAD_IDX = 0
UNK_IDX = 1


def basic_tokenize(text: str) -> list[str]:
    """
    Simple word-level tokenizer:
      - lowercase
      - giữ lại apostrophe contraction (don't → don't)
      - split trên khoảng trắng / ký tự đặc biệt còn lại
    """
    text = str(text).lower().strip()
    # Giữ lại chữ, số, apostrophe
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    # Normalize whitespace
    tokens = text.split()
    return tokens


class Vocabulary:
    """
    Xây dựng vocab từ danh sách câu, hỗ trợ encode/decode.
    """

    def __init__(self):
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self.word_freq: Counter = Counter()
        self._built = False

    # ── Build ─────────────────────────────────────────────────────────────────
    def build(self, texts: list[str]) -> None:
        """
        texts: list các raw text strings (train split only!).
        """
        print("Building vocabulary...")

        # Đếm tần suất
        for text in texts:
            tokens = basic_tokenize(text)
            self.word_freq.update(tokens)

        # Lọc theo MIN_FREQ và top-K
        valid_words = [
            word for word, freq in self.word_freq.most_common(MAX_VOCAB_SIZE)
            if freq >= MIN_FREQ
        ]

        # Gán index: special tokens trước, sau đó words theo frequency
        all_tokens = SPECIAL_TOKENS + valid_words
        self.word2idx = {tok: idx for idx, tok in enumerate(all_tokens)}
        self.idx2word = {idx: tok for tok, idx in self.word2idx.items()}
        self._built = True

        print(f"  Total unique words in train : {len(self.word_freq):,}")
        print(f"  Vocab size (after filtering): {len(self.word2idx):,}")

    # ── Encode / Decode ───────────────────────────────────────────────────────
    def encode(self, text: str, max_len: int) -> list[int]:
        """
        Chuyển text → list indices, pad/truncate về max_len.
        """
        assert self._built, "Call build() first."
        tokens = basic_tokenize(text)[:max_len]
        ids = [self.word2idx.get(tok, UNK_IDX) for tok in tokens]
        # Padding
        ids += [PAD_IDX] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        return " ".join(
            self.idx2word.get(i, UNK_TOKEN)
            for i in ids
            if i != PAD_IDX
        )

    def __len__(self) -> int:
        return len(self.word2idx)

    # ── Save / Load ───────────────────────────────────────────────────────────
    def save(self, path: str | None = None) -> None:
        path = path or os.path.join(SAVE_DIR, "vocab.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx}, f, ensure_ascii=False, indent=2)
        print(f"Vocabulary saved → {path}")

    def load(self, path: str | None = None) -> None:
        path = path or os.path.join(SAVE_DIR, "vocab.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self._built = True
        print(f"Vocabulary loaded ← {path}  |  size: {len(self.word2idx):,}")
