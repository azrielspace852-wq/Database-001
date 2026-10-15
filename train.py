#!/usr/bin/env python3
"""
Pretrain TinyLM-ContextMLP untuk GitHub Actions.

Model:
- Vocabulary size: 5000
- Context window: 16 token
- H (hidden/embedding): 125
- D: 125
- Max steps: 10000
- Batch: 100
- Loss: cross-entropy dari semua token
- Checkpoint: setiap 10 batch
- Output final: output.zip berisi <model_name>.bin dan manifest.json
"""

import argparse
import json
import math
import os
import pathlib
import random
import shutil
import struct
import time
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

# ================= Konfigurasi =================
VOCAB_SIZE = 5000
CONTEXT_WINDOW = 16
H = 125
D = 125
MAX_STEPS = 10000
BATCH_SIZE = 100
MAX_OUTPUT_TOKENS = 2084
CHECKPOINT_EVERY = 10
TOKENS_PER_STEP = BATCH_SIZE * CONTEXT_WINDOW

# Token khusus
PAD = 0
UNK = 1
BOS = 2
EOS = 3
FIRST_REAL_TOKEN = 4
TOTAL_SPECIAL = 4

# Arsitektur TinyLM-ContextMLP
# E: Embedding [V, H]
# W1: Dense+Tanh [cw*H, H]
# b1: bias [H]
# W2: Dense+Softmax [H, V]
# b2: bias [V]
PARAMETER_COUNT = (
    VOCAB_SIZE * H
    + CONTEXT_WINDOW * H * H
    + H
    + H * VOCAB_SIZE
    + VOCAB_SIZE
)
PARAMETER_BYTES_FLOAT32 = PARAMETER_COUNT * 4


class Adam:
    def __init__(self, params: List[np.ndarray], lr=3e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: List[np.ndarray]):
        self.t += 1
        for i, p in enumerate(self.params):
            g = grads[i]
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


def tokenize_text(text: str, vocab: Dict[str, int]) -> List[int]:
    """
    Tokenizer sederhana mirip regex AXION:
    - Kata alfanumerik (lowercase untuk hitung freq)
    - Angka
    - Simbol non-spasi
    """
    import re
    pattern = r"\p{L}[\p{L}\p{M}]*|\p{N}+|[^\s]"
    # Python re tidak support \p, jadi pakai versi pendekatan:
    # [A-Za-z0-9]+ | [^\s]
    tokens: List[int] = []
    for m in re.finditer(r"[A-Za-z0-9]+|[^\s]", text.lower()):
        tok = m.group()
        tokens.append(vocab.get(tok, UNK))
    return tokens


def build_vocab(texts: List[str], target_size: int) -> Dict[str, int]:
    """
    Build vocab dari dataset. Vocab id 0-3 special, sisanya frekuensi.
    Ukuran final dipaksa target_size.
    """
    counter: Counter[str] = Counter()
    for text in texts:
        import re
        for m in re.finditer(r"[A-Za-z0-9]+|[^\s]", text.lower()):
            counter[m.group()] += 1

    vocab: Dict[str, int] = {
        "<pad>": PAD,
        "<unk>": UNK,
        "<bos>": BOS,
        "<eos>": EOS,
    }
    needed = target_size - TOTAL_SPECIAL
    for tok, _ in counter.most_common(needed):
        if tok not in vocab:
            vocab[tok] = len(vocab)

    # Fill jika kurang
    i = len(vocab)
    while i < target_size:
        vocab[f"__fill_{i}"] = i
        i += 1
    return vocab


def load_dataset(jsonl_files: List[pathlib.Path]) -> List[str]:
    texts = []
    for f in jsonl_files:
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Structure: {"text": {"lang": "", "category": "", "content": ""}}
                content = ""
                if isinstance(obj, dict):
                    text_field = obj.get("text", {})
                    if isinstance(text_field, dict):
                        content = text_field.get("content", "")
                    else:
                        content = str(text_field)
                texts.append(content)
    return texts


def make_sequences(
    token_ids: List[int], seq_len: int, window: int
) -> List[List[int]]:
    """
    Buat training pair: [seq, target]
    seq: window token
    target: token berikutnya
    """
    samples = []
    for i in range(0, len(token_ids) - window, seq_len):
        seq = token_ids[i : i + window]
        target = token_ids[i + window : i + window + seq_len]
        if len(target) == seq_len:
            samples.append((seq, target))
    return samples


def forward(
    seqs: np.ndarray,
    E: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
):
    """
    Forward pass untuk batch.
    seqs: [B, cw]
    return logits: [B, V]
    """
    B, cw = seqs.shape
    V, Hh = E.shape

    # Embedding
    x = E[seqs]  # [B, cw, H]

    # Flatten context
    x_flat = x.reshape(B, cw * Hh)  # [B, cw*H]

    # Dense + Tanh
    h = np.tanh(x_flat @ W1 + b1)  # [B, H]

    # Language model head
    logits = h @ W2 + b2  # [B, V]
    return logits, x_flat, h


def cross_entropy_loss_and_grads(
    logits: np.ndarray, targets: np.ndarray
) -> tuple:
    """
    Stable softmax cross-entropy.
    logits: [B, V], targets: [B]
    Return: (loss, d_logits)
    """
    B, V = logits.shape
    # log-softmax numerik stabil
    max_logit = np.max(logits, axis=1, keepdims=True)
    log_probs = logits - max_logit - np.log(np.sum(np.exp(logits - max_logit), axis=1, keepdims=True))
    loss = -np.mean(log_probs[np.arange(B), targets])

    # Gradient
    probs = np.exp(log_probs)
    d_logits = probs
    d_logits[np.arange(B), targets] -= 1.0
    d_logits /= B
    return loss, d_logits


def save_checkpoint(
    path: pathlib.Path,
    E: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    step: int,
    loss: float,
    vocab: Dict[str, int],
    model_name: str,
):
    """
    Simpan checkpoint .bin sederhana dengan struktur:
    Header JSON + tensors berturut-turut.
    """
    header = {
        "format": "mini-lab-ai-model",
        "version": "3.0.0",
        "schema_version": 3,
        "model": {
            "name": model_name,
            "file": f"{model_name}.bin",
            "architecture": "TinyLM-ContextMLP",
            "input_shape": [CONTEXT_WINDOW, H],
            "output_shape": [VOCAB_SIZE],
            "layers": [
                {"name": "embedding", "type": "Embedding", "output_shape": [CONTEXT_WINDOW, H]},
                {"name": "context_mlp", "type": "Dense+Tanh", "output_shape": [H]},
                {"name": "language_model_head", "type": "Dense+Softmax", "output_shape": [VOCAB_SIZE]},
            ],
            "optimizer": "Adam",
            "loss": "cross-entropy",
            "hyperparameters": {"cw": CONTEXT_WINDOW, "D": D, "H": H},
            "configuration": {
                "model_type": "pretrain",
                "cfg": {"cw": CONTEXT_WINDOW, "D": D, "H": H},
                "parameter_count": PARAMETER_COUNT,
                "parameter_bytes_float32": PARAMETER_BYTES_FLOAT32,
            },
            "tensors": [
                {"name": "E", "dtype": "float32", "shape": [VOCAB_SIZE, H], "length": VOCAB_SIZE * H},
                {"name": "W1", "dtype": "float32", "shape": [CONTEXT_WINDOW * H, H], "length": CONTEXT_WINDOW * H * H},
                {"name": "b1", "dtype": "float32", "shape": [H], "length": H},
                {"name": "W2", "dtype": "float32", "shape": [H, VOCAB_SIZE], "length": H * VOCAB_SIZE},
                {"name": "b2", "dtype": "float32", "shape": [VOCAB_SIZE], "length": VOCAB_SIZE},
            ],
            "tokenizer": {
                "type": "AXION AI-Lab regex tokenizer",
                "pattern": "\\p{L}[\\p{L}\\p{M}]*|\\p{N}+|[^\\s]",
                "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
            },
            "vocab": {
                "size": VOCAB_SIZE,
                "id_to_token": [k for k, v in sorted(vocab.items(), key=lambda x: x[1])],
            },
        },
        "training": {
            "step": step,
            "loss": loss,
        },
    }
    header_json = json.dumps(header, ensure_ascii=False, indent=2).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(header_json)))
        f.write(header_json)
        for tensor in [E, W1, b1, W2, b2]:
            f.write(tensor.astype(np.float32).tobytes())


def save_final_model(
    out_dir: pathlib.Path,
    E: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    vocab: Dict[str, int],
    model_name: str,
    stats: dict,
):
    bin_name = f"{model_name}.bin"
    bin_path = out_dir / bin_name

    # Bin format sama dengan checkpoint, tapi header final
    save_checkpoint(bin_path, E, W1, b1, W2, b2, stats.get("step", 0), stats.get("loss", 0.0), vocab, model_name)

    # Manifest lengkap sesuai contoh lampiran
    manifest = {
        "format": "mini-lab-ai-model",
        "version": "3.0.0",
        "schema_version": 3,
        "model": {
            "name": model_name,
            "file": bin_name,
            "architecture": "TinyLM-ContextMLP",
            "input_shape": [CONTEXT_WINDOW, H],
            "output_shape": [VOCAB_SIZE],
            "layers": [
                {"name": "embedding", "type": "Embedding", "output_shape": [CONTEXT_WINDOW, H]},
                {"name": "context_mlp", "type": "Dense+Tanh", "output_shape": [H]},
                {"name": "language_model_head", "type": "Dense+Softmax", "output_shape": [VOCAB_SIZE]},
            ],
            "optimizer": "Adam",
            "loss": "cross-entropy",
            "metrics": [],
            "hyperparameters": {"cw": CONTEXT_WINDOW, "D": D, "H": H},
            "configuration": {
                "model_type": "pretrain",
                "cfg": {"cw": CONTEXT_WINDOW, "D": D, "H": H},
                "parameter_count": PARAMETER_COUNT,
                "parameter_bytes_float32": PARAMETER_BYTES_FLOAT32,
            },
            "tensors": [
                {"name": "E", "dtype": "float32", "shape": [VOCAB_SIZE, H], "length": VOCAB_SIZE * H},
                {"name": "W1", "dtype": "float32", "shape": [CONTEXT_WINDOW * H, H], "length": CONTEXT_WINDOW * H * H},
                {"name": "b1", "dtype": "float32", "shape": [H], "length": H},
                {"name": "W2", "dtype": "float32", "shape": [H, VOCAB_SIZE], "length": H * VOCAB_SIZE},
                {"name": "b2", "dtype": "float32", "shape": [VOCAB_SIZE], "length": VOCAB_SIZE},
            ],
            "tokenizer": {
                "type": "AXION AI-Lab regex tokenizer",
                "pattern": "\\p{L}[\\p{L}\\p{M}]*|\\p{N}+|[^\\s]",
                "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
            },
            "preprocessing": {
                "text_source": "dataset.zip",
                "masking": "none",
                "content_tokenization": True,
                "captions_tokenization": True,
            },
            "vocab": {
                "size": VOCAB_SIZE,
                "id_to_token": [k for k, v in sorted(vocab.items(), key=lambda x: x[1])],
            },
            "labels": [],
            "metadata": {
                "id": model_name,
                "type": "pretrain",
                "parent_id": None,
                "stats": stats,
            },
        },
        "text_model": {
            "architecture": "TinyLM-ContextMLP",
            "cfg": {"cw": CONTEXT_WINDOW, "D": D, "H": H},
            "vocab_size": VOCAB_SIZE,
            "tokenizer": {
                "type": "AXION AI-Lab regex tokenizer",
                "pattern": "\\p{L}[\\p{L}\\p{M}]*|\\p{N}+|[^\\s]",
                "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
            },
            "binary_tensors": ["E", "W1", "b1", "W2", "b2"],
        },
        "multimodal": {"enabled": False, "encoders": []},
        "training": {
            "last_loss": stats.get("last_loss", 0.0),
            "best_loss": stats.get("best_loss", 0.0),
            "step": stats.get("step", 0),
            "max_steps": MAX_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": 3e-4,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "dataset": {
                "name": "dataset.zip",
                "tokens": stats.get("tokens", 0),
                "documents": stats.get("documents", 0),
            },
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "model"))
    parser.add_argument("--data-dir", default="dataset_extracted")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    model_name = args.model_name
    data_dir = pathlib.Path(args.data_dir)

    # Carikan semua .jsonl
    jsonl_files = list(data_dir.rglob("*.jsonl"))
    if not jsonl_files:
        # fallback: semua .json
        jsonl_files = list(data_dir.rglob("*.json"))
    print(f"Menemukan {len(jsonl_files)} file dataset: {jsonl_files}")

    texts = load_dataset(jsonl_files)
    print(f"Total dokumen teks: {len(texts)}")

    vocab = build_vocab(texts, VOCAB_SIZE)
    print(f"Vocab size: {len(vocab)}")

    # Tokenisasi
    all_tokens: List[int] = []
    total_tokens = 0
    for text in texts:
        ids = tokenize_text(text, vocab)
        all_tokens.extend(ids)
        all_tokens.append(EOS)
        total_tokens += len(ids)

    print(f"Total token: {total_tokens}")

    # Inisialisasi parameter
    rng = np.random.default_rng(seed=42)
    E = rng.normal(0, 0.02, size=(VOCAB_SIZE, H)).astype(np.float32)
    W1 = rng.normal(0, 0.02, size=(CONTEXT_WINDOW * H, H)).astype(np.float32)
    b1 = np.zeros((H,), dtype=np.float32)
    W2 = rng.normal(0, 0.02, size=(H, VOCAB_SIZE)).astype(np.float32)
    b2 = np.zeros((VOCAB_SIZE,), dtype=np.float32)

    optimizer = Adam([E, W1, b1, W2, b2])

    # Siapkan batch sampel
    samples = make_sequences(all_tokens, seq_len=BATCH_SIZE, window=CONTEXT_WINDOW)
    print(f"Total sample: {len(samples)}")
    if len(samples) == 0:
        # Jika dataset terlalu kecil, pakai sliding window lebih agresif
        samples = make_sequences(all_tokens, seq_len=1, window=CONTEXT_WINDOW)
        print(f"Total sample (fallback): {len(samples)}")

    # Training
    ckpt_dir = pathlib.Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    best_loss = float("inf")
    last_loss = 0.0
    start_time = time.time()
    token_consumed = 0

    while step < args.max_steps:
        # Ambil batch
        batch_idx = rng.choice(len(samples), size=BATCH_SIZE, replace=True)
        batch_seqs = np.array([samples[i][0] for i in batch_idx], dtype=np.int32)
        batch_targets = np.array([samples[i][1] for i in batch_idx if samples[i][1]], dtype=np.int32)
        if len(batch_targets) < BATCH_SIZE:
            # Jika sample target length-nya pendek, pad
            # Untuk script sederhana: ambil token berikutnya saja
            batch_targets = np.array([all_tokens[(min(i + CONTEXT_WINDOW + s, len(all_tokens)-1))] for i, s in zip(batch_idx, [0]*BATCH_SIZE)], dtype=np.int32)

        # Forward
        logits, x_flat, h = forward(batch_seqs, E, W1, b1, W2, b2)

        # Loss: semua token di batch
        loss, d_logits = cross_entropy_loss_and_grads(logits, batch_targets)
        last_loss = loss
        if loss < best_loss:
            best_loss = loss

        # Backprop
        B = batch_seqs.shape[0]
        dh = d_logits @ W2.T  # [B, H]
        dx_flat = dh * (1 - h * h) @ W1.T  # [B, cw*H]
        dx = dx_flat.reshape(B, CONTEXT_WINDOW, H)
        dW2 = h.T @ d_logits  # [H, V]
        db2 = np.sum(d_logits, axis=0)
        dW1 = x_flat.T @ (dh * (1 - h * h))  # [cw*H, H]
        db1 = np.sum(dh * (1 - h * h), axis=0)
        dE = np.zeros_like(E)
        np.add.at(dE, (np.arange(B)[:, None], batch_seqs), dx)
        dE /= B

        # Optimizer step
        optimizer.step([dE, dW1, db1, dW2, db2])

        step += 1
        token_consumed += B

        # Cek checkpoint
        if step % CHECKPOINT_EVERY == 0 or step == args.max_steps:
            ckpt_path = ckpt_dir / f"checkpoint_{step}.bin"
            save_checkpoint(ckpt_path, E, W1, b1, W2, b2, step, last_loss, vocab, model_name)
            print(f"[Step {step}/{args.max_steps}] loss={last_loss:.6f} best={best_loss:.6f} -> {ckpt_path.name}")

    stats = {
        "steps": step,
        "loss": last_loss,
        "best_loss": best_loss,
        "dataset": "dataset.zip",
        "tokens": total_tokens,
        "docs": len(texts),
        "elapsed": time.time() - start_time,
        "documents": len(texts),
    }

    # Simpan final
    out_dir = pathlib.Path(".")
    save_final_model(out_dir, E, W1, b1, W2, b2, vocab, model_name, stats)
    print(f"Final model: {model_name}.bin")

    # Buat zip
    bin_path = pathlib.Path(f"{model_name}.bin")
    manifest_path = pathlib.Path("manifest.json")
    zip_path = pathlib.Path("output.zip")
    shutil.make_archive(str(zip_path.without_suffix()), "zip", root_dir=".", base_dir="")
    # make_archive membuat output.zip di root, tapi kita butuh isi: <model>.bin + manifest.json
    import zipfile
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(bin_path, bin_path.name)
        z.write(manifest_path, manifest_path.name)
    print(f"ZIP dibuat: {zip_path}")


if __name__ == "__main__":
    main()
