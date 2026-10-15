#!/usr/bin/env python3
"""
TinyLM-ContextMLP Pretraining Script (Updated)
- Builds vocabulary from actual dataset content
- Handles nested JSONL structure: {"text": {"content": ...}}
- Produces web-compatible .bin + manifest.json
- Checkpointing every N batches
"""

import os
import sys
import json
import struct
import hashlib
import argparse
import glob
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================================
# CONFIGURATION & CONSTANTS (UPDATED)
# ============================================================================

# Python re-compatible pattern (equivalent to \p{L}[\p{L}\p{M}]*|\p{N}+|[^\s])
# \w matches [a-zA-Z0-9_] + Unicode letters/digits in Python 3 with re.UNICODE
# This is the closest stdlib equivalent without requiring 'regex' package
TOKENIZER_PATTERN = r"\w+|[^\s\w]"
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


# ============================================================================
# VOCABULARY BUILDER (FIXED)
# ============================================================================

def build_vocab_from_data(data_dir: str, vocab_size: int) -> Tuple[Dict[str, int], List[str]]:
    """
    Build vocabulary from actual dataset content.
    Uses Python stdlib re (no \p{} support needed).
    Returns (token_to_id dict, id_to_token list).
    """
    counter = Counter()
    jsonl_files = sorted(glob.glob(os.path.join(data_dir, "**/*.jsonl"), recursive=True))

    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {data_dir}")

    print(f"📚 Building vocab from {len(jsonl_files)} files...")
    # Use stdlib-compatible pattern with re.UNICODE flag
    pattern = re.compile(TOKENIZER_PATTERN, re.UNICODE)

    for fpath in tqdm(jsonl_files, desc="Scanning tokens"):
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    content = ""
                    if isinstance(obj.get("text"), dict):
                        content = obj["text"].get("content", "")
                    elif isinstance(obj.get("text"), str):
                        content = obj["text"]

                    if content:
                        tokens = pattern.findall(content.lower())
                        counter.update(tokens)
                except (json.JSONDecodeError, KeyError):
                    continue

    # Reserve slots for special tokens, fill rest with most frequent
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    most_common = counter.most_common(vocab_size - len(SPECIAL_TOKENS))

    for tok, _ in most_common:
        if tok not in vocab:
            vocab[tok] = len(vocab)
        if len(vocab) >= vocab_size:
            break

    # Build reverse lookup
    id_to_token = ["<unk>"] * vocab_size
    for tok, idx in vocab.items():
        if idx < vocab_size:
            id_to_token[idx] = tok

    print(f"✅ Vocab built: {len(vocab)} unique tokens (capped at {vocab_size})")
    print(f"   Top 10: {[t for t, _ in most_common[:10]]}")
    return vocab, id_to_token


# ============================================================================
# DATASET
# ============================================================================

class JsonlTextDataset(Dataset):
    """Handles nested JSONL structure: {"text": {"lang": "", "category": "", "content": ""}}"""

    def __init__(self, data_dir: str, vocab: Dict[str, int], context_window: int):
        self.vocab = vocab
        self.context_window = context_window
        self.tokens: List[int] = []
        self.pattern = re.compile(TOKENIZER_PATTERN, re.UNICODE)

        jsonl_files = sorted(glob.glob(os.path.join(data_dir, "**/*.jsonl"), recursive=True))
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in {data_dir}")

        print(f"📖 Loading {len(jsonl_files)} JSONL files...")
        doc_count = 0
        for fpath in tqdm(jsonl_files, desc="Parsing"):
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        content = ""
                        if isinstance(obj.get("text"), dict):
                            content = obj["text"].get("content", "")
                        elif isinstance(obj.get("text"), str):
                            content = obj["text"]

                        if content:
                            toks = self._tokenize(content)
                            if toks:
                                self.tokens.extend(toks)
                                self.tokens.append(vocab["<eos>"])
                                doc_count += 1
                    except (json.JSONDecodeError, KeyError):
                        continue

        if len(self.tokens) < context_window + 1:
            raise ValueError(
                f"Insufficient tokens ({len(self.tokens)}) for context_window {context_window}. "
                f"Need at least {context_window + 1}."
            )

        self.doc_count = doc_count
        print(f"✅ Total tokens: {len(self.tokens)} | Documents: {doc_count}")

    def _tokenize(self, text: str) -> List[int]:
        # Use stdlib-compatible pattern (same as build_vocab_from_data)
        raw_tokens = re.findall(TOKENIZER_PATTERN, text.lower(), re.UNICODE)
        unk_id = self.vocab.get("<unk>", 1)
        return [self.vocab.get(t, unk_id) for t in raw_tokens]

    def __len__(self):
        return max(0, len(self.tokens) - self.context_window)

    def __getitem__(self, idx):
        x = self.tokens[idx: idx + self.context_window]
        y = self.tokens[idx + 1: idx + self.context_window + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class TinyLMContextMLP(nn.Module):
    """
    Architecture:
      - Embedding(VocabSize, D)
      - ContextMLP: Linear(cw * D -> H) + Tanh
      - LMHead: Linear(H -> VocabSize)
    Loss computed on LAST next-token per sequence.
    """

    def __init__(self, vocab_size: int, cw: int, d: int, h: int):
        super().__init__()
        self.cw = cw
        self.d = d
        self.h = h
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d, padding_idx=0)
        self.context_mlp = nn.Sequential(
            nn.Linear(cw * d, h),
            nn.Tanh()
        )
        self.lm_head = nn.Linear(h, vocab_size)

    def forward(self, x):
        # x: (batch, cw)
        emb = self.embedding(x)             # (batch, cw, d)
        emb_flat = emb.view(x.size(0), -1)  # (batch, cw * d)
        hidden = self.context_mlp(emb_flat)  # (batch, h)
        logits = self.lm_head(hidden)        # (batch, vocab_size)
        return logits


# ============================================================================
# BINARY SERIALIZATION (Web-Compatible)
# ============================================================================

def save_model_bin(model: TinyLMContextMLP, output_path: str) -> List[Dict[str, Any]]:
    """
    Save model weights as raw float32 binary concatenated into one .bin
    Order: E, W1, b1, W2, b2
    Returns tensor metadata for manifest.
    """
    state = model.state_dict()

    tensor_specs = [
        ("E",  state["embedding.weight"],     [model.vocab_size, model.d]),
        ("W1", state["context_mlp.0.weight"], [model.cw * model.d, model.h]),
        ("b1", state["context_mlp.0.bias"],   [model.h]),
        ("W2", state["lm_head.weight"],       [model.h, model.vocab_size]),
        ("b2", state["lm_head.bias"],         [model.vocab_size]),
    ]

    tensors_meta = []
    with open(output_path, "wb") as f:
        for name, tensor, shape in tensor_specs:
            arr = tensor.detach().cpu().numpy().astype(np.float32)
            arr = np.ascontiguousarray(arr.flatten())
            f.write(arr.tobytes())
            tensors_meta.append({
                "name": name,
                "dtype": "float32",
                "shape": shape,
                "length": int(arr.size)
            })

    return tensors_meta


def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def build_manifest(
    model: TinyLMContextMLP,
    tensors_meta: List[Dict],
    training_stats: Dict,
    vocab_id_to_token: List[str],
    bin_path: str
) -> Dict[str, Any]:
    """Build complete manifest.json conforming to schema_version 3."""

    param_count = sum(p.numel() for p in model.parameters())
    param_bytes = param_count * 4  # float32

    checksum_val = compute_sha256(bin_path)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    manifest = {
        "format": "mini-lab-ai-model",
        "version": FORMAT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "model": {
            "name": MODEL_NAME,
            "file": f"{MODEL_NAME}.bin",
            "architecture": ARCHITECTURE,
            "input_shape": [model.cw, model.d],
            "output_shape": [model.vocab_size],
            "layers": [
                {"name": "embedding", "type": "Embedding", "output_shape": [model.cw, model.d]},
                {"name": "context_mlp", "type": "Dense+Tanh", "output_shape": [model.h]},
                {"name": "language_model_head", "type": "Dense+Softmax", "output_shape": [model.vocab_size]}
            ],
            "optimizer": "Adam",
            "loss": "cross-entropy",
            "metrics": [],
            "hyperparameters": {
                "cw": model.cw,
                "D": model.d,
                "H": model.h
            },
            "configuration": {
                "model_type": "pretrain",
                "cfg": {"cw": model.cw, "D": model.d, "H": model.h},
                "parameter_count": param_count,
                "parameter_bytes_float32": param_bytes
            },
            "tensors": tensors_meta,
            "tokenizer": {
                "type": "AXION AI-Lab regex tokenizer",
                "pattern": TOKENIZER_PATTERN,
                "special_tokens": SPECIAL_TOKENS
            },
            "preprocessing": {
                "text_source": training_stats.get("dataset", "Dataset gabungan (JSONL)"),
                "masking": "none",
                "content_tokenization": True,
                "captions_tokenization": False
            },
            "vocab": {
                "size": model.vocab_size,
                "id_to_token": vocab_id_to_token
            },
            "labels": []
        },
        "metadata": {
            "id": hashlib.md5(now_iso.encode()).hexdigest()[:10],
            "type": "pretrain",
            "parent_id": None,
            "stats": training_stats
        },
        "text_model": {
            "architecture": ARCHITECTURE,
            "cfg": {"cw": model.cw, "D": model.d, "H": model.h},
            "vocab_size": model.vocab_size,
            "tokenizer": {
                "type": "AXION AI-Lab regex tokenizer",
                "pattern": TOKENIZER_PATTERN,
                "special_tokens": SPECIAL_TOKENS
            },
            "binary_tensors": [t["name"] for t in tensors_meta]
        },
        "multimodal": {"enabled": False, "encoders": []},
        "training": {
            "last_loss": training_stats.get("loss", 0),
            "best_loss": training_stats.get("best_loss", 0),
            "epoch": training_stats.get("epoch", 0),
            "step": training_stats.get("steps", 0),
            "accuracy": 0,
            "val_loss": 0,
            "val_accuracy": 0,
            "learning_rate": training_stats.get("lr", 0),
            "batch_size": training_stats.get("batch_size", 0),
            "dataset": {
                "name": training_stats.get("dataset", ""),
                "tokens": training_stats.get("tokens", 0),
                "documents": training_stats.get("docs", 0)
            },
            "metrics": [],
            "stats": training_stats
        },
        "checksum": {
            "algorithm": "sha256",
            "value": checksum_val
        },
        "created_at": now_iso,
        "updated_at": now_iso,
        "app": {
            "name": "aplikasi-mini-lab-ai",
            "version": "2.0.0"
        }
    }
    return manifest


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Step 1: Build vocab from actual dataset
    vocab, id_to_token = build_vocab_from_data(args.data_dir, args.vocab_size)

    # Step 2: Load dataset
    dataset = JsonlTextDataset(args.data_dir, vocab, args.context_window)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True
    )

    # Step 3: Initialize model
    model = TinyLMContextMLP(
        vocab_size=args.vocab_size,
        cw=args.context_window,
        d=args.dim_d,
        h=args.dim_h
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore <pad>

    # Checkpoint dirs
    os.makedirs("checkpoints/final", exist_ok=True)

    global_step = 0
    running_loss = 0.0
    best_loss = float("inf")
    start_time = datetime.now(timezone.utc)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🚀 Training Config:")
    print(f"   Model: {MODEL_NAME} ({ARCHITECTURE})")
    print(f"   Parameters: {total_params:,}")
    print(f"   Vocab: {args.vocab_size} | CW: {args.context_window} | D: {args.dim_d} | H: {args.dim_h}")
    print(f"   Batch: {args.batch_size} | Max Steps: {args.max_steps}")
    print(f"   Checkpoint every: {args.checkpoint_every} batches\n")

    model.train()
    progress = tqdm(total=args.max_steps, desc="Training")

    while global_step < args.max_steps:
        for batch_x, batch_y in dataloader:
            if global_step >= args.max_steps:
                break

            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            # Forward: predict LAST next-token in context window
            logits = model(batch_x)       # (batch, vocab_size)
            target = batch_y[:, -1]       # (batch,) - last position target

            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}")

            # Save checkpoint every N batches
            if global_step % args.checkpoint_every == 0:
                avg_loss = running_loss / args.checkpoint_every
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

                ckpt_dir = f"checkpoints/step_{global_step}"
                os.makedirs(ckpt_dir, exist_ok=True)

                bin_path = os.path.join(ckpt_dir, f"{MODEL_NAME}.bin")
                tensors_meta = save_model_bin(model, bin_path)

                stats = {
                    "steps": global_step,
                    "loss": round(avg_loss, 6),
                    "best_loss": round(min(best_loss, avg_loss), 6),
                    "ema": round(avg_loss, 6),
                    "dataset": "Dataset gabungan (JSONL)",
                    "tokens": len(dataset.tokens),
                    "docs": dataset.doc_count,
                    "elapsed": round(elapsed, 2),
                    "lr": optimizer.param_groups[0]["lr"],
                    "batch_size": args.batch_size,
                    "epoch": 0
                }

                manifest = build_manifest(model, tensors_meta, stats, id_to_token, bin_path)
                manifest_path = os.path.join(ckpt_dir, "manifest.json")
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=1, ensure_ascii=False)

                # Track best and copy to final
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    import shutil
                    shutil.copytree(ckpt_dir, "checkpoints/final", dirs_exist_ok=True)
                    print(f"\n⭐ [BEST] Step {global_step} | Loss: {avg_loss:.4f} | Saved to final/")
                else:
                    print(f"\n💾 [CKPT] Step {global_step} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")

                running_loss = 0.0

    progress.close()

    # Ensure final checkpoint exists even if no improvement after last save
    final_bin = os.path.join("checkpoints/final", f"{MODEL_NAME}.bin")
    final_manifest = os.path.join("checkpoints/final", "manifest.json")

    if not os.path.exists(final_bin):
        print("\n⚠️  No best checkpoint saved. Saving current state as final...")
        tensors_meta = save_model_bin(model, final_bin)
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        remaining = global_step % args.checkpoint_every
        final_avg = running_loss / max(remaining, 1)

        stats = {
            "steps": global_step,
            "loss": round(final_avg, 6),
            "best_loss": round(best_loss, 6),
            "ema": round(final_avg, 6),
            "dataset": "Dataset gabungan (JSONL)",
            "tokens": len(dataset.tokens),
            "docs": dataset.doc_count,
            "elapsed": round(elapsed, 2),
            "lr": optimizer.param_groups[0]["lr"],
            "batch_size": args.batch_size,
            "epoch": 0
        }
        manifest = build_manifest(model, tensors_meta, stats, id_to_token, final_bin)
        with open(final_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✅ Training complete!")
    print(f"   Final model: checkpoints/final/{MODEL_NAME}.bin")
    print(f"   Manifest:    checkpoints/final/manifest.json")
    print(f"   Best loss:   {best_loss:.4f}")
    print(f"   Total steps: {global_step}")
    print(f"{'='*60}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyLM-ContextMLP Pretraining")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to extracted dataset directory")
    parser.add_argument("--context_window", type=int, default=16, help="Context window size (cw)")
    parser.add_argument("--batch_size", type=int, default=100, help="Training batch size")
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--vocab_size", type=int, default=5000, help="Vocabulary size")
    parser.add_argument("--dim_h", type=int, default=125, help="Hidden dimension (H)")
    parser.add_argument("--dim_d", type=int, default=125, help="Embedding dimension (D)")
    parser.add_argument("--max_output_tokens", type=int, default=2084, help="Max output tokens for generation")
    parser.add_argument("--checkpoint_every", type=int, default=10, help="Save checkpoint every N batches")

    args = parser.parse_args()
    train(args) 
