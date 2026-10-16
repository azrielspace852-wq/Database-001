#!/usr/bin/env python3
"""
TinyLM-Transformer Pretraining Script (Updated)
- Builds vocabulary from actual dataset content
- Handles 4 JSONL structures:
    1) {"prompt": "", "response": ""}
    2) {"id": "", "kategori": "", "tingkat_kesulitan": number,
        "messages": [{"role": "", "content": ""}]}
    3) {"text": ""}
    4) {"text": {"lang": "", "category": "", "content": ""}}   <-- struktur lama
- Produces web-compatible .bin + manifest.json
- Checkpointing every N batches
"""

import os
import sys
import math
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
# CONFIGURATION & CONSTANTS
# ============================================================================

TOKENIZER_PATTERN = r"\w+|[^\s\w]"
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
MODEL_NAME     = "tinylm-transformer"
ARCHITECTURE   = "TinyLMTransformer"
FORMAT_VERSION = "2.0.0"
SCHEMA_VERSION = 4


# ============================================================================
# JSONL STRUCTURE HANDLER
# ============================================================================

FIELD_SEP = "\n"
MSG_SEP   = "\n"
ROLE_SEP  = ": "


def extract_content(obj: Any) -> str:
    if not isinstance(obj, dict):
        return ""

    # Struktur 1: prompt / response
    if "prompt" in obj or "response" in obj:
        parts: List[str] = []
        prompt = obj.get("prompt")
        response = obj.get("response")
        if isinstance(prompt, str) and prompt.strip():
            parts.append(prompt.strip())
        if isinstance(response, str) and response.strip():
            parts.append(response.strip())
        if parts:
            return FIELD_SEP.join(parts)

    # Struktur 2: messages[{role, content}]
    if "messages" in obj and isinstance(obj["messages"], list):
        parts: List[str] = []
        for msg in obj["messages"]:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")
            role = msg.get("role", "")
            if not isinstance(content, str) or not content.strip():
                continue
            if isinstance(role, str) and role.strip():
                parts.append(f"{role.strip()}{ROLE_SEP}{content.strip()}")
            else:
                parts.append(content.strip())
        if parts:
            return MSG_SEP.join(parts)

    # Struktur 4 & 3: field "text"
    text = obj.get("text")
    if isinstance(text, dict):
        content = text.get("content", "")
        if isinstance(content, str):
            return content.strip()
        return ""
    if isinstance(text, str):
        return text.strip()

    return ""


def iter_jsonl_files(data_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "**/*.jsonl"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {data_dir}")
    return files


# ============================================================================
# VOCABULARY BUILDER
# ============================================================================

def build_vocab_from_data(data_dir: str, vocab_size: int) -> Tuple[Dict[str, int], List[str]]:
    counter = Counter()
    jsonl_files = iter_jsonl_files(data_dir)

    print(f"📚 Building vocab from {len(jsonl_files)} files...")
    pattern = re.compile(TOKENIZER_PATTERN, re.UNICODE)

    for fpath in tqdm(jsonl_files, desc="Scanning tokens"):
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = extract_content(obj)
                if content:
                    tokens = pattern.findall(content.lower())
                    counter.update(tokens)

    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    most_common = counter.most_common(vocab_size - len(SPECIAL_TOKENS))

    for tok, _ in most_common:
        if tok not in vocab:
            vocab[tok] = len(vocab)
        if len(vocab) >= vocab_size:
            break

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
    def __init__(self, data_dir: str, vocab: Dict[str, int], context_window: int):
        self.vocab = vocab
        self.context_window = context_window
        self.tokens: List[int] = []
        self.pattern = re.compile(TOKENIZER_PATTERN, re.UNICODE)

        jsonl_files = iter_jsonl_files(data_dir)

        print(f"📖 Loading {len(jsonl_files)} JSONL files...")
        doc_count = 0
        skipped = 0

        for fpath in tqdm(jsonl_files, desc="Parsing"):
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue

                    content = extract_content(obj)
                    if not content:
                        skipped += 1
                        continue

                    toks = self._tokenize(content)
                    if toks:
                        self.tokens.extend(toks)
                        self.tokens.append(vocab["<eos>"])
                        doc_count += 1

        if len(self.tokens) < context_window + 1:
            raise ValueError(
                f"Insufficient tokens ({len(self.tokens)}) for context_window {context_window}. "
                f"Need at least {context_window + 1}."
            )

        self.doc_count = doc_count
        print(f"✅ Total tokens: {len(self.tokens)} | Documents: {doc_count} | Skipped lines: {skipped}")

    def _tokenize(self, text: str) -> List[int]:
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
# MODEL DEFINITION — CAUSAL TRANSFORMER
# ============================================================================

class TinyLMTransformer(nn.Module):
    """
    Causal transformer mini untuk language modeling.
    Output: (B, T, vocab_size) — logits per-posisi.
    """

    def __init__(
        self,
        vocab_size: int,
        cw: int,
        d: int,
        h: int,
        n_layer: int = 4,
        n_head: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d % n_head == 0, f"d ({d}) harus habis dibagi n_head ({n_head})"

        self.cw = cw
        self.d = d
        self.h = h
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head

        self.tok_emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pos_emb = nn.Embedding(cw, d)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_head,
            dim_feedforward=h,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)

        # Causal mask (upper triangular = True artinya di-mask)
        self.register_buffer(
            "_causal_mask",
            torch.triu(torch.ones(cw, cw, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

        # Init
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.drop(h)

        mask = self._causal_mask[:T, :T]
        h = self.encoder(h, mask=mask)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits  # (B, T, vocab_size)


# ============================================================================
# BINARY SERIALIZATION
# ============================================================================

def save_model_bin(model: nn.Module, output_path: str) -> List[Dict[str, Any]]:
    """Save semua parameter model ke .bin (float32, row-major flattened)."""
    state = model.state_dict()
    tensors_meta = []

    with open(output_path, "wb") as f:
        for name, tensor in state.items():
            if not torch.is_floating_point(tensor):
                tensor = tensor.float()
            arr = tensor.detach().cpu().numpy().astype(np.float32)
            shape = list(arr.shape)
            arr = np.ascontiguousarray(arr.flatten())
            f.write(arr.tobytes())
            tensors_meta.append({
                "name": name,
                "dtype": "float32",
                "shape": shape,
                "length": int(arr.size),
            })
    return tensors_meta


def compute_sha256(filepath: str) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def build_manifest(
    model: TinyLMTransformer,
    tensors_meta: List[Dict],
    training_stats: Dict,
    vocab_id_to_token: List[str],
    bin_path: str,
) -> Dict[str, Any]:
    param_count = sum(p.numel() for p in model.parameters())
    param_bytes = param_count * 4

    checksum_val = compute_sha256(bin_path)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    cfg = {
        "cw": model.cw,
        "D": model.d,
        "H": model.h,
        "n_layer": model.n_layer,
        "n_head": model.n_head,
        "vocab_size": model.vocab_size,
    }

    manifest = {
        "format": "mini-lab-ai-model",
        "version": FORMAT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "model": {
            "name": MODEL_NAME,
            "file": f"{MODEL_NAME}.bin",
            "architecture": ARCHITECTURE,
            "input_shape": [model.cw],
            "output_shape": [model.cw, model.vocab_size],
            "layers": [
                {"name": "token_embedding", "type": "Embedding", "output_shape": [model.cw, model.d]},
                {"name": "position_embedding", "type": "Embedding", "output_shape": [model.cw, model.d]},
                {"name": "transformer_encoder", "type": f"TransformerEncoder(x{model.n_layer})",
                 "output_shape": [model.cw, model.d]},
                {"name": "final_norm", "type": "LayerNorm", "output_shape": [model.cw, model.d]},
                {"name": "lm_head", "type": "Dense+Softmax", "output_shape": [model.cw, model.vocab_size]},
            ],
            "optimizer": "AdamW",
            "loss": "cross-entropy (full-sequence)",
            "metrics": [],
            "hyperparameters": cfg,
            "configuration": {
                "model_type": "pretrain",
                "cfg": cfg,
                "parameter_count": param_count,
                "parameter_bytes_float32": param_bytes,
            },
            "tensors": tensors_meta,
            "tokenizer": {
                "type": "AXION AI-Lab regex tokenizer",
                "pattern": TOKENIZER_PATTERN,
                "special_tokens": SPECIAL_TOKENS,
            },
            "preprocessing": {
                "text_source": training_stats.get("dataset", "Dataset gabungan (JSONL)"),
                "masking": "causal",
                "content_tokenization": True,
                "captions_tokenization": False,
                "supported_jsonl_structures": [
                    "prompt_response",
                    "id_kategori_messages",
                    "text_string",
                    "text_object_content",
                ],
            },
            "vocab": {
                "size": model.vocab_size,
                "id_to_token": vocab_id_to_token,
            },
            "labels": [],
        },
        "metadata": {
            "id": hashlib.md5(now_iso.encode()).hexdigest()[:10],
            "type": "pretrain",
            "parent_id": None,
            "stats": training_stats,
        },
        "text_model": {
            "architecture": ARCHITECTURE,
            "cfg": cfg,
            "vocab_size": model.vocab_size,
            "tokenizer": {
                "type": "AXION AI-Lab regex tokenizer",
                "pattern": TOKENIZER_PATTERN,
                "special_tokens": SPECIAL_TOKENS,
            },
            "binary_tensors": [t["name"] for t in tensors_meta],
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
                "documents": training_stats.get("docs", 0),
            },
            "metrics": [],
            "stats": training_stats,
        },
        "checksum": {
            "algorithm": "sha256",
            "value": checksum_val,
        },
        "created_at": now_iso,
        "updated_at": now_iso,
        "app": {
            "name": "aplikasi-mini-lab-ai",
            "version": "2.0.0",
        },
    }
    return manifest


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    vocab, id_to_token = build_vocab_from_data(args.data_dir, args.vocab_size)

    dataset = JsonlTextDataset(args.data_dir, vocab, args.context_window)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = TinyLMTransformer(
        vocab_size=args.vocab_size,
        cw=args.context_window,
        d=args.dim_d,
        h=args.dim_h,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
    ).to(device)

    # ---- Optimizer + Scheduler ----
    decay_params = []
    no_decay_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    warmup = args.warmup_steps

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, args.max_steps - warmup)
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Full-sequence loss; ignore <pad> (id 0)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    os.makedirs("checkpoints/final", exist_ok=True)

    global_step = 0
    micro_step = 0
    running_loss = 0.0
    running_count = 0
    best_loss = float("inf")
    start_time = datetime.now(timezone.utc)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🚀 Training Config:")
    print(f"   Model: {MODEL_NAME} ({ARCHITECTURE})")
    print(f"   Parameters: {total_params:,}")
    print(f"   Vocab: {args.vocab_size} | CW: {args.context_window} | D: {args.dim_d} | H: {args.dim_h}")
    print(f"   n_layer: {args.n_layer} | n_head: {args.n_head} | dropout: {args.dropout}")
    print(f"   Batch: {args.batch_size} | GradAccum: {args.grad_accum} | EffBatch: {args.batch_size * args.grad_accum}")
    print(f"   LR: {args.lr} | Warmup: {args.warmup_steps} | MinLR ratio: {args.min_lr_ratio}")
    print(f"   Max Steps: {args.max_steps} | Checkpoint every: {args.checkpoint_every} batches\n")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=args.max_steps, desc="Training")

    epoch = 0
    stop = False
    while not stop:
        epoch += 1
        for batch_x, batch_y in dataloader:
            if global_step >= args.max_steps:
                stop = True
                break

            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            # ---- Forward ----
            logits = model(batch_x)  # (B, T, V)

            # Full-sequence loss (predict t+1 from t for semua posisi)
            loss = criterion(
                logits.reshape(-1, args.vocab_size),
                batch_y.reshape(-1),
            )

            # Scale buat grad accumulation
            loss_scaled = loss / args.grad_accum
            loss_scaled.backward()

            micro_step += 1

            if micro_step % args.grad_accum == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                running_loss += loss.item()
                running_count += 1

                progress.update(1)
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    ppl=f"{math.exp(min(loss.item(), 20)):.0f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

                # ---- Checkpoint ----
                if global_step % args.checkpoint_every == 0:
                    avg_loss = running_loss / max(1, running_count)
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
                        "ppl": round(math.exp(min(avg_loss, 20)), 2),
                        "dataset": "Dataset gabungan (JSONL, 4 struktur)",
                        "tokens": len(dataset.tokens),
                        "docs": dataset.doc_count,
                        "elapsed": round(elapsed, 2),
                        "lr": scheduler.get_last_lr()[0],
                        "batch_size": args.batch_size,
                        "grad_accum": args.grad_accum,
                        "epoch": epoch,
                    }

                    manifest = build_manifest(model, tensors_meta, stats, id_to_token, bin_path)
                    manifest_path = os.path.join(ckpt_dir, "manifest.json")
                    with open(manifest_path, "w", encoding="utf-8") as f:
                        json.dump(manifest, f, indent=1, ensure_ascii=False)

                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        import shutil
                        shutil.copytree(ckpt_dir, "checkpoints/final", dirs_exist_ok=True)
                        print(f"\n⭐ [BEST] Step {global_step} | Loss: {avg_loss:.4f} | PPL: {math.exp(min(avg_loss,20)):.1f}")
                    else:
                        print(f"\n💾 [CKPT] Step {global_step} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")

                    running_loss = 0.0
                    running_count = 0

    progress.close()

    final_bin = os.path.join("checkpoints/final", f"{MODEL_NAME}.bin")
    final_manifest = os.path.join("checkpoints/final", "manifest.json")

    if not os.path.exists(final_bin):
        print("\n⚠️  No best checkpoint saved. Saving current state as final...")
        tensors_meta = save_model_bin(model, final_bin)
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        final_avg = running_loss / max(1, running_count)

        stats = {
            "steps": global_step,
            "loss": round(final_avg, 6),
            "best_loss": round(best_loss, 6),
            "ema": round(final_avg, 6),
            "ppl": round(math.exp(min(final_avg, 20)), 2),
            "dataset": "Dataset gabungan (JSONL, 4 struktur)",
            "tokens": len(dataset.tokens),
            "docs": dataset.doc_count,
            "elapsed": round(elapsed, 2),
            "lr": scheduler.get_last_lr()[0],
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "epoch": epoch,
        }
        manifest = build_manifest(model, tensors_meta, stats, id_to_token, final_bin)
        with open(final_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✅ Training complete!")
    print(f"   Final model: checkpoints/final/{MODEL_NAME}.bin")
    print(f"   Manifest:    checkpoints/final/manifest.json")
    print(f"   Best loss:   {best_loss:.4f}")
    print(f"   Best PPL:    {math.exp(min(best_loss, 20)):.1f}")
    print(f"   Total steps: {global_step}")
    print(f"{'='*60}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyLM-Transformer Pretraining")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to extracted dataset directory")
    parser.add_argument("--context_window", type=int, default=256, help="Context window size (cw)")
    parser.add_argument("--batch_size", type=int, default=16, help="Micro-batch size")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=20000, help="Maximum optimizer steps")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--dim_h", type=int, default=1024, help="Feedforward hidden dim (H)")
    parser.add_argument("--dim_d", type=int, default=256, help="Embedding dim (D)")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Minimum LR as ratio of peak")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clip norm")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--max_output_tokens", type=int, default=2084, help="Max output tokens for generation")
    parser.add_argument("--checkpoint_every", type=int, default=100, help="Save checkpoint every N optimizer steps")

    args = parser.parse_args()

    # Sanity check
    assert args.dim_d % args.n_head == 0, f"dim_d ({args.dim_d}) harus habis dibagi n_head ({args.n_head})"

    train(args)
