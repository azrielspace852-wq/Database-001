#!/usr/bin/env python3
"""
TinyLM-ContextMLP Pretraining Script
Architecture: Embedding -> ContextMLP (Dense+Tanh) -> LMHead (Dense+Softmax)
Target: Web-ready .bin format with manifest.json
"""

import os
import sys
import json
import struct
import hashlib
import argparse
import glob
from datetime import datetime, timezone
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

MODEL_NAME = "kicau1-nano-lite"
ARCHITECTURE = "TinyLM-ContextMLP"
SCHEMA_VERSION = 3
FORMAT_VERSION = "3.0.0"
TOKENIZER_PATTERN = r"\p{L}[\p{L}\p{M}]*|\p{N}+|[^\s]"
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


# ============================================================================
# DATASET
# ============================================================================

class JsonlTextDataset(Dataset):
    """Handles nested JSONL structure: {"text": {"lang": "", "category": "", "content": ""}}"""
    
    def __init__(self, data_dir: str, vocab: Dict[str, int], context_window: int):
        self.vocab = vocab
        self.context_window = context_window
        self.tokens: List[int] = []
        
        jsonl_files = sorted(glob.glob(os.path.join(data_dir, "**/*.jsonl"), recursive=True))
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files found in {data_dir}")
            
        print(f"Loading {len(jsonl_files)} JSONL files...")
        for fpath in tqdm(jsonl_files, desc="Parsing"):
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        # Handle nested structure safely
                        content = ""
                        if isinstance(obj.get("text"), dict):
                            content = obj["text"].get("content", "")
                        elif isinstance(obj.get("text"), str):
                            content = obj["text"]
                        
                        if content:
                            toks = self._tokenize(content)
                            self.tokens.extend(toks)
                            self.tokens.append(vocab["<eos>"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        if len(self.tokens) < context_window + 1:
            raise ValueError(f"Insufficient tokens ({len(self.tokens)}) for context_window {context_window}")
            
        print(f"Total tokens loaded: {len(self.tokens)}")

    def _tokenize(self, text: str) -> List[int]:
        """Simple whitespace + punctuation tokenizer matching AXION regex pattern."""
        import re
        pattern = r"\w+|[^\s\w]"
        raw_tokens = re.findall(pattern, text.lower())
        return [self.vocab.get(t, self.vocab["<unk>"]) for t in raw_tokens]

    def __len__(self):
        return max(0, len(self.tokens) - self.context_window)

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.context_window]
        y = self.tokens[idx + 1 : idx + self.context_window + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class TinyLMContextMLP(nn.Module):
    """
    Architecture:
      - Embedding(VocabSize, D)
      - ContextMLP: Linear(cw * D -> H) + Tanh
      - LMHead: Linear(H -> VocabSize) + Softmax (applied in loss)
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
        emb = self.embedding(x)           # (batch, cw, d)
        emb_flat = emb.view(x.size(0), -1) # (batch, cw * d)
        hidden = self.context_mlp(emb_flat) # (batch, h)
        logits = self.lm_head(hidden)      # (batch, vocab_size)
        return logits


# ============================================================================
# BINARY SERIALIZATION (Web-Compatible)
# ============================================================================

def save_model_bin(model: TinyLMContextMLP, output_path: str) -> Dict[str, Any]:
    """
    Save model weights as raw float32 binary tensor files concatenated into one .bin
    Returns tensor metadata for manifest.
    Order: E, W1, b1, W2, b2
    """
    state = model.state_dict()
    
    tensor_specs = [
        ("E",  state["embedding.weight"],   [model.vocab_size, model.d]),
        ("W1", state["context_mlp.0.weight"], [model.cw * model.d, model.h]),
        ("b1", state["context_mlp.0.bias"],   [model.h]),
        ("W2", state["lm_head.weight"],       [model.h, model.vocab_size]),
        ("b2", state["lm_head.bias"],         [model.vocab_size]),
    ]
    
    tensors_meta = []
    with open(output_path, "wb") as f:
        for name, tensor, shape in tensor_specs:
            arr = tensor.detach().cpu().numpy().astype(np.float32)
            # Ensure contiguous C-order
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
                "text_source": "Dataset gabungan (JSONL)",
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
            "best_loss": training_stats.get("loss", 0),
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
    print(f"Using device: {device}")
    
    # Build simple vocab from special tokens + placeholder
    # In production, build from corpus. Here we create a fixed-size vocab.
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    # Fill remaining vocab slots with token placeholders
    for i in range(len(SPECIAL_TOKENS), args.vocab_size):
        vocab[f"<tok_{i}>"] = i
    id_to_token = [""] * args.vocab_size
    for tok, idx in vocab.items():
        id_to_token[idx] = tok
    
    # Load dataset
    dataset = JsonlTextDataset(args.data_dir, vocab, args.context_window)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=2
    )
    
    # Initialize model
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
    
    print(f"Starting training: max_steps={args.max_steps}, batch={args.batch_size}, cw={args.context_window}")
    
    model.train()
    progress = tqdm(total=args.max_steps, desc="Training")
    
    while global_step < args.max_steps:
        for batch_x, batch_y in dataloader:
            if global_step >= args.max_steps:
                break
                
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward: logits are (batch, vocab_size), targets need flattening
            # Since ContextMLP produces ONE prediction per sequence (not per token),
            # we predict the LAST next-token in the context window
            logits = model(batch_x)  # (batch, vocab_size)
            target = batch_y[:, -1]   # (batch,) - last position target
            
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
                ckpt_dir = f"checkpoints/step_{global_step}"
                os.makedirs(ckpt_dir, exist_ok=True)
                
                tensors_meta = save_model_bin(model, os.path.join(ckpt_dir, f"{MODEL_NAME}.bin"))
                
                elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                stats = {
                    "steps": global_step,
                    "loss": round(avg_loss, 6),
                    "ema": round(avg_loss, 6),
                    "dataset": "Dataset gabungan (JSONL)",
                    "tokens": len(dataset.tokens),
                    "docs": 0,  # Will be updated below
                    "elapsed": round(elapsed, 2),
                    "lr": optimizer.param_groups[0]["lr"],
                    "batch_size": args.batch_size,
                    "epoch": 0
                }
                
                manifest = build_manifest(model, tensors_meta, stats, id_to_token,
                                          os.path.join(ckpt_dir, f"{MODEL_NAME}.bin"))
                with open(os.path.join(ckpt_dir, "manifest.json"), "w") as f:
                    json.dump(manifest, f, indent=1)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    # Copy best to final
                    import shutil
                    shutil.copytree(ckpt_dir, "checkpoints/final", dirs_exist_ok=True)
                
                running_loss = 0.0
                print(f"\n[Checkpoint] Step {global_step} | Loss: {avg_loss:.4f} | Saved to {ckpt_dir}")
    
    progress.close()
    
    # Ensure final checkpoint exists
    final_bin = "checkpoints/final/model.bin"
    final_manifest = "checkpoints/final/manifest.json"
    
    if not os.path.exists(final_bin):
        tensors_meta = save_model_bin(model, final_bin)
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        stats = {
            "steps": global_step,
            "loss": round(running_loss / max(global_step % args.checkpoint_every, 1), 6),
            "ema": round(best_loss, 6),
            "dataset": "Dataset gabungan (JSONL)",
            "tokens": len(dataset.tokens),
            "docs": 0,
            "elapsed": round(elapsed, 2),
            "lr": optimizer.param_groups[0]["lr"],
            "batch_size": args.batch_size,
            "epoch": 0
        }
        manifest = build_manifest(model, tensors_meta, stats, id_to_token, final_bin)
        with open(final_manifest, "w") as f:
            json.dump(manifest, f, indent=1)
    
    print(f"\n✅ Training complete. Final model saved to checkpoints/final/")
    print(f"   Binary: {final_bin}")
    print(f"   Manifest: {final_manifest}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyLM-ContextMLP Pretraining")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--context_window", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--vocab_size", type=int, default=5000)
    parser.add_argument("--dim_h", type=int, default=125)
    parser.add_argument("--dim_d", type=int, default=156)
    parser.add_argument("--max_output_tokens", type=int, default=2084)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    
    args = parser.parse_args()
    train(args)
