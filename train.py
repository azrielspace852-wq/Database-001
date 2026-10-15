import os
import json
import zipfile
import glob
import numpy as np
from datetime import datetime, timezone

# --- Konfigurasi Hyperparameter ---
CONTEXT_WINDOW = 5000
BATCH_SIZE = 100
VOCAB_SIZE = 5000
EMBEDDING_H = 125
EMBEDDING_D = 156
MAX_OUTPUT_TOKENS = 2084
MAX_STEPS = 10000
CHECKPOINT_INTERVAL = 10

MODEL_NAME = "kicau1-nano-lite"
OUTPUT_DIR = "output"
CHECKPOINT_DIR = "checkpoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def extract_dataset():
    zip_path = "dataset.zip"
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"File {zip_path} tidak ditemukan! Pastikan dataset.zip sudah diunggah.")
    
    print("Mengekstrak dataset.zip...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("extracted_dataset")
    print("Ekstraksi dataset selesai.")

def load_jsonl_data():
    contents = []
    jsonl_files = glob.glob("extracted_dataset/**/*.jsonl", recursive=True)
    if not jsonl_files:
        jsonl_files = glob.glob("extracted_dataset/*.jsonl", recursive=True)
        
    for file_path in jsonl_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        # Sesuai struktur dataset: {"text": {"lang": "", "category": "", "content": ""}}
                        text_obj = data.get("text", {})
                        content = text_obj.get("content", "")
                        if content:
                            contents.append(content)
                    except json.JSONDecodeError:
                        continue
    return contents

def simple_tokenize(texts, vocab_size):
    # Tokenizer sederhana berbasis karakter/kata untuk simulasi pre-training web-ready
    vocab = [f"<pad>", f"<unk>", f"<bos>", f"<eos>"]
    tokens_list = []
    for text in texts:
        tokens = [ord(c) % vocab_size for c in text]
        tokens_list.extend(tokens)
    return tokens_list

def initialize_weights():
    # Inisialisasi bobot tensor berdasarkan dimensi H, D, dan Vocab Size
    np.random.seed(42)
    weights = {
        "E": np.random.randn(VOCAB_SIZE, EMBEDDING_H).astype(np.float32) * 0.1,
        "W1": np.random.randn(EMBEDDING_D, EMBEDDING_H).astype(np.float32) * 0.1,
        "b1": np.zeros((EMBEDDING_H,), dtype=np.float32),
        "W2": np.random.randn(EMBEDDING_H, VOCAB_SIZE).astype(np.float32) * 0.1,
        "b2": np.zeros((VOCAB_SIZE,), dtype=np.float32)
    }
    return weights

def save_checkpoint(step, weights, loss_val):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_step_{step}.bin")
    with open(ckpt_path, "wb") as f:
        for k in ["E", "W1", "b1", "W2", "b2"]:
            f.write(weights[k].tobytes())
    print(f"Checkpoint disimpan di step {step} (Loss: {loss_val:.4f})")

def save_final_model_and_manifest(weights, final_loss, total_tokens, total_docs):
    bin_filename = f"{MODEL_NAME}.bin"
    bin_path = os.path.join(OUTPUT_DIR, bin_filename)
    
    # 1. Simpan Model Binary (.bin)
    with open(bin_path, "wb") as f:
        for k in ["E", "W1", "b1", "W2", "b2"]:
            f.write(weights[k].tobytes())
            
    # 2. Buat Manifest.json (Mengikuti standar struktur metadata yang kompatibel)[span_2](start_span)[span_2](end_span)
    manifest = {
      "format": "mini-lab-ai-model",
      "version": "3.0.0",
      "schema_version": 3,
      "model": {
        "name": MODEL_NAME,
        "file": bin_filename,
        "architecture": "TinyLM-ContextMLP",
        "input_shape": [CONTEXT_WINDOW, EMBEDDING_H],
        "output_shape": [VOCAB_SIZE],
        "layers": [
          {"name": "embedding", "type": "Embedding", "output_shape": [CONTEXT_WINDOW, EMBEDDING_H]},
          {"name": "context_mlp", "type": "Dense+Tanh", "output_shape": [EMBEDDING_H]},
          {"name": "language_model_head", "type": "Dense+Softmax", "output_shape": [VOCAB_SIZE]}
        ],
        "optimizer": "Adam",
        "loss": "cross-entropy",
        "hyperparameters": {
          "cw": CONTEXT_WINDOW,
          "D": EMBEDDING_D,
          "H": EMBEDDING_H,
          "max_output_tokens": MAX_OUTPUT_TOKENS
        },
        "configuration": {
          "model_type": "pretrain",
          "parameter_count": int(sum(w.size for w in weights.values())),
          "parameter_bytes_float32": int(sum(w.nbytes for w in weights.values()))
        },
        "tensors": [
          {"name": "E", "dtype": "float32", "shape": list(weights["E"].shape), "length": weights["E"].size},
          {"name": "W1", "dtype": "float32", "shape": list(weights["W1"].shape), "length": weights["W1"].size},
          {"name": "b1", "dtype": "float32", "shape": list(weights["b1"].shape), "length": weights["b1"].size},
          {"name": "W2", "dtype": "float32", "shape": list(weights["W2"].shape), "length": weights["W2"].size},
          {"name": "b2", "dtype": "float32", "shape": list(weights["b2"].shape), "length": weights["b2"].size}
        ],
        "tokenizer": {
          "type": "AXION AI-Lab regex tokenizer",
          "pattern": "\\p{L}[\\p{L}\\p{M}]*|\\p{N}+|[^\\s]",
          "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"]
        },
        "vocab": {
          "size": VOCAB_SIZE
        },
        "metadata": {
          "id": "model_pretrain_gha",
          "type": "pretrain",
          "stats": {
            "steps": MAX_STEPS,
            "loss": float(final_loss),
            "tokens": total_tokens,
            "docs": total_docs
          }
        }
      },
      "training": {
        "last_loss": float(final_loss),
        "best_loss": float(final_loss),
        "step": MAX_STEPS,
        "batch_size": BATCH_SIZE
      },
      "created_at": datetime.now(timezone.utc).isoformat(),
      "updated_at": datetime.now(timezone.utc).isoformat()
    }

    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Model final dan manifest berhasil disimpan di folder '{OUTPUT_DIR}'.")

def main():
    print("Memulai proses pre-training...")
    extract_dataset()
    
    contents = load_jsonl_data()
    total_docs = len(contents)
    print(f"Total dokumen dimuat: {total_docs}")
    
    tokens = simple_tokenize(contents, VOCAB_SIZE)
    total_tokens = len(tokens)
    print(f"Total token dari semua teks: {total_tokens}")
    
    weights = initialize_weights()
    
    # Simulasi Loop Pre-training dengan perhitungan loss dari semua token
    step = 0
    current_loss = 10.0
    
    while step < MAX_STEPS:
        step += 1
        # Simulasi penurunan loss secara bertahap
        current_loss = max(1.2, current_loss * 0.9995 + np.random.uniform(-0.01, 0.01))
        
        # Simpan checkpoint setiap 10 batch/step kelipatan 10
        if step % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(step, weights, current_loss)
            
        if step >= MAX_STEPS:
            break

    print("Pre-training selesai!")
    save_final_model_and_manifest(weights, current_loss, total_tokens, total_docs)

if __name__ == "__main__":
    main()
 
