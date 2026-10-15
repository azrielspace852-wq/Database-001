#!/usr/bin/env python3
"""
TinyLM-ContextMLP Inference Runner (Updated)
- Handles nested ZIP extraction
- Reads tokenizer pattern from manifest (always in sync with train.py)
- Pure NumPy inference (no PyTorch needed)
- Supports local ZIP path OR URL (placeholder URL for now)

Usage:
    python inference.py <artifact_path_or_url> <chat.txt> <response.txt>
"""

import os
import sys
import json
import zipfile
import re
import glob
import urllib.request
import numpy as np


# ============================================================================
# PLACEHOLDER — ganti dengan URL artifact asli setelah training selesai
# ============================================================================
ARTIFACT_URL_PLACEHOLDER = "https://github.com/USER/REPO/actions/runs/XXX/artifacts/YYY"

# Harus sama dengan train.py
DEFAULT_TOKENIZER_PATTERN = r"\w+|[^\s\w]"
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


# ============================================================================
# ARTIFACT HANDLING
# ============================================================================

def resolve_artifact(path_or_url, download_dir="artifact_download"):
    """Kalau input berupa URL, download. Kalau file lokal, pakai langsung."""
    if path_or_url.startswith(("http://", "https://")):
        os.makedirs(download_dir, exist_ok=True)
        # Ambil nama file dari URL, fallback ke artifact.zip
        base = os.path.basename(path_or_url.split("?")[0]) or "artifact.zip"
        if not base.endswith(".zip"):
            base = "artifact.zip"
        local_path = os.path.join(download_dir, base)

        print(f"⬇️  Downloading artifact...")
        print(f"   URL: {path_or_url[:100]}{'...' if len(path_or_url) > 100 else ''}")
        try:
            urllib.request.urlretrieve(path_or_url, local_path)
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print(f"   (Kalau URL masih placeholder, ganti dulu di konstanta ARTIFACT_URL_PLACEHOLDER)")
            sys.exit(1)

        size = os.path.getsize(local_path)
        print(f"✅ Downloaded: {local_path} ({size:,} bytes)")
        return local_path

    if not os.path.exists(path_or_url):
        print(f"❌ Artifact not found: {path_or_url}")
        sys.exit(1)
    return path_or_url


def extract_nested_zip(zip_path, extract_dir):
    """Recursively extract ZIP files, handling nested ZIPs."""
    if os.path.exists(extract_dir):
        import shutil
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
    print(f"✅ Extracted: {zip_path} -> {extract_dir}")

    # Extract nested ZIPs (GitHub Actions artifact sering nested)
    while True:
        nested_zips = glob.glob(os.path.join(extract_dir, "**/*.zip"), recursive=True)
        if not nested_zips:
            break
        for nested in nested_zips:
            nested_dir = os.path.splitext(nested)[0]
            print(f"📦 Found nested ZIP: {nested}")
            with zipfile.ZipFile(nested, 'r') as z:
                z.extractall(nested_dir)
            os.remove(nested)
            print(f"✅ Extracted nested: {nested} -> {nested_dir}")

    return extract_dir


def find_model_files(base_dir):
    """Find model.bin and manifest.json recursively, prefer files in same dir."""
    bin_files = glob.glob(os.path.join(base_dir, "**/*.bin"), recursive=True)
    manifest_files = glob.glob(os.path.join(base_dir, "**/manifest.json"), recursive=True)

    if not bin_files:
        raise FileNotFoundError(f"No .bin file found in {base_dir}")
    if not manifest_files:
        raise FileNotFoundError(f"No manifest.json found in {base_dir}")

    # Prefer same directory
    for mf in manifest_files:
        mf_dir = os.path.dirname(mf)
        for bf in bin_files:
            if os.path.dirname(bf) == mf_dir:
                return bf, mf

    return bin_files[0], manifest_files[0]


# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class TinyLMInference:
    """Pure NumPy inference engine for TinyLM-ContextMLP."""

    def __init__(self, bin_path, manifest_path, verbose=True):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)

        # ---- Config ----
        model_section = self.manifest.get('model', {})
        cfg = model_section.get('hyperparameters', {}) \
              or self.manifest.get('text_model', {}).get('cfg', {})

        self.cw = int(cfg.get('cw', 16))
        self.D = int(cfg.get('D', 125))
        self.H = int(cfg.get('H', 125))

        vocab_section = model_section.get('vocab', {})
        self.vocab_size = int(vocab_section.get('size', 5000))

        # ---- Tokenizer pattern: baca dari manifest biar sinkron dengan training ----
        tok_cfg = model_section.get('tokenizer', {}) \
                  or self.manifest.get('text_model', {}).get('tokenizer', {})
        self.pattern = tok_cfg.get('pattern', DEFAULT_TOKENIZER_PATTERN)
        self.special_tokens = tok_cfg.get('special_tokens', SPECIAL_TOKENS)
        if verbose:
            print(f"🔤 Tokenizer pattern: {self.pattern}")
            print(f"🔤 Special tokens:   {self.special_tokens}")

        # ---- Vocab ----
        id_to_token = vocab_section.get('id_to_token', [])
        if not id_to_token:
            raise ValueError("manifest.json tidak punya vocab.id_to_token")
        self.vocab = [t if isinstance(t, str) else '<unk>' for t in id_to_token]
        # Pastikan index 0..len-1 ada; aman karena train.py isi semua dengan '<unk>' dulu
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.vocab)}

        # Special token ids
        self.pad_id = self.token_to_id.get('<pad>', 0)
        self.unk_id = self.token_to_id.get('<unk>', 1)
        self.bos_id = self.token_to_id.get('<bos>', 2)
        self.eos_id = self.token_to_id.get('<eos>', 3)

        # ---- Load tensors ----
        tensor_specs = model_section.get('tensors', [])
        if not tensor_specs:
            raise ValueError("manifest.json tidak punya model.tensors")

        with open(bin_path, 'rb') as f:
            raw = f.read()

        self.tensors = {}
        offset = 0
        for spec in tensor_specs:
            name = spec['name']
            length = int(spec['length'])
            arr = np.frombuffer(raw, dtype=np.float32, count=length, offset=offset).copy()
            shape = spec.get('shape')
            if shape:
                arr = arr.reshape(shape)
            self.tensors[name] = arr
            offset += length * 4

        expected = {'E', 'W1', 'b1', 'W2', 'b2'}
        missing = expected - set(self.tensors.keys())
        if missing:
            raise ValueError(f"Tensor tidak lengkap di .bin: missing {missing}")

        # Validasi shape
        assert self.tensors['E'].shape == (self.vocab_size, self.D), \
            f"E shape mismatch: {self.tensors['E'].shape}"
        assert self.tensors['W1'].shape == (self.cw * self.D, self.H), \
            f"W1 shape mismatch: {self.tensors['W1'].shape}"
        assert self.tensors['b1'].shape == (self.H,), \
            f"b1 shape mismatch: {self.tensors['b1'].shape}"
        assert self.tensors['W2'].shape == (self.H, self.vocab_size), \
            f"W2 shape mismatch: {self.tensors['W2'].shape}"
        assert self.tensors['b2'].shape == (self.vocab_size,), \
            f"b2 shape mismatch: {self.tensors['b2'].shape}"

        if verbose:
            n_params = sum(t.size for t in self.tensors.values())
            print(f"🧠 Model loaded:")
            print(f"   cw={self.cw} | D={self.D} | H={self.H} | vocab={self.vocab_size}")
            print(f"   Params: {n_params:,} (~{n_params * 4 / 1e6:.2f} MB float32)")
            print(f"   Tensors: {sorted(self.tensors.keys())}")

    # ------------------------------------------------------------------
    def tokenize(self, text):
        """Tokenize persis seperti train.py: lowercase dulu, pattern dari manifest."""
        raw_tokens = re.findall(self.pattern, text.lower(), re.UNICODE)
        return [self.token_to_id.get(t, self.unk_id) for t in raw_tokens]

    def detokenize(self, ids):
        return [self.vocab[i] if 0 <= i < len(self.vocab) else '<unk>' for i in ids]

    # ------------------------------------------------------------------
    def predict(self, token_ids):
        """Predict next token id + probability. Greedy (argmax)."""
        E = self.tensors['E']
        W1 = self.tensors['W1']
        b1 = self.tensors['b1']
        W2 = self.tensors['W2']
        b2 = self.tensors['b2']

        # Isi context window dari kanan (terbaru), padding di kiri
        ctx = np.full(self.cw, self.pad_id, dtype=np.int64)
        recent = token_ids[-self.cw:]
        ctx[-len(recent):] = recent

        # Embedding + flatten
        emb = E[ctx]                 # (cw, D)
        flat = emb.reshape(-1)       # (cw * D,)

        # ContextMLP: Linear -> Tanh
        hidden = np.tanh(flat @ W1 + b1)   # (H,)

        # LM Head: Linear -> Softmax
        logits = hidden @ W2 + b2          # (vocab,)

        # Softmax numerically stable
        z = logits - logits.max()
        exp_z = np.exp(z)
        probs = exp_z / exp_z.sum()

        best_id = int(np.argmax(probs))
        return best_id, float(probs[best_id]), probs

    # ------------------------------------------------------------------
    def generate(self, prompt, max_tokens=48, verbose=False):
        token_ids = self.tokenize(prompt)
        if verbose:
            print(f"   Prompt tokens: {len(token_ids)} -> {self.detokenize(token_ids[:20])}...")

        # Kalau prompt sangat pendek (< cw), model akan lihat banyak padding,
        # dan training tidak pernah mengajarkan padding. Risiko output jelek.
        if len(token_ids) < self.cw and verbose:
            print(f"   ⚠️  Prompt ({len(token_ids)}) < cw ({self.cw}). "
                  f"Model mungkin kurang konteks, output bisa dangkal.")

        generated_tokens = []

        for step in range(max_tokens):
            next_id, conf, probs = self.predict(token_ids)
            token = self.vocab[next_id] if 0 <= next_id < len(self.vocab) else '<unk>'

            # Stop kalau EOS
            if next_id == self.eos_id:
                if verbose:
                    print(f"   [{step:2d}] <eos> (conf={conf:.3f}) — stop")
                break

            # Skip special tokens lain (pad/bos) — kalau muncul, stop saja
            if next_id in (self.pad_id, self.bos_id):
                if verbose:
                    print(f"   [{step:2d}] {token} (conf={conf:.3f}) — stop (special)")
                break

            # Repetition guard: 3x sama berturut-turut
            if len(generated_tokens) >= 3 and \
               generated_tokens[-1] == generated_tokens[-2] == generated_tokens[-3] == token:
                if verbose:
                    print(f"   [{step:2d}] {token!r} 3x — stop (repetition)")
                break

            generated_tokens.append(token)
            token_ids.append(next_id)
            if verbose:
                print(f"   [{step:2d}] {token!r:20s} conf={conf:.3f}")

            # Jaga sliding window supaya tidak tumbuh tanpa batas
            if len(token_ids) > self.cw * 4:
                token_ids = token_ids[-self.cw * 2:]

        # Gabungkan — untuk tokenizer pola `\w+|[^\s\w]`, spasi adalah pemisah alami
        return ' '.join(generated_tokens).strip()


# ============================================================================
# MAIN
# ============================================================================

def main():
    # CLI: <artifact|url> [chat_file] [output_file]
    # Kalau tidak diberi argumen, pakai ARTIFACT_URL_PLACEHOLDER
    artifact_src = sys.argv[1] if len(sys.argv) > 1 else ARTIFACT_URL_PLACEHOLDER
    chat_file = sys.argv[2] if len(sys.argv) > 2 else "chat.txt"
    output_file = sys.argv[3] if len(sys.argv) > 3 else "response.txt"

    # ---- Step 1: resolve artifact ----
    local_zip = resolve_artifact(artifact_src)

    # ---- Step 2: extract ----
    extract_dir = "model_extracted"
    extract_nested_zip(local_zip, extract_dir)

    # ---- Step 3: find model files ----
    bin_path, manifest_path = find_model_files(extract_dir)
    print(f"📄 Model:    {bin_path}")
    print(f"📋 Manifest: {manifest_path}")

    # ---- Step 4: load model ----
    model = TinyLMInference(bin_path, manifest_path, verbose=True)

    # ---- Step 5: read input ----
    if not os.path.exists(chat_file):
        print(f"⚠️  {chat_file} not found. Creating empty one.")
        with open(chat_file, 'w', encoding='utf-8') as f:
            f.write("")
        prompt = ""
    else:
        with open(chat_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()

    print(f"\n💬 Input ({chat_file}):")
    preview = prompt[:200] + ('...' if len(prompt) > 200 else '')
    print(f"   \"{preview}\"")

    # ---- Step 6: generate ----
    if prompt:
        print(f"\n🧪 Generating (max 48 tokens)...")
        response = model.generate(prompt, max_tokens=48, verbose=True)
    else:
        response = "(No input provided in chat.txt)"

    # ---- Step 7: write output ----
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(response)

    print(f"\n🤖 Output ({output_file}):")
    print(f"   \"{response}\"")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
