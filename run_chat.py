#!/usr/bin/env python3
"""
TinyLM-ContextMLP Inference Runner for GitHub Actions
- Handles nested ZIP extraction
- Reads prompt from chat.txt
- Writes response to response.txt
- Pure NumPy inference (no PyTorch needed)
"""

import os
import sys
import json
import zipfile
import re
import glob
import numpy as np


def extract_nested_zip(zip_path, extract_dir):
    """Recursively extract ZIP files, handling nested ZIPs."""
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_dir)
        print(f"✅ Extracted: {zip_path} -> {extract_dir}")
    
    # Check for nested ZIPs and extract them too
    nested_zips = glob.glob(os.path.join(extract_dir, "**/*.zip"), recursive=True)
    for nested in nested_zips:
        nested_dir = os.path.splitext(nested)[0]
        print(f"📦 Found nested ZIP: {nested}")
        with zipfile.ZipFile(nested, 'r') as z:
            z.extractall(nested_dir)
        print(f"✅ Extracted nested: {nested} -> {nested_dir}")
        # Clean up nested zip after extraction
        os.remove(nested)
    
    return extract_dir


def find_model_files(base_dir):
    """Find model.bin and manifest.json recursively."""
    bin_files = glob.glob(os.path.join(base_dir, "**/*.bin"), recursive=True)
    manifest_files = glob.glob(os.path.join(base_dir, "**/manifest.json"), recursive=True)
    
    if not bin_files:
        raise FileNotFoundError("No .bin file found in extracted archive")
    if not manifest_files:
        raise FileNotFoundError("No manifest.json found in extracted archive")
    
    # Prefer files in same directory
    for mf in manifest_files:
        mf_dir = os.path.dirname(mf)
        for bf in bin_files:
            if os.path.dirname(bf) == mf_dir:
                return bf, mf
    
    return bin_files[0], manifest_files[0]


class TinyLMInference:
    """Pure NumPy inference engine for TinyLM-ContextMLP."""
    
    def __init__(self, bin_path, manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)
        
        cfg = self.manifest.get('model', {}).get('hyperparameters', {})
        if not cfg:
            cfg = self.manifest.get('text_model', {}).get('cfg', {})
        
        self.cw = cfg.get('cw', 16)
        self.D = cfg.get('D', 125)
        self.H = cfg.get('H', 125)
        self.vocab_size = self.manifest.get('model', {}).get('vocab', {}).get('size', 5000)
        
        # Build vocab lookup
        id_to_token = self.manifest.get('model', {}).get('vocab', {}).get('id_to_token', [])
        self.vocab = [t.strip() if isinstance(t, str) else '<unk>' for t in id_to_token]
        self.token_to_id = {tok: idx for idx, tok in enumerate(self.vocab)}
        
        # Load tensors from binary
        tensor_specs = self.manifest.get('model', {}).get('tensors', [])
        self.tensors = {}
        
        with open(bin_path, 'rb') as f:
            raw = f.read()
        
        offset = 0
        for spec in tensor_specs:
            name = spec['name'].strip()
            length = spec['length']
            arr = np.frombuffer(raw, dtype=np.float32, count=length, offset=offset).copy()
            
            shape = spec.get('shape', None)
            if shape:
                arr = arr.reshape(shape)
            
            self.tensors[name] = arr
            offset += length * 4
        
        print(f"🧠 Model loaded: cw={self.cw}, D={self.D}, H={self.H}, vocab={self.vocab_size}")
        print(f"   Tensors: {list(self.tensors.keys())}")
    
    def tokenize(self, text):
        pattern = r"\p{L}[\p{L}\p{M}]*|\p{N}+|[^\s]"
        try:
            tokens = re.findall(pattern, text.lower(), re.UNICODE)
        except Exception:
            tokens = re.findall(r"\w+|[^\s\w]", text.lower())
        return [self.token_to_id.get(t, self.token_to_id.get('<unk>', 1)) for t in tokens]
    
    def predict(self, token_ids):
        E = self.tensors['E']
        W1 = self.tensors['W1']
        b1 = self.tensors['b1']
        W2 = self.tensors['W2']
        b2 = self.tensors['b2']
        
        # Pad/truncate to context window
        ctx = np.zeros(self.cw, dtype=np.int64)
        start = max(0, len(token_ids) - self.cw)
        ctx[:len(token_ids[start:])] = token_ids[start:]
        
        # Embedding lookup + flatten
        emb = E[ctx]  # (cw, D)
        flat = emb.flatten()  # (cw * D,)
        
        # Context MLP: Linear + Tanh
        hidden = np.tanh(flat @ W1 + b1)  # (H,)
        
        # LM Head: Linear + Softmax
        logits = hidden @ W2 + b2  # (vocab_size,)
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        best_id = int(np.argmax(probs))
        return best_id, probs[best_id]
    
    def generate(self, prompt, max_tokens=48):
        token_ids = self.tokenize(prompt)
        generated = []
        
        for _ in range(max_tokens):
            next_id, conf = self.predict(token_ids)
            token = self.vocab[next_id] if next_id < len(self.vocab) else '<unk>'
            
            if token in ('<eos>', '<pad>', '<bos>'):
                break
            
            generated.append(token)
            token_ids.append(next_id)
            
            if len(token_ids) > self.cw * 2:
                token_ids = token_ids[-self.cw:]
        
        return ' '.join(generated).strip()


def main():
    artifact_zip = sys.argv[1] if len(sys.argv) > 1 else "model_artifact.zip"
    chat_file = sys.argv[2] if len(sys.argv) > 2 else "chat.txt"
    output_file = sys.argv[3] if len(sys.argv) > 3 else "response.txt"
    
    # Step 1: Extract (handles nested ZIPs)
    extract_dir = "model_extracted"
    extract_nested_zip(artifact_zip, extract_dir)
    
    # Step 2: Find model files
    bin_path, manifest_path = find_model_files(extract_dir)
    print(f"📄 Model: {bin_path}")
    print(f"📋 Manifest: {manifest_path}")
    
    # Step 3: Load model
    model = TinyLMInference(bin_path, manifest_path)
    
    # Step 4: Read input
    if not os.path.exists(chat_file):
        print(f"⚠️  {chat_file} not found. Creating empty one.")
        with open(chat_file, 'w') as f:
            f.write("")
        prompt = ""
    else:
        with open(chat_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
    
    print(f"\n💬 Input ({chat_file}):")
    print(f"   \"{prompt[:200]}{'...' if len(prompt) > 200 else ''}\"")
    
    # Step 5: Generate
    if prompt:
        response = model.generate(prompt, max_tokens=48)
    else:
        response = "(No input provided in chat.txt)"
    
    # Step 6: Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(response)
    
    print(f"\n🤖 Output ({output_file}):")
    print(f"   \"{response}\"")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
