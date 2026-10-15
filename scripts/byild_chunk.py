#!/usr/bin/env python3
"""
Build 1 chunk dataset JSONL (gzip streaming, hemat disk).

Format per baris:
    {"text": {"lang": "...", "category": "...", "content": "..."}}

Contoh:
    python scripts/build_chunk.py \
        --chunk-id 0 --chunk-tokens 25000000 --output part_000.jsonl.gz
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import signal
import sys
import time
from pathlib import Path

import requests

USER_AGENT = (
    "AxionAI-DatasetBuilder/1.0 "
    "(educational; +https://github.com/azrielspace852-wq/axion-ai)"
)
CHARS_PER_TOKEN   = 4
MIN_ARTICLE_CHARS = 800

LANG_WEIGHTS = {
    "en": 0.40,
    "de": 0.10,
    "fr": 0.10,
    "es": 0.10,
    "ru": 0.08,
    "it": 0.07,
    "ja": 0.06,
    "zh": 0.05,
    "pt": 0.04,
}

CATEGORY_PREFIXES = (
    "Category:", "Kategori:", "Catégorie:", "Kategorie:",
    "Categoría:", "Категория:", "Categoria:", "カテゴリ:", "分类:",
)

MAX_RETRIES    = 4
RETRY_BACKOFF  = 1.6     # detik → 1.6, 2.56, 4.1, 6.55
FETCH_BATCH_SZ = 20      # exlimit=max hanya melayani ~20 extract per query


# ---------------------- utils ----------------------

def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def pick_langs(n: int) -> list[str]:
    """Pilih n bahasa unik dengan bobot sesuai LANG_WEIGHTS."""
    if n <= 0:
        return []
    keys, weights = list(LANG_WEIGHTS.keys()), list(LANG_WEIGHTS.values())
    chosen: list[str] = []
    for _ in range(n * 8):
        if len(chosen) >= n:
            break
        c = random.choices(keys, weights=weights, k=1)[0]
        if c not in chosen:
            chosen.append(c)
    for k in keys:                       # fallback deterministik
        if len(chosen) >= n:
            break
        if k not in chosen:
            chosen.append(k)
    return chosen[:n]


def fingerprint(text: str) -> str:
    return hashlib.md5(text[:500].encode("utf-8", "ignore")).hexdigest()


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return s


# ---------------------- fetch ----------------------

def fetch_batch(session: requests.Session, lang: str, count: int) -> list[dict]:
    """Ambil batch artikel acak Wikipedia <lang> dengan retry+backoff."""
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "generator": "random",
        "grnnamespace": 0,
        "grnlimit": count,
        "prop": "extracts|categories|info",
        "inprop": "url",
        "explaintext": True,
        "exlimit": "max",
        "cllimit": "max",
        "clshow": "!hidden",
    }

    data = None
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=60)
            # backoff khusus untuk rate limit
            if r.status_code in (429, 503):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            last_err = e
            wait = RETRY_BACKOFF ** attempt
            print(f"[!] fetch {lang} gagal ({attempt}/{MAX_RETRIES}): {e} "
                  f"→ tunggu {wait:.1f}s", flush=True)
            time.sleep(wait)

    if data is None:
        print(f"[!] menyerah fetch {lang}: {last_err}", flush=True)
        return []

    out: list[dict] = []
    for page in data.get("query", {}).get("pages", []) or []:
        extract = (page.get("extract") or "").strip()
        if len(extract) < MIN_ARTICLE_CHARS:
            continue

        category = "general"
        for c in page.get("categories", []) or []:
            title = c.get("title", "")
            for prefix in CATEGORY_PREFIXES:
                if title.startswith(prefix):
                    category = title[len(prefix):] or "general"
                    break
            if category != "general":
                break

        out.append({
            "page_id": page.get("pageid"),
            "text": {
                "lang":     lang,
                "category": category,
                "content":  extract,
            },
        })
    return out


# ---------------------- main ----------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-id",     type=int, required=True)
    ap.add_argument("--chunk-tokens", type=int, default=25_000_000)
    ap.add_argument("--output",       required=True)
    ap.add_argument("--max-langs",    type=int, default=4,
                    help="jumlah bahasa berbeda yang dicoba per iterasi")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.chunk_tokens <= 0:
        print("[✗] --chunk-tokens harus > 0", flush=True)
        return 2

    random.seed(1000 + args.chunk_id)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    target = args.chunk_tokens
    total  = 0
    seen_ids: set[tuple[str, int | None]] = set()
    seen_fp:  dict[str, set[str]] = {}

    session = _session()

    print(f"[chunk {args.chunk_id}] target={target:,} token → {args.output}",
          flush=True)

    # ---- graceful shutdown (agar file gzip tetap valid) ----
    stop = {"flag": False}
    def _on_signal(signum, _frame):
        stop["flag"] = True
        print(f"[!] signal {signum} diterima → menutup file dengan rapi...",
              flush=True)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT,  _on_signal)

    empty_streak = 0
    try:
        with gzip.open(args.output, "wt", encoding="utf-8",
                       compresslevel=6) as f:
            while total < target and not stop["flag"]:
                for lang in pick_langs(args.max_langs):
                    if total >= target or stop["flag"]:
                        break

                    batch = fetch_batch(session, lang, FETCH_BATCH_SZ)
                    if not batch:
                        empty_streak += 1
                        if empty_streak >= 20:
                            raise RuntimeError(
                                "API gagal 20x berturut-turut — hentikan."
                            )
                        continue
                    empty_streak = 0

                    for rec in batch:
                        key = (rec["text"]["lang"], rec["page_id"])
                        if key in seen_ids:
                            continue
                        seen_ids.add(key)

                        fp_set = seen_fp.setdefault(rec["text"]["lang"], set())
                        fp = fingerprint(rec["text"]["content"])
                        if fp in fp_set:
                            continue
                        fp_set.add(fp)

                        f.write(json.dumps({"text": rec["text"]},
                                           ensure_ascii=False) + "\n")
                        total += estimate_tokens(rec["text"]["content"])
                        if total >= target:
                            break

                    pct = (100.0 * total / target) if target else 0.0
                    print(f"[chunk {args.chunk_id}] {total:,} token "
                          f"({pct:5.1f}%)", flush=True)
                    time.sleep(0.15)

    except RuntimeError as e:
        # sengaja non-fatal: file gzip tetap ditutup dengan rapi
        print(f"[!] {e}", flush=True)
    except Exception as e:
        print(f"[!] error tak terduga: {e}", flush=True)
        raise

    out = Path(args.output)
    if not out.exists() or out.stat().st_size == 0:
        print(f"[✗] {out} tidak terbentuk / kosong", flush=True)
        return 1

    size_mb = out.stat().st_size / 1e6
    print(f"[✓] {out}: {total:,} token, {size_mb:.1f} MB gzip", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main()) 
