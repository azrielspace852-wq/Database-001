#!/usr/bin/env python3
"""
Build 1 chunk dataset JSONL (gzip streaming, hemat disk).
Format per baris:
    {"text": {"lang": "...", "category": "...", "content": "..."}}
"""
import argparse
import gzip
import hashlib
import json
import random
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

USER_AGENT      = "AxionAI-Dataset/1.0 (educational; contact via GitHub)"
CHARS_PER_TOKEN = 4

# Hanya bahasa dengan korpus besar (>=250M token) — hindari duplikasi
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

MIN_ARTICLE_CHARS = 800          # skip artikel pendek (kualitas)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def pick_langs(n: int = 4) -> list:
    langs = random.choices(
        list(LANG_WEIGHTS.keys()),
        weights=list(LANG_WEIGHTS.values()),
        k=n,
    )
    # dedup tapi tetap acak
    return list(dict.fromkeys(langs))


def fetch_batch(lang: str, count: int = 50) -> list:
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
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
    try:
        r = requests.get(
            url, params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[!] fetch {lang} gagal: {e}", flush=True)
        return []

    out = []
    for page in data.get("query", {}).get("pages", {}).values():
        extract = (page.get("extract") or "").strip()
        if len(extract) < MIN_ARTICLE_CHARS:
            continue

        cats = page.get("categories", [])
        category = "general"
        for c in cats:
            t = c["title"]
            for prefix in ("Category:", "Kategori:", "Catégorie:",
                           "Kategorie:", "Categoría:", "Категория:",
                           "Categoria:", "カテゴリ:", "分类:"):
                if t.startswith(prefix):
                    category = t[len(prefix):]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-id",     type=int, required=True)
    ap.add_argument("--chunk-tokens", type=int, default=25_000_000)
    ap.add_argument("--output",       required=True)
    args = ap.parse_args()

    random.seed(1000 + args.chunk_id)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    total         = 0
    target        = args.chunk_tokens
    seen_page_ids = set()              # (lang, page_id)
    seen_titles   = {}                 # lang -> set judul (anti-redirect)

    print(f"[chunk {args.chunk_id}] target={target:,} token → {args.output}",
          flush=True)

    with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=6) as f:
        empty_streak = 0
        while total < target:
            for lang in pick_langs(4):
                if total >= target:
                    break
                batch = fetch_batch(lang, 50)
                if not batch:
                    empty_streak += 1
                    if empty_streak > 20:
                        raise RuntimeError("API terus gagal, hentikan.")
                    continue
                empty_streak = 0

                for rec in batch:
                    key = (rec["text"]["lang"], rec["page_id"])
                    if key in seen_page_ids:
                        continue
                    seen_page_ids.add(key)

                    # anti-duplikat judul (redirect ganda)
                    title_set = seen_titles.setdefault(rec["text"]["lang"], set())
                    title_hash = hashlib.md5(
                        rec["text"]["content"][:200].encode()
                    ).hexdigest()
                    if title_hash in title_set:
                        continue
                    title_set.add(title_hash)

                    f.write(json.dumps(rec["text"] and {"text": rec["text"]},
                                       ensure_ascii=False) + "\n")
                    total += estimate_tokens(rec["text"]["content"])

                pct = 100 * total / target
                print(f"[chunk {args.chunk_id}] {total:,} token ({pct:5.1f}%)",
                      flush=True)
                time.sleep(0.15)

    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"[✓] {args.output}: {total:,} token, {size_mb:.1f} MB gzip",
          flush=True)


if __name__ == "__main__":
    main()
