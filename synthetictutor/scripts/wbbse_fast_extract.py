"""
wbbse_fast_extract.py
Multi-core high-speed hybrid extractor for WBBSE textbooks.
1. Instant STM font decoding for text-layer PDFs (0.05s/book)
2. 4x Parallel EasyOCR fallback for pure image scans (120 DPI)
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import pymupdf
from stm_fast_extract import extract_pdf_stm
from parallel_ocr import run_parallel_ocr

WBBSE_DIR = Path(__file__).parent / "northbengal-dataset-forge-2026-08-19" / \
            "northbengal-dataset-forge" / "raw-textbooks" / "wbbse"

BOOK_MAP = {
    # Class 6
    "Poribesh O Bigyan Class VI.pdf":     ("class_6", "paribesh_o_bigyan", "poribesh_o_bigyan_6"),
    "Ganit Probha Class VI.pdf":          ("class_6", "ganit_probha",      "ganit_probha_6"),
    "Otit O Oitijhyo Class VI.pdf":       ("class_6", "otit_o_oitijhyo",   "otit_o_oitijhyo_6"),
    "Amader Prithibi Class VI.pdf":       ("class_6", "amader_prithibi",   "amader_prithibi_6"),
    "Sahitya Mela Class VI.pdf":          ("class_6", "sahitya_mela",      "sahitya_mela_6"),
    "Bhasa Charcha Class VI.pdf":         ("class_6", "bhasa_chorcha",     "bhasa_chorcha_6"),
    # Class 7
    "Poribesh O Bigyan Class VII.pdf":    ("class_7", "paribesh_o_bigyan", "poribesh_o_bigyan_7"),
    "Ganit Probha Class VII.pdf":         ("class_7", "ganit_probha",      "ganit_probha_7"),
    "Otit O Oitijhyo Class VII.pdf":      ("class_7", "otit_o_oitijhyo",   "otit_o_oitijhyo_7"),
    "Amader Prithibi Class VII.pdf":      ("class_7", "amader_prithibi",   "amader_prithibi_7"),
    "Sahitya Mela Class VII.pdf":         ("class_7", "sahitya_mela",      "sahitya_mela_7"),
    # Class 8
    "Poribesh O Bigyan Class VIII.pdf":   ("class_8", "paribesh_o_bigyan", "poribesh_o_bigyan_8"),
    "Ganit Probha Class VIII.pdf":        ("class_8", "ganit_probha",      "ganit_probha_8"),
    "Otit O Oitijhyo Class VIII.pdf":     ("class_8", "otit_o_oitijhyo",   "otit_o_oitijhyo_8"),
    "Amader Prithibi Class VIII.pdf":     ("class_8", "amader_prithibi",   "amader_prithibi_8"),
    "Sahitya Mela Class VIII.pdf":        ("class_8", "sahitya_mela",      "sahitya_mela_8"),
    "Bhasa Charcha Class VIII.pdf":       ("class_8", "bhasa_chorcha",     "bhasa_chorcha_8"),
    # Class 9
    "Ganit Prokash Class IX.pdf":         ("class_9", "ganit_prakash",     "ganit_prakash_9"),
    "Sahity Sanchayan Class IX.pdf":      ("class_9", "sahitya_sanchayan", "sahitya_sanchayan_9"),
    "Bliss Class IX.pdf":                 ("class_9", "bliss",             "bliss_9"),
    # Class 10
    "Ganit Prokash Class X.pdf":          ("class_10", "ganit_prakash",    "ganit_prakash_10"),
    "Bliss Class X.pdf":                  ("class_10", "bliss",            "bliss_10"),
}


def download_pdf(file_id: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 5000:
        return True
    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=1"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, stream=True, timeout=120, headers=headers)
        content = b"".join(r.iter_content(chunk_size=8192))
        if b"%PDF" not in content[:10]:
            confirm = re.search(rb'confirm=([0-9A-Za-z_-]+)', content)
            if confirm:
                url2 = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm.group(1).decode()}"
                r2 = requests.get(url2, stream=True, timeout=120, headers=headers)
                content = b"".join(r2.iter_content(chunk_size=8192))
        if b"%PDF" in content[:10]:
            dest.write_bytes(content)
            return True
    except Exception as e:
        print(f"    Download error: {e}")
    return False


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    drive_json = Path("e:/Downloads/SyntheticTutor/wbbse_drive_files.json")
    files = json.loads(drive_json.read_text(encoding="utf-8-sig"))
    name_to_id = {f["name"]: f["id"] for f in files}
    
    tmp_dir = Path("C:/Temp/wbbse_pdfs")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    ok = fail = 0
    start_t = time.time()
    
    for filename, (class_dir, subject_dir, slug) in BOOK_MAP.items():
        out_dir = WBBSE_DIR / class_dir / subject_dir
        out_txt = out_dir / f"{slug}.txt"
        
        if out_txt.exists() and out_txt.stat().st_size > 500:
            print(f"SKIP (exists): {out_txt.relative_to(WBBSE_DIR)}")
            ok += 1
            continue
        
        file_id = name_to_id.get(filename)
        if not file_id:
            print(f"NOT IN DRIVE: {filename}")
            fail += 1
            continue
        
        print(f"\nProcessing: {filename} ({class_dir}/{subject_dir}/{slug}.txt)...")
        pdf_path = tmp_dir / f"{file_id}.pdf"
        
        if not download_pdf(file_id, pdf_path):
            print(f"  FAILED download: {filename}")
            fail += 1
            continue
        
        # Method 1: Try instant STM font decoding
        t0 = time.time()
        text = extract_pdf_stm(pdf_path)
        dt = time.time() - t0
        bn_count = len(re.findall(r"[\u0980-\u09FF]", text))
        
        if bn_count > 1000:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_txt.write_text(text, encoding="utf-8")
            print(f"  OK (STM decoded in {dt:.2f}s): {len(text):,} chars, {bn_count:,} Bengali Unicode codepoints -> {out_txt.name}")
            ok += 1
            continue
        
        # Method 2: Multi-core Parallel EasyOCR fallback for image scans
        print(f"  STM yields 0 text -> running 4x Parallel EasyOCR fallback...")
        t0 = time.time()
        text = run_parallel_ocr(pdf_path, max_workers=4, dpi=120)
        dt = time.time() - t0
        bn_count = len(re.findall(r"[\u0980-\u09FF]", text))
        
        if bn_count > 100:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_txt.write_text(text, encoding="utf-8")
            print(f"  OK (Parallel EasyOCR in {dt:.1f}s): {len(text):,} chars, {bn_count:,} Bengali Unicode codepoints -> {out_txt.name}")
            ok += 1
        else:
            print(f"  FAILED extraction: insufficient text extracted ({bn_count} Bengali chars)")
            fail += 1
    
    elapsed = time.time() - start_t
    print(f"\n=== Completed in {elapsed:.1f}s: {ok} ok, {fail} failed ===")


if __name__ == "__main__":
    main()