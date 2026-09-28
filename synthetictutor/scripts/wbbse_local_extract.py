"""
wbbse_local_extract.py
Offline Bengali PDF text extractor using EasyOCR + PyMuPDF.

Downloads WBBSE PDFs from Google Drive folder, renders pages to images,
performs Bengali OCR via EasyOCR (CPU/GPU), and outputs UTF-8 Bengali .txt files
into raw-textbooks/wbbse/class_X/subject/.

Usage:
    python wbbse_local_extract.py              # extract all 22 books
    python wbbse_local_extract.py --dry-run    # print plan only
    python wbbse_local_extract.py --class 6    # extract Class 6 only
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import pymupdf
import easyocr

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


def extract_book_text_ocr(pdf_path: Path, reader: easyocr.Reader) -> str:
    doc = pymupdf.open(pdf_path)
    page_texts = []
    print(f"    Processing {len(doc)} pages with EasyOCR...")
    
    tmp_img = Path("C:/Temp") / f"tmp_ocr_{os.getpid()}.png"
    tmp_img.parent.mkdir(exist_ok=True)
    
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(dpi=150)
        pix.save(tmp_img)
        
        lines = reader.readtext(str(tmp_img), detail=0)
        page_str = "\n".join(lines).strip()
        
        if page_str:
            page_texts.append(f"## পৃষ্ঠা {i+1}\n\n{page_str}")
        
        if (i + 1) % 10 == 0 or i == len(doc) - 1:
            print(f"    Processed {i+1}/{len(doc)} pages...")
    
    tmp_img.unlink(missing_ok=True)
    return "\n\n".join(page_texts)


def process_book(file_id: str, filename: str, class_dir: str, subject_dir: str,
                 slug: str, reader: easyocr.Reader = None, dry_run: bool = False) -> bool:
    out_dir = WBBSE_DIR / class_dir / subject_dir
    out_txt = out_dir / f"{slug}.txt"
    
    if out_txt.exists() and out_txt.stat().st_size > 500:
        print(f"  SKIP (exists): {out_txt.relative_to(WBBSE_DIR)}")
        return True
    
    print(f"\n{'[DRY] ' if dry_run else ''}Book: {filename}")
    print(f"  -> {class_dir}/{subject_dir}/{slug}.txt")
    
    if dry_run:
        return True
    
    tmp = Path("C:/Temp") / f"{file_id}.pdf"
    tmp.parent.mkdir(exist_ok=True)
    
    print(f"  Downloading...")
    if not download_pdf(file_id, tmp):
        print(f"  FAILED download")
        return False
    print(f"  Downloaded: {tmp.stat().st_size // 1024} KB")
    
    try:
        text = extract_book_text_ocr(tmp, reader)
    except Exception as e:
        print(f"  FAILED extraction: {e}")
        return False
    finally:
        tmp.unlink(missing_ok=True)
    
    if len(text.strip()) < 200:
        print(f"  WARNING: short output ({len(text)} chars)")
        return False
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(text, encoding="utf-8")
    bn = len(re.findall(r"[\u0980-\u09FF]", text))
    print(f"  OK: {len(text):,} chars, {bn:,} Bengali codepoints -> {out_txt.name}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--class-filter", help="Filter by class_6, class_7 etc.")
    args = ap.parse_args()
    
    drive_json = Path("e:/Downloads/SyntheticTutor/wbbse_drive_files.json")
    files = json.loads(drive_json.read_text(encoding="utf-8-sig"))
    name_to_id = {f["name"]: f["id"] for f in files}
    
    reader = None
    if not args.dry_run:
        print("Initializing EasyOCR Bengali Reader...")
        reader = easyocr.Reader(['bn'], gpu=False)
    
    ok = fail = 0
    for filename, (class_dir, subject_dir, slug) in BOOK_MAP.items():
        if args.class_filter and class_dir != args.class_filter:
            continue
        file_id = name_to_id.get(filename)
        if not file_id:
            print(f"  NOT IN DRIVE: {filename}")
            fail += 1
            continue
        result = process_book(file_id, filename, class_dir, subject_dir, slug, reader, args.dry_run)
        if result:
            ok += 1
        else:
            fail += 1
    
    print(f"\n=== {ok} ok  {fail} failed ===")


if __name__ == "__main__":
    main()