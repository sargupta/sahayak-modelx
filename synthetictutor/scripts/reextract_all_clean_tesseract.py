"""
reextract_all_clean_tesseract.py
Re-extracts ALL WBBSE books using Tesseract 5.4 C++ OCR ONLY,
completely eliminating garbled font glyphs (þ, s, ß, iú).
"""

import io
import json
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
import pymupdf
import pytesseract
import gdown

TESS_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESS_DATA = r"e:\Downloads\SyntheticTutor\tessdata"
pytesseract.pytesseract.tesseract_cmd = TESS_EXE
os.environ["TESSDATA_PREFIX"] = TESS_DATA

WBBSE_DIR = Path(__file__).parent / "northbengal-dataset-forge-2026-08-19" / \
            "northbengal-dataset-forge" / "raw-textbooks" / "wbbse"

# Comprehensive map of all books in the Drive folder to their correct target directories
FULL_BOOK_MAP = {
    # Class 6
    "Amader Prithibi Class VI.pdf":       ("class_6", "amader_prithibi",   "amader_prithibi_6"),
    "Ganit Probha Class VI.pdf":          ("class_6", "ganit_probha",      "ganit_probha_6"),
    "Bhasa Charcha Class VI.pdf":         ("class_6", "bhasa_chorcha",     "bhasa_chorcha_6"),
    "Ha Ja Ba Ra La Class VI.pdf":        ("class_6", "sahitya_mela",      "hajabarala_6"),
    "Blossoms Class VI.pdf":              ("class_6", "blossoms",          "blossoms_6"),
    # Class 7
    "Maku Class VII.pdf":                 ("class_7", "sahitya_mela",      "maku_7"),
    "Blossoms Class VII.pdf":             ("class_7", "blossoms",          "blossoms_7"),
    # Class 8
    "Amader Prithibi Class VIII.pdf":     ("class_8", "amader_prithibi",   "amader_prithibi_8"),
    "Pather Panchali Class VIII.pdf":     ("class_8", "sahitya_mela",      "pather_panchali_8"),
    "Blossoms Class VIII.pdf":            ("class_8", "blossoms",          "blossoms_8"),
    # Class 9
    "Professor Shankur Dairy Class IX.pdf": ("class_9", "sahitya_sanchayan", "professor_shankur_9"),
    # Class 10
    "Koni\xa0Class X.pdf":                ("class_10", "sahitya_sanchayan", "koni_10"),
    "Sahitya Sanchayan\xa0Class X.pdf":   ("class_10", "sahitya_sanchayan", "sahitya_sanchayan_10"),
}


def _tess_page_worker(args):
    pdf_path_str, page_num, dpi = args
    pytesseract.pytesseract.tesseract_cmd = TESS_EXE
    os.environ["TESSDATA_PREFIX"] = TESS_DATA
    doc = pymupdf.open(pdf_path_str)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    doc.close()
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    text = pytesseract.image_to_string(img, lang="ben")
    return page_num, text.strip()


def run_parallel_tesseract(pdf_path: Path, max_workers: int = 4, dpi: int = 150) -> str:
    doc = pymupdf.open(pdf_path)
    num_pages = len(doc)
    doc.close()
    
    print(f"  Running Tesseract 5.4 C++ OCR on {num_pages} pages...")
    tasks = [(str(pdf_path), i, dpi) for i in range(num_pages)]
    results = {}
    t0 = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_tess_page_worker, task): task[1] for task in tasks}
        completed = 0
        for future in as_completed(futures):
            page_num, page_str = future.result()
            results[page_num] = page_str
            completed += 1
            if completed % 25 == 0 or completed == num_pages:
                dt = time.time() - t0
                rate = completed / max(1e-5, dt)
                print(f"    Completed {completed}/{num_pages} pages ({rate:.1f} pages/sec)...")
    
    pages_text = []
    for i in range(num_pages):
        page_str = results.get(i, "").strip()
        if page_str:
            pages_text.append(f"## পৃষ্ঠা {i+1}\n\n{page_str}")
    return "\n\n".join(pages_text)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    drive_json = Path("e:/Downloads/SyntheticTutor/wbbse_drive_files.json")
    files = json.loads(drive_json.read_text(encoding="utf-8-sig"))
    name_to_id = {f["name"]: f["id"] for f in files}
    
    tmp_dir = Path("C:/Temp/wbbse_pdfs")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, (class_dir, subject_dir, slug) in FULL_BOOK_MAP.items():
        out_dir = WBBSE_DIR / class_dir / subject_dir
        out_txt = out_dir / f"{slug}.txt"
        
        file_id = name_to_id.get(filename)
        if not file_id:
            print(f"NOT IN DRIVE MAP: {repr(filename)}")
            continue
        
        print(f"\nRe-extracting with Clean Tesseract OCR: {filename}...")
        pdf_path = tmp_dir / f"{file_id}.pdf"
        
        if not pdf_path.exists() or pdf_path.stat().st_size < 5000:
            print(f"  Downloading {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url=url, output=str(pdf_path), quiet=False)
        
        t0 = time.time()
        text = run_parallel_tesseract(pdf_path, max_workers=4, dpi=150)
        dt = time.time() - t0
        bn_count = len(re.findall(r"[\u0980-\u09FF]", text))
        
        out_dir.mkdir(parents=True, exist_ok=True)
        out_txt.write_text(text, encoding="utf-8")
        print(f"  RE-EXTRACTED OK (Tesseract 5.4 in {dt:.1f}s): {len(text):,} chars, {bn_count:,} Bengali Unicode codepoints -> {out_txt.name}")

if __name__ == "__main__":
    main()