"""
parallel_ocr.py
Optimized multi-core OCR runner with single-thread PyTorch enforcement.
Prevents PyTorch OpenMP/MKL thread thrashing across CPU cores.
"""

import os
# Enforce single-thread PyTorch before any torch imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pymupdf
import easyocr
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

_global_reader = None

def _init_worker():
    global _global_reader
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    _global_reader = easyocr.Reader(['bn'], gpu=False)

def _ocr_page_worker(args):
    global _global_reader
    pdf_path_str, page_num, dpi = args
    doc = pymupdf.open(pdf_path_str)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    tmp_img = f"C:/Temp/ocr_{os.getpid()}_{page_num}.png"
    pix.save(tmp_img)
    doc.close()
    
    lines = _global_reader.readtext(tmp_img, detail=0)
    
    if os.path.exists(tmp_img):
        try:
            os.remove(tmp_img)
        except Exception:
            pass
    return page_num, "\n".join(lines)

def run_parallel_ocr(pdf_path: Path, max_workers: int = 2, dpi: int = 100) -> str:
    doc = pymupdf.open(pdf_path)
    num_pages = len(doc)
    doc.close()
    
    print(f"  Running Single-Threaded-PyTorch EasyOCR ({max_workers} workers, {dpi} DPI) on {num_pages} pages...")
    tasks = [(str(pdf_path), i, dpi) for i in range(num_pages)]
    
    results = {}
    t0 = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as executor:
        futures = {executor.submit(_ocr_page_worker, task): task[1] for task in tasks}
        completed = 0
        for future in as_completed(futures):
            page_num, page_str = future.result()
            results[page_num] = page_str
            completed += 1
            if completed % 5 == 0 or completed == num_pages:
                dt = time.time() - t0
                rate = completed / max(1e-5, dt)
                print(f"    Completed {completed}/{num_pages} pages ({rate:.2f} pages/sec, {dt/completed:.1f}s/page)...")
    
    pages_text = []
    for i in range(num_pages):
        page_str = results.get(i, "").strip()
        if page_str:
            pages_text.append(f"## পৃষ্ঠা {i+1}\n\n{page_str}")
    
    return "\n\n".join(pages_text)

if __name__ == "__main__":
    test_pdf = Path("C:/Temp/wbbse_pdfs/16dMEAXs11_IlbV7SOKp9zBeY7NgPhjb7.pdf")
    if test_pdf.exists():
        t0 = time.time()
        print("Benchmarking 5 pages with single-thread PyTorch enforcement...")
        tasks = [(str(test_pdf), i, 100) for i in range(5)]
        with ProcessPoolExecutor(max_workers=2, initializer=_init_worker) as ex:
            res = list(ex.map(_ocr_page_worker, tasks))
        dt = time.time() - t0
        print(f"DONE 5 pages in {dt:.2f}s ({dt/5:.2f}s/page)")