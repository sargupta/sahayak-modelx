"""
Deep OCR Cleaner and Sanitizer for SahayakAI Grounding Corpus.
Removes workbook glyph noise, character/number stutters, mixed-script OCR concatenation,
garbled math symbols, and quarantines unrecoverable handwriting/tracing practice pages.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set

try:
    from .coverage_audit import CorpusCoverageAuditor
except ImportError:
    from synthetictutor.corpus.coverage_audit import CorpusCoverageAuditor


# Regex definitions for deep cleaning
OCR_GLYPH_BOXES = re.compile(r'[☐☑☒■□▲▼►◄\u2600-\u26FF\u2700-\u27BFœ€®©™]')
REPEATED_STUTTERS = re.compile(r'(\b[a-zA-Z0-9০-৯\u0980-\u09FF]\b\s*){4,}')
CHAR_RUNS = re.compile(r'([^\s0-9০-৯\.\-_])\1{3,}')
NUM_RUNS = re.compile(r'([0-9০-৯])\1{4,}')
GARBLED_MATH = re.compile(r'(\+\=|\=\+|\/\s*\/|\*\s*\*|\(\s*\(|\)\s*\))')
MIXED_SCRIPT_BN_EN = re.compile(r'([\u0980-\u09FF]+)([a-zA-Z]+)')
MIXED_SCRIPT_EN_BN = re.compile(r'([a-zA-Z]+)([\u0980-\u09FF]+)')
STANDALONE_PUNCT_DUMPS = re.compile(r'(\s*[\.,:;\|\/\-_\?]\s*){4,}')


class DeepOCRCleaner:
    def __init__(self, corpus_dir: str = "datasets/final_grounding_corpus"):
        self.corpus_dir = Path(corpus_dir)
        self.text_only_path = self.corpus_dir / "grounding_chunks_text_only.jsonl"
        self.visual_context_path = self.corpus_dir / "grounding_chunks_visual_context.jsonl"
        self.source_pool_path = self.corpus_dir / "wbbse_sft_source_pool.jsonl"
        self.excluded_archive_path = self.corpus_dir / "excluded_archive.jsonl"

    def clean_deep_text(self, text: str) -> str:
        if not text:
            return ""

        # 1. Strip OCR glyph boxes and symbol artifacts
        cleaned = OCR_GLYPH_BOXES.sub(' ', text)

        # 2. Clean repeated alphabet drills / tracing stutters (e.g. q q q q q, a a a a)
        cleaned = REPEATED_STUTTERS.sub(' ', cleaned)

        # 3. Clean long character/number runs
        cleaned = CHAR_RUNS.sub(r'\1', cleaned)
        cleaned = NUM_RUNS.sub(r'\1', cleaned)

        # 4. Clean standalone punctuation dumps
        cleaned = STANDALONE_PUNCT_DUMPS.sub(' ', cleaned)

        # 5. Separate mixed script glued tokens
        cleaned = MIXED_SCRIPT_BN_EN.sub(r'\1 \2', cleaned)
        cleaned = MIXED_SCRIPT_EN_BN.sub(r'\1 \2', cleaned)

        # 6. Normalize garbled math operators
        cleaned = GARBLED_MATH.sub(' ', cleaned)

        # 7. Normalize whitespaces
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    def is_tracing_or_workbook_junk(self, text: str) -> bool:
        """Flags chunks that are predominantly letter tracing / workbook fill-in grids."""
        words = text.split()
        if not words or len(words) < 5:
            return True

        # Check ratio of single-character tokens (tracing drills)
        single_char_tokens = [w for w in words if len(w) == 1 and w not in ['a', 'I', 'ও', 'এ', 'বা']]
        if len(single_char_tokens) / max(1, len(words)) > 0.40:
            return True

        # Check total meaningful Bengali / English words
        meaningful_words = [w for w in words if len(w) >= 3 and re.search(r'[\u0980-\u09FFa-zA-Z]', w)]
        if len(meaningful_words) < 4:
            return True

        return False

    def clean_file(self, fpath: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
        clean_records = []
        quarantined = []
        stats = {"total": 0, "cleaned": 0, "quarantined": 0}

        if not fpath.exists():
            return clean_records, quarantined, stats

        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue

                stats["total"] += 1
                try:
                    rec = json.loads(line_str)
                except Exception:
                    stats["quarantined"] += 1
                    continue

                raw_text = rec.get("source_text", rec.get("text", ""))
                cleaned_text = self.clean_deep_text(raw_text)

                if self.is_tracing_or_workbook_junk(cleaned_text):
                    rec["exclusion_reason"] = "WORKBOOK_TRACING_OR_OCR_GRID_JUNK"
                    quarantined.append(rec)
                    stats["quarantined"] += 1
                    continue

                if cleaned_text != raw_text:
                    stats["cleaned"] += 1

                if "source_text" in rec:
                    rec["source_text"] = cleaned_text
                elif "text" in rec:
                    rec["text"] = cleaned_text

                clean_records.append(rec)

        return clean_records, quarantined, stats

    def execute(self) -> Dict[str, Any]:
        results = {}
        all_quarantined = []

        # 1. Clean text-only
        clean_txt, quar_txt, stats_txt = self.clean_file(self.text_only_path)
        with open(self.text_only_path, "w", encoding="utf-8") as f:
            for r in clean_txt:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        all_quarantined.extend(quar_txt)
        results["grounding_chunks_text_only.jsonl"] = stats_txt

        # 2. Clean visual context
        clean_vis, quar_vis, stats_vis = self.clean_file(self.visual_context_path)
        with open(self.visual_context_path, "w", encoding="utf-8") as f:
            for r in clean_vis:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        all_quarantined.extend(quar_vis)
        results["grounding_chunks_visual_context.jsonl"] = stats_vis

        # 3. Clean source pool
        clean_src, quar_src, stats_src = self.clean_file(self.source_pool_path)
        with open(self.source_pool_path, "w", encoding="utf-8") as f:
            for r in clean_src:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        all_quarantined.extend(quar_src)
        results["wbbse_sft_source_pool.jsonl"] = stats_src

        # 4. Append quarantined
        if all_quarantined:
            with open(self.excluded_archive_path, "a", encoding="utf-8") as f:
                for r in all_quarantined:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 5. Regenerate manifest and audit
        auditor = CorpusCoverageAuditor(corpus_dir=self.corpus_dir)
        auditor.load_corpus()
        audit_json = auditor.generate_audit_json()
        manifest_json = auditor.generate_manifest_v1_1()

        results["total_quarantined"] = len(all_quarantined)
        results["final_text_only_count"] = len(clean_txt)
        results["final_visual_context_count"] = len(clean_vis)
        results["final_source_pool_count"] = len(clean_src)

        return results


if __name__ == "__main__":
    cleaner = DeepOCRCleaner()
    res = cleaner.execute()
    print("Deep OCR Cleaning Completed Successfully!")
    print(json.dumps(res, ensure_ascii=False, indent=2))
