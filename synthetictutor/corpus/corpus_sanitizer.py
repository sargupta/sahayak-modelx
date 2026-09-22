"""
Automated Corpus Sanitizer and Auto-Cleaner for SahayakAI Grounding Corpus.
Cleans OCR noise, strips repetitive punctuation, repairs unclosed LaTeX tags, fixes mojibake,
auto-corrects language medium tags, and quarantines unrecoverable garbage chunks.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
try:
    from .coverage_audit import CorpusCoverageAuditor
except ImportError:
    from synthetictutor.corpus.coverage_audit import CorpusCoverageAuditor




MOJIBAKE_REGEX = re.compile(r'[\ufffd\u0080-\u009f]|à[¦§]|Ã|Â|â€')
EXCESSIVE_PUNCT_REGEX = re.compile(r'([!?.~@#$%^&*_=+|\\/-])\1{2,}')
REPETITIVE_DOTS = re.compile(r'\.{3,}')
REPETITIVE_DASHES = re.compile(r'-{3,}')
BENGALI_CHARS_REGEX = re.compile(r'[\u0980-\u09FF]')
ENGLISH_CHARS_REGEX = re.compile(r'[a-zA-Z]')
STRAY_PIPE_REGEX = re.compile(r'\|{2,}')


class CorpusSanitizer:
    """
    Cleans, repairs, and sanitizes grounding chunks across the entire corpus.
    """

    def __init__(self, corpus_dir: str = "datasets/final_grounding_corpus"):
        self.corpus_dir = Path(corpus_dir)
        self.text_only_path = self.corpus_dir / "grounding_chunks_text_only.jsonl"
        self.visual_context_path = self.corpus_dir / "grounding_chunks_visual_context.jsonl"
        self.source_pool_path = self.corpus_dir / "wbbse_sft_source_pool.jsonl"
        self.excluded_archive_path = self.corpus_dir / "excluded_archive.jsonl"

    def clean_text(self, text: str) -> str:
        """Applies heuristic rule-based cleaning to an individual chunk text."""
        if not text:
            return ""

        # 1. Remove Mojibake and replacement characters
        cleaned = MOJIBAKE_REGEX.sub(' ', text)

        # 2. Normalize repetitive OCR noise (dashes, dots, pipes, underlines)
        cleaned = REPETITIVE_DASHES.sub(' ', cleaned)
        cleaned = REPETITIVE_DOTS.sub('... ', cleaned)
        cleaned = STRAY_PIPE_REGEX.sub(' ', cleaned)
        cleaned = EXCESSIVE_PUNCT_REGEX.sub(r'\1', cleaned)

        # 3. Clean orphan standalone viramas/hasant
        cleaned = re.sub(r'^\s*্', '', cleaned)
        cleaned = re.sub(r'\s্', ' ', cleaned)

        # 4. Repair unclosed LaTeX math dollar signs
        dollar_count = cleaned.count('$')
        if dollar_count % 2 != 0:
            # If odd number of $ signs, replace lone $ with empty space unless it matches a pair
            parts = cleaned.split('$')
            if len(parts) == 2:
                # Lone single dollar sign
                cleaned = cleaned.replace('$', '')
            else:
                # Drop trailing unpaired dollar
                cleaned = '$'.join(parts[:-1]) + ' ' + parts[-1]

        # 5. Normalize whitespace
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    def detect_true_medium(self, text: str, declared_medium: str) -> str:
        """Detects if a chunk is predominantly English or Bengali."""
        bn_count = len(BENGALI_CHARS_REGEX.findall(text))
        en_count = len(ENGLISH_CHARS_REGEX.findall(text))

        if en_count > 50 and en_count > (bn_count * 2):
            return "English"
        elif bn_count > 30:
            return "Bengali"
        return declared_medium

    def is_irreparable_garbage(self, text: str) -> bool:
        """Flags chunks that are completely unrecoverable junk or too degraded for model training."""
        stripped = text.strip()
        if len(stripped) < 25:
            return True

        # Check total meaningful alphanumeric characters
        alpha_count = len(re.findall(r'[a-zA-Z0-9\u0980-\u09FF]', stripped))
        if alpha_count < 15:
            return True

        # High non-alphanumeric noise ratio (> 60% symbols/noise)
        if (alpha_count / max(1, len(stripped))) < 0.35:
            return True

        return False

    def sanitize_file(self, fpath: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
        """Cleans a single JSONL file, separating clean vs quarantined records."""
        clean_records = []
        quarantined_records = []
        stats = {
            "total": 0,
            "cleaned": 0,
            "medium_corrected": 0,
            "quarantined": 0
        }

        if not fpath.exists():
            return clean_records, quarantined_records, stats

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
                cleaned_text = self.clean_text(raw_text)

                if self.is_irreparable_garbage(cleaned_text):
                    rec["exclusion_reason"] = "CORRUPTED_OR_HIGH_OCR_NOISE"
                    quarantined_records.append(rec)
                    stats["quarantined"] += 1
                    continue

                if cleaned_text != raw_text:
                    stats["cleaned"] += 1

                # Update text
                if "source_text" in rec:
                    rec["source_text"] = cleaned_text
                elif "text" in rec:
                    rec["text"] = cleaned_text

                # Auto-correct medium tag
                orig_medium = rec.get("medium", "Bengali")
                true_medium = self.detect_true_medium(cleaned_text, orig_medium)
                if true_medium != orig_medium:
                    rec["medium"] = true_medium
                    stats["medium_corrected"] += 1

                rec["source_quality"] = "HIGH"
                clean_records.append(rec)

        return clean_records, quarantined_records, stats

    def run_sanitization(self) -> Dict[str, Any]:
        """Runs sanitization across all corpus files and updates manifest/audit."""
        results = {}
        all_quarantined = []

        # 1. Sanitize text-only chunks
        clean_text, quar_text, stats_text = self.sanitize_file(self.text_only_path)
        with open(self.text_only_path, "w", encoding="utf-8") as f:
            for r in clean_text:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        all_quarantined.extend(quar_text)
        results["grounding_chunks_text_only.jsonl"] = stats_text

        # 2. Sanitize visual-context chunks
        clean_vis, quar_vis, stats_vis = self.sanitize_file(self.visual_context_path)
        with open(self.visual_context_path, "w", encoding="utf-8") as f:
            for r in clean_vis:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        all_quarantined.extend(quar_vis)
        results["grounding_chunks_visual_context.jsonl"] = stats_vis

        # 3. Sanitize source pool
        clean_pool, quar_pool, stats_pool = self.sanitize_file(self.source_pool_path)
        with open(self.source_pool_path, "w", encoding="utf-8") as f:
            for r in clean_pool:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        all_quarantined.extend(quar_pool)
        results["wbbse_sft_source_pool.jsonl"] = stats_pool

        # 4. Append quarantined records to excluded_archive.jsonl
        if all_quarantined:
            with open(self.excluded_archive_path, "a", encoding="utf-8") as f:
                for r in all_quarantined:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 5. Regenerate Audit and Manifest v1.1
        auditor = CorpusCoverageAuditor(corpus_dir=self.corpus_dir)
        auditor.load_corpus()
        audit_json = auditor.generate_audit_json()
        manifest_json = auditor.generate_manifest_v1_1()

        results["total_quarantined_new"] = len(all_quarantined)
        results["final_text_only_count"] = len(clean_text)
        results["final_visual_context_count"] = len(clean_vis)
        results["final_source_pool_count"] = len(clean_pool)

        return results


if __name__ == "__main__":
    sanitizer = CorpusSanitizer()
    res = sanitizer.run_sanitization()
    print("Corpus Sanitization Completed Successfully!")
    print(json.dumps(res, ensure_ascii=False, indent=2))
