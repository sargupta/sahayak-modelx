"""
Comprehensive Corruption and Anomaly Scanner for SahayakAI Grounding Corpus.
Scans for:
1. Encoding & Mojibake errors (\\ufffd, cp1252/latin-1 UTF-8 misdecoding artifacts).
2. Broken Bengali conjuncts and orphaned virama characters.
3. Repetitive noise & OCR artifacts.
4. Language tagging mismatches (low Bengali character density in Bengali-tagged records).
5. Length anomalies (< 30 chars or > 8000 chars).
6. Malformed LaTeX equations (unpaired $ delimiters).
"""

import os
import re
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple


MOJIBAKE_PATTERN = re.compile(r'[\ufffd\u0080-\u009f]|à[¦§]|Ã|Â|â€')
BENGALI_RANGE = re.compile(r'[\u0980-\u09FF]')
EXCESSIVE_PUNCT = re.compile(r'([!?.~@#$%^&*_=+|\\/-])\1{4,}')
BROKEN_CONJUNCT = re.compile(r'^\s*্|\s্')
UNCLOSED_LATEX = re.compile(r'(?<!\\)\$')


class CorpusCorruptionScanner:
    def __init__(self, corpus_dir: str = "datasets/final_grounding_corpus"):
        self.corpus_dir = Path(corpus_dir)
        self.target_files = [
            "grounding_chunks_text_only.jsonl",
            "grounding_chunks_visual_context.jsonl",
            "wbbse_sft_source_pool.jsonl"
        ]

    def scan(self) -> Dict[str, Any]:
        total_scanned = 0
        corrupted_records = []
        issue_counts = Counter()
        file_counts = Counter()

        for fname in self.target_files:
            p = self.corpus_dir / fname
            if not p.exists():
                continue

            with open(p, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line_str = line.strip()
                    if not line_str:
                        continue

                    total_scanned += 1
                    file_counts[fname] += 1

                    try:
                        rec = json.loads(line_str)
                    except Exception:
                        issue_counts["JSON_PARSE_ERROR"] += 1
                        corrupted_records.append({
                            "file": fname,
                            "line": idx + 1,
                            "id": f"line_{idx+1}",
                            "issues": ["JSON_PARSE_ERROR"],
                            "preview": line_str[:100]
                        })
                        continue

                    text = rec.get("source_text", rec.get("text", ""))
                    medium = rec.get("medium", "Bengali")
                    subject = rec.get("subject", "")
                    chunk_id = rec.get("chunk_id", f"line_{idx+1}")

                    issues = []

                    # 1. Mojibake / replacement chars
                    if MOJIBAKE_PATTERN.search(text):
                        issues.append("MOJIBAKE_OR_REPLACEMENT_CHAR")

                    # 2. Broken conjuncts
                    if BROKEN_CONJUNCT.search(text):
                        issues.append("ORPHAN_BENGALI_VIRAMA")

                    # 3. Excessive punctuation noise
                    if EXCESSIVE_PUNCT.search(text):
                        issues.append("EXCESSIVE_PUNCTUATION_NOISE")

                    # 4. Length anomalies
                    if len(text.strip()) < 30:
                        issues.append("TOO_SHORT_UNDER_30_CHARS")
                    elif len(text.strip()) > 8000:
                        issues.append("TOO_LONG_OVER_8000_CHARS")

                    # 5. Language mismatch
                    if medium == "Bengali" and subject != "English":
                        bn_chars = len(BENGALI_RANGE.findall(text))
                        total_alpha = len(re.findall(r'[a-zA-Z\u0980-\u09FF]', text))
                        if total_alpha > 50 and (bn_chars / total_alpha) < 0.20:
                            issues.append("LOW_BENGALI_RATIO_IN_BENGALI_CHUNK")

                    # 6. Unclosed LaTeX
                    dollar_count = len(UNCLOSED_LATEX.findall(text))
                    if dollar_count % 2 != 0:
                        issues.append("UNCLOSED_LATEX_DOLLAR")

                    if issues:
                        for iss in issues:
                            issue_counts[iss] += 1
                        corrupted_records.append({
                            "file": fname,
                            "line": idx + 1,
                            "id": chunk_id,
                            "issues": issues,
                            "preview": text[:120].replace("\n", " ")
                        })

        return {
            "total_scanned": total_scanned,
            "total_flagged": len(corrupted_records),
            "flagged_percentage": round((len(corrupted_records) / max(1, total_scanned)) * 100, 2),
            "by_file": dict(file_counts),
            "issue_breakdown": dict(issue_counts.most_common()),
            "flagged_samples": corrupted_records[:20]
        }


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    scanner = CorpusCorruptionScanner()
    result = scanner.scan()

    report_path = Path("datasets/final_grounding_corpus/corpus_corruption_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Scanned: {result['total_scanned']} chunks across {len(result['by_file'])} files.")
    print(f"Flagged with issues: {result['total_flagged']} ({result['flagged_percentage']}%)")
    print("Issues breakdown:")
    for k, v in result["issue_breakdown"].items():
        print(f"  - {k}: {v}")
    print(f"Full report written to: {report_path}")

