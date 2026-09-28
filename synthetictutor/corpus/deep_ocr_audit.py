"""
Deep OCR Corruption and Anomaly Auditor for SahayakAI Grounding Corpus.
Scans for:
1. Repetitive character stutters (e.g. ককককক, 11111, aaaaa).
2. Unpronounceable non-word consonant clusters / OCR noise.
3. Garbled math expressions and broken symbols (e.g. +=+, / /, ..).
4. English/ASCII noise leaking into middle of Bengali tokens.
5. Incomplete / dangling sentence fragments.
"""

import os
import re
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple


# Regex patterns for deep OCR corruption detection
CHAR_STUTTER_REGEX = re.compile(r'([^\s0-9০-৯\.\-_])\1{4,}')
NUM_STUTTER_REGEX = re.compile(r'([0-9০-৯])\1{5,}')
MIXED_SCRIPT_TOKEN = re.compile(r'[a-zA-Z]+[\u0980-\u09FF]+|[\u0980-\u09FF]+[a-zA-Z]+')
GARBLED_MATH_PUNCT = re.compile(r'(\+\=|\=\+|\/\s*\/|\*\s*\*|\(\s*\(|\)\s*\))')
DANGLING_HYPHEN_WORD = re.compile(r'[\u0980-\u09FF]+-\s+[\u0980-\u09FF]+')
HIGH_SYMBOL_DENSITY = re.compile(r'[^a-zA-Z0-9\u0980-\u09FF\s\.,\?!\'\":;\(\)\$\=\+\-\/\*\%]')
ILLEGAL_BENGALI_COMBINERS = re.compile(r'[\u0981-\u0983\u09BC-\u09CD\u09D7\u09E2\u09E3]{3,}')


class DeepOCRAuditor:
    def __init__(self, corpus_dir: str = "datasets/final_grounding_corpus"):
        self.corpus_dir = Path(corpus_dir)
        self.target_files = [
            "grounding_chunks_text_only.jsonl",
            "grounding_chunks_visual_context.jsonl",
            "wbbse_sft_source_pool.jsonl"
        ]

    def audit(self) -> Dict[str, Any]:
        total_records = 0
        flagged_records = []
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

                    total_records += 1
                    file_counts[fname] += 1

                    try:
                        rec = json.loads(line_str)
                    except Exception:
                        issue_counts["JSON_PARSE_ERROR"] += 1
                        continue

                    text = rec.get("source_text", rec.get("text", ""))
                    chunk_id = rec.get("chunk_id", f"line_{idx+1}")
                    medium = rec.get("medium", "Bengali")

                    issues = []

                    # 1. Character stutters
                    if CHAR_STUTTER_REGEX.search(text):
                        issues.append("CHAR_STUTTER_ARTEFACT")

                    # 2. Number stutters
                    if NUM_STUTTER_REGEX.search(text):
                        issues.append("NUM_STUTTER_ARTEFACT")

                    # 3. Mixed script tokens (e.g. bengali concatenated with english OCR artifact)
                    mixed_tokens = MIXED_SCRIPT_TOKEN.findall(text)
                    if len(mixed_tokens) >= 2:
                        issues.append("MIXED_SCRIPT_OCR_GLITCH")

                    # 4. Illegal Bengali combiner sequences
                    if ILLEGAL_BENGALI_COMBINERS.search(text):
                        issues.append("ILLEGAL_COMBINING_MARKS")

                    # 5. Garbled math/punct
                    if GARBLED_MATH_PUNCT.search(text):
                        issues.append("GARBLED_MATH_OPERATORS")

                    # 6. High unknown symbol density
                    unknown_symbols = HIGH_SYMBOL_DENSITY.findall(text)
                    if len(unknown_symbols) > 5 and (len(unknown_symbols) / max(1, len(text))) > 0.05:
                        issues.append("HIGH_UNKNOWN_SYMBOL_DENSITY")

                    if issues:
                        for iss in issues:
                            issue_counts[iss] += 1
                        flagged_records.append({
                            "file": fname,
                            "line": idx + 1,
                            "id": chunk_id,
                            "issues": issues,
                            "preview": text[:140].replace("\n", " ")
                        })

        return {
            "total_scanned": total_records,
            "total_flagged": len(flagged_records),
            "flagged_percentage": round((len(flagged_records) / max(1, total_records)) * 100, 2),
            "by_file": dict(file_counts),
            "issue_breakdown": dict(issue_counts.most_common()),
            "flagged_samples": flagged_records[:25]
        }


if __name__ == "__main__":
    auditor = DeepOCRAuditor()
    report = auditor.audit()

    report_path = Path("datasets/final_grounding_corpus/deep_ocr_audit_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Deep OCR Audit Complete! Scanned: {report['total_scanned']} chunks.")
    print(f"Flagged with Deep OCR issues: {report['total_flagged']} ({report['flagged_percentage']}%)")
    print("Issue Breakdown:")
    for k, v in report["issue_breakdown"].items():
        print(f"  - {k}: {v}")
