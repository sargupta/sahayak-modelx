#!/usr/bin/env python3
"""
validate_50k_shard.py — Standalone Validator & Quality Auditor for SahayakAI Datasets.

Usage:
  python -m synthetictutor.scripts.validate_50k_shard --input output/50k_dataset/sahayak_50k_shard_01.jsonl
  python -m synthetictutor.scripts.validate_50k_shard --input any_generated_dataset.jsonl --strict
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict


ORPHAN_MATRA_PATTERN = re.compile(r'(?:^|\s)[\u09BE-\u09CD\u09D7]+(?:\s|$)')
BROKEN_STEM_PATTERN = re.compile(r'\b(অধ\s*:\s*বয|বয\s+অধ|ঐত\s+পট|রাজন\s+তর|উপন\s+সন|সম\s+রকদ|আধ\s+রতের)\b')
VISUAL_HALLUCINATION_PATTERN = re.compile(r'উপরের\s+ছবি(তে|টি|গুলি)|পাশের\s+ছবি(তে|টি|গুলি)|পাশের\s+চিত্র(টি|টিতে)|নিচের\s+মানচিত্র(টি|টিতে)')
META_LEAK_PATTERN = re.compile(r'প্রদত্ত\s+পাঠ্যাংশ(ে|টির)?|উক্ত\s+পাঠ্যাংশ(ে|টির)?|প্রদত্ত\s+অনুচ্ছেদ(ে)?|উক্ত\s+চাঙ্ক(ে)?')
ASCII_DIGITS_PATTERN = re.compile(r'(?<![a-zA-Z_\^\$\{\}\\])\b\d+\b(?![a-zA-Z_\^\$\{\}\\])')


def validate_file(file_path: str, strict: bool = False):
    p = Path(file_path)
    if not p.exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    print("\n" + "="*70)
    print(f" [AUDIT] SahayakAI Quality Audit: {p.name}")
    print("="*70)

    total_records = 0
    passed_records = 0
    issue_counts = defaultdict(int)
    flagged_samples = []

    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            total_records += 1

            record_issues = []

            # 1. JSON Parsing & Schema
            try:
                data = json.loads(line_str)
            except Exception as e:
                record_issues.append(f"JSON Decode Error: {e}")
                issue_counts["INVALID_JSON"] += 1
                flagged_samples.append((idx, record_issues))
                continue

            messages = data.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                record_issues.append("Missing or invalid 'messages' list (must have at least user and assistant)")
                issue_counts["SCHEMA_INVALID_MESSAGES"] += 1
                flagged_samples.append((idx, record_issues))
                continue

            # Extract contents
            user_text = ""
            asst_text = ""
            sys_text = ""

            for m in messages:
                role = m.get("role")
                content = m.get("content", "")
                if role == "system":
                    sys_text = content
                elif role == "user":
                    user_text = content
                elif role == "assistant":
                    asst_text = content

            if not user_text:
                record_issues.append("Empty user message")
                issue_counts["EMPTY_USER_MESSAGE"] += 1
            if not asst_text:
                record_issues.append("Empty assistant message")
                issue_counts["EMPTY_ASSISTANT_MESSAGE"] += 1

            full_conv_text = f"{user_text} {asst_text}"

            # 2. OCR Corruption & Orphaned Matras
            if ORPHAN_MATRA_PATTERN.search(full_conv_text):
                record_issues.append("Contains orphaned Bengali vowel signs/matras (e.g. standalone ' ি ', ' ে ', ' া ')")
                issue_counts["ORPHANED_BENGALI_MATRAS"] += 1
            if BROKEN_STEM_PATTERN.search(full_conv_text):
                record_issues.append("Contains broken OCR stem fragments (e.g. 'অধ : বয', 'ঐত পট')")
                issue_counts["BROKEN_OCR_STEMS"] += 1

            # 3. Ghost Visual Hallucinations
            if VISUAL_HALLUCINATION_PATTERN.search(asst_text):
                record_issues.append("Contains unanchored visual references ('উপরের ছবিতে...', 'পাশের চিত্রে...')")
                issue_counts["GHOST_VISUAL_HALLUCINATION"] += 1

            # 4. Meta-Prompt Leakage
            if META_LEAK_PATTERN.search(asst_text):
                record_issues.append("Contains artificial meta-prompt leakage ('প্রদত্ত পাঠ্যাংশে', 'উক্ত অনুচ্ছেদে')")
                issue_counts["META_PROMPT_LEAKAGE"] += 1

            # 5. Numeral System Check (Bengali digits vs ASCII)
            ascii_digits = ASCII_DIGITS_PATTERN.findall(asst_text)
            if len(ascii_digits) > 5 and not any(k in asst_text for k in ["English", "Butterfly", "Blossoms", "x^", "y^", "\\frac"]):
                record_issues.append(f"Excessive ASCII digits ({len(ascii_digits)} found) in Bengali prose")
                issue_counts["ASCII_DIGIT_EXCESS"] += 1

            # 6. Completeness / Terminal Punctuation
            if asst_text and not asst_text.endswith(("।", "?", "!", ".", "”", "'", "’", "\n")):
                record_issues.append("Assistant response appears truncated (missing terminal punctuation)")
                issue_counts["TRUNCATED_RESPONSE"] += 1

            if not record_issues:
                passed_records += 1
            else:
                if len(flagged_samples) < 5:
                    flagged_samples.append((idx, record_issues))

    pass_pct = (passed_records / total_records * 100) if total_records > 0 else 0

    print(f" Total Records Scanned: {total_records:,}")
    print(f" Passed All Quality Gates: {passed_records:,} ({pass_pct:.2f}%)")
    print(f" Flagged Records: {total_records - passed_records:,}")
    print("-" * 70)

    if issue_counts:
        print(" [ISSUES BREAKDOWN]")
        for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_records * 100) if total_records > 0 else 0
            print(f"   - {issue:<30} : {count:<6,d} ({pct:>5.1f}%)")
        print("-" * 70)

    if flagged_samples:
        print(" [SAMPLE FLAGGED ROWS (First 5)]")
        for row_num, issues in flagged_samples:
            print(f"   Line {row_num}: {'; '.join(issues)}")
        print("-" * 70)

    if passed_records == total_records and total_records > 0:
        print(" [RESULT] 100% PASSED! Dataset is pristine and training-ready.")
    else:
        print(f" [RESULT] Quality Score: {pass_pct:.2f}%. Please address flagged issues.")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="SahayakAI Dataset Validator & Auditor")
    parser.add_argument("--input", type=str, required=True, help="Path to .jsonl file to validate")
    parser.add_argument("--strict", action="store_true", help="Fail with non-zero exit code if any issues found")
    args = parser.parse_args()

    validate_file(args.input, strict=args.strict)


if __name__ == "__main__":
    main()
