"""
Corpus Coverage & Distribution Auditor for SahayakAI Grounding Corpus.
Computes 4D coverage matrix (board x grade x subject x chapter) against the >=15 chunks/chapter threshold.
Generates corpus_distribution_audit.json and validates wbbse_grounding_v1_manifest.json (v1.1).
"""

import os
import json
import hashlib
from enum import Enum
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any, List, Optional, Tuple, Set


class CoverageStatus(str, Enum):
    COVERED = "COVERED"    # >= 15 text-only chunks per chapter
    THIN = "THIN"          # 1 to 14 text-only chunks per chapter
    MISSING = "MISSING"    # 0 chunks


class ChapterCoverageRecord:
    def __init__(
        self,
        board: str,
        grade: str,
        subject: str,
        chapter: str,
        chunk_count: int = 0,
        content_types: Optional[List[str]] = None
    ):
        self.board = board
        self.grade = str(grade)
        self.subject = subject
        self.chapter = chapter
        self.chunk_count = chunk_count
        self.content_types = content_types or []

    @property
    def status(self) -> CoverageStatus:
        if self.chunk_count >= 15:
            return CoverageStatus.COVERED
        elif self.chunk_count > 0:
            return CoverageStatus.THIN
        return CoverageStatus.MISSING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "board": self.board,
            "grade": self.grade,
            "subject": self.subject,
            "chapter": self.chapter,
            "chunk_count": self.chunk_count,
            "coverage_status": self.status.value,
            "content_types": list(set(self.content_types))
        }


class CorpusCoverageAuditor:
    """
    4D Coverage Matrix Auditor for SahayakAI Grounding Corpus.
    """
    THRESHOLD_CHUNK_PER_CHAPTER = 15

    # Core required syllabus chapters for WBBSE Classes VI-X
    WBBSE_CORE_CURRICULUM = {
        "10": {
            "Mathematics": [
                "একচলবিশিষ্ট দ্বিঘাত সমীকরণ", "সরল সুদকষা", "বৃত্ত সম্পর্কিত উপপাদ্য",
                "আয়তঘন", "অনুপাত ও সমানুপাত", "চক্রবৃদ্ধি সুদ ও সমাহার বৃদ্ধি বা হ্রাস",
                "বৃত্তস্থ কোণ সম্পর্কিত উপপাদ্য", "লম্ব বৃত্তাকার চোঙ", "দ্বিঘাত করণী",
                "বৃত্তস্থ চতুর্ভুজ সংক্রান্ত উপপাদ্য", "সম্পাদ্য: ত্রিভুজের পরিবৃত্ত ও অন্তর্বৃত্ত অঙ্কন",
                "গোলক", "ভেদ", "অংশীদারি কারবার", "বৃত্তের স্পর্শক সংক্রান্ত উপপাদ্য",
                "লম্ব বৃত্তাকার শঙ্কু", "সম্পাদ্য: বৃত্তের স্পর্শক অঙ্কন", "সদৃশতা",
                "বিভিন্ন ঘনবস্তু সংক্রান্ত বাস্তব সমস্যা", "ত্রিকোণমিতি: কোণ পরিমাপন ধারণা",
                "সম্পাদ্য: মধ্যসমানুপাতী নির্ণয়", "পীথাগোরাসের উপপাদ্য",
                "ত্রিকোণমিতিক অনুপাত এবং অভেদাবলী", "পূরক কোণের ত্রিকোণমিতিক অনুপাত",
                "ত্রিকোণমিতিক দূরত্বের সমস্যা (উচ্চতা ও দূরত্ব)", "রাশিবিজ্ঞান (গড়, মধ্যমা, ওজাইভ, সংখ্যাগুরুমান)"
            ],
            "Physical Science": [
                "পরিবেশের জন্য ভাবনা", "গ্যাসের আচরণ", "রাসায়নিক গণনা",
                "তাপের ঘটনাসমূহ", "আলো", "চলতড়িৎ", "পরমাণুর নিউক্লিয়াস",
                "পর্যায় সারণি ও মৌলদের ধর্মের পর্যায়বৃত্ততা", "আয়নীয় ও সমযোজী বন্ধন",
                "তড়িৎপ্রবাহ ও রাসায়নিক বিক্রিয়া", "অজৈব রসায়ন", "ধাতুবিদ্যা", "জৈব রসায়ন"
            ],
            "Life Science": [
                "জীবজগতের নিয়ন্ত্রণ ও সমন্বয়", "জীবনের প্রবাহমানতা", "বংশগতি এবং কয়েকটি সাধারণ জিনগত রোগ",
                "অভিব্যক্তি ও অভিযোজন", "পরিবেশ, তার সম্পদ এবং তাদের সংরক্ষণ"
            ],
            "History": [
                "ইতিহাসের ধারণা", "সংস্কার: বৈশিষ্ট্য ও পর্যালোচনা", "প্রতিরোধ ও বিদ্রোহ",
                "সংঘবদ্ধতার গোড়ার কথা", "বিকল্প চিন্তা ও উদ্যোগ", "বিশ শতকের ভারতে কৃষক, শ্রমিক ও বামপন্থী আন্দোলন",
                "বিশ শতকের ভারতে নারী, ছাত্র ও প্রান্তিক জনগোষ্ঠীর আন্দোলন", "উত্তর-ঔপনিবেশিক ভারত"
            ],
            "Geography": [
                "বহির্জাত প্রক্রিয়া ও তাদের দ্বারা সৃষ্ট ভূমিরূপ", "বায়ুমণ্ডল", "বারিমণ্ডল",
                "বর্জ্য ব্যবস্থাপনা", "ভারত: প্রাকৃতিক ও অর্থনৈতিক পরিবেশ", "উপগ্রহ চিত্র ও ভূবৈচিত্র্যসূচক মানচিত্র"
            ]
        },
        "9": {
            "Mathematics": [
                "বাস্তব সংখ্যা", "বহুপদী রাশিমালা", "লেখচিত্র", "স্থানাঙ্ক জ্যামিতি",
                "রৈখিক সহসমীকরণ", "সামান্তরিকের ধর্ম", "ক্ষেত্রফল সংক্রান্ত উপপাদ্য",
                "রাশিবিজ্ঞান", "বৃত্তের পরিধি ও ক্ষেত্রফল"
            ],
            "Physical Science": [
                "পরিমাপ", "বল ও গতি", "পরমাণুর গঠন ও পদার্থের ভৌত ও রাসায়নিক ধর্মসমূহ",
                "দ্রবণ", "কার্য, ক্ষমতা ও শক্তি", "শব্দ"
            ],
            "Life Science": [
                "জীবন ও তার বৈচিত্র্য", "জীবন সংগঠনের স্তর", "জৈবনিক প্রক্রিয়া",
                "জীববিদ্যা ও মানবকল্যাণ", "পরিবেশ ও তার সম্পদ"
            ],
            "History": [
                "ফরাসি বিপ্লবের কয়েকটি দিক", "বিপ্লবী আদর্শের সংঘাত ও নেপোলিয়ন", "উনিশ শতকের ইউরোপ",
                "শিল্পবিপ্লব ও জাতীয়তাবাদ", "প্রথম বিশ্বযুদ্ধ ও পরবর্তীকাল", "দ্বিতীয় বিশ্বযুদ্ধ"
            ],
            "Geography": [
                "গ্রহরূপে পৃথিবী", "পৃথিবীর গতিসমূহ", "ভূপৃষ্ঠে কোনো স্থানের অবস্থান নির্ণয়",
                "ভূমিরূপ গঠনকারী প্রক্রিয়া ও ভূমিরূপ", "আবহাওয়া বিকার", "পশ্চিমবঙ্গ: প্রাকৃতিক ও সম্পদ"
            ]
        },
        "8": {
            "Mathematics": ["মূলদ সংখ্যা", "পাই চিত্র", "বীজগাণিতিক সংখ্যামালার গুণ ও ভাগ", "ঘনফল নির্ণয়", "ত্রিভুজের কোণ ও বাহুর সম্পর্ক"],
            "Science": ["বল ও চাপ", "স্পর্শ ছাড়া ক্রিয়াশীল বল", "পদার্থের প্রকৃতি", "রাসায়নিক বিক্রিয়া", "মানবদেহ ও খাদ্য"],
            "History": ["মুঘল সাম্রাজ্যের পতন", "ঔপনিবেশিক কর্তৃত্ব প্রতিষ্ঠা", "ঔপনিবেশিক অর্থনীতি", "জাতীয়তাবাদের উন্মেষ"],
            "Geography": ["অশ্মমণ্ডল", "বায়ুমণ্ডল", "জলমণ্ডল", "ভারতের প্রতিবেশী দেশসমূহ", "মানুষের কার্যাবলী ও পরিবেশ"]
        },
        "7": {
            "Mathematics": ["পূর্বপাঠের পুনরালোচনা", "অনুপাত", "ভগ্নাংশের বর্গমূল", "বীজগাণিতিক সূত্রাবলী", "ত্রিভুজ অঙ্কন"],
            "Science": ["তাপ", "আলো", "চুম্বক", "তড়িৎ", "পরিবেশ ও উদ্ভিদ জগত", "মানুষের খাদ্য"],
            "History": ["ভারতের রাজনৈতিক ইতিহাস", "সুলতানি আমল", "মুঘল শাসন ব্যবস্থা", "সংস্কৃতি ও ধর্ম"],
            "Geography": ["ভূপৃষ্ঠের পরিবর্তন", "বায়ুর চাপ ও বায়ুপ্রবাহ", "নদী ও জলধারা", "ইউরোপ ও আফ্রিকা"]
        },
        "6": {
            "Mathematics": ["অঙ্কপাতন ও স্থানীয় মান", "রোমান সংখ্যা", "ভগ্নাংশ ও দশমিক", "লসাগু ও গসাগু", "জ্যামিতিক ধারণা"],
            "Science": ["আমাদের পরিবেশ ও জীবজগত", "পদার্থের রূপান্তর", "গতি ও শক্তি", "শব্দ ও আলো", "বাসস্থান ও পরিবেশ"],
            "History": ["আদিম মানুষ", "সিন্ধু সভ্যতা", "বৈদিক যুগ", "জনপদ ও মহাজনপদ", "মৌর্য ও গুপ্ত সাম্রাজ্য"],
            "Geography": ["আকাশ ও সৌরজগৎ", "পৃথিবীর আকার", "মানচিত্র শিক্ষা", "ভারতের ভূপ্রকৃতি ও জলবায়ু"]
        }
    }

    def __init__(self, corpus_dir: Optional[Path] = None):
        self.corpus_dir = Path(corpus_dir or "datasets/final_grounding_corpus")
        self.text_only_path = self.corpus_dir / "grounding_chunks_text_only.jsonl"
        self.visual_context_path = self.corpus_dir / "grounding_chunks_visual_context.jsonl"
        self.audit_json_path = self.corpus_dir / "corpus_distribution_audit.json"
        self.manifest_json_path = self.corpus_dir / "wbbse_grounding_v1_manifest.json"

        self.records: List[Dict[str, Any]] = []
        self.matrix: Dict[str, Dict[str, Dict[str, Dict[str, ChapterCoverageRecord]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )

    def load_corpus(self) -> int:
        """Loads text-only grounding chunks from jsonl."""
        self.records = []
        if not self.text_only_path.exists():
            return 0

        with open(self.text_only_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        self.records.append(json.loads(line_str))
                    except Exception:
                        pass

        self._build_matrix()
        return len(self.records)

    def _build_matrix(self):
        """Indexes all loaded records into 4D matrix."""
        self.matrix.clear()
        for r in self.records:
            board = r.get("board", "WBBSE")
            grade = str(r.get("grade", ""))
            subject = r.get("subject", "General")
            chapter = r.get("chapter", "General")
            ctype = r.get("content_type", "CONCEPT_EXPLANATION")

            if chapter not in self.matrix[board][grade][subject]:
                self.matrix[board][grade][subject][chapter] = ChapterCoverageRecord(
                    board=board,
                    grade=grade,
                    subject=subject,
                    chapter=chapter,
                    chunk_count=0,
                    content_types=[]
                )

            rec = self.matrix[board][grade][subject][chapter]
            rec.chunk_count += 1
            rec.content_types.append(ctype)

    def get_coverage_summary(self) -> Dict[str, Any]:
        """Calculates global coverage statistics, thin cells, and chapter metrics."""
        total_chunks = len(self.records)
        by_board = Counter()
        by_grade = Counter()
        by_subject = Counter()
        by_content_type = Counter()
        by_status = Counter()

        thin_cells = []
        covered_chapters = 0
        total_chapters = 0

        for board, grades in self.matrix.items():
            for grade, subjects in grades.items():
                for subject, chapters in subjects.items():
                    for chapter, record in chapters.items():
                        total_chapters += 1
                        by_board[board] += record.chunk_count
                        by_grade[grade] += record.chunk_count
                        by_subject[subject] += record.chunk_count
                        by_status[record.status.value] += 1

                        if record.status == CoverageStatus.COVERED:
                            covered_chapters += 1
                        else:
                            thin_cells.append(record.to_dict())

                        for ct in record.content_types:
                            by_content_type[ct] += 1

        # Check WBBSE VI-X Core syllabus coverage
        core_audit = self.audit_wbbse_core_curriculum()

        return {
            "summary": {
                "total_text_only_chunks": total_chunks,
                "total_indexed_chapters": total_chapters,
                "covered_chapters_count": covered_chapters,
                "thin_or_missing_chapters_count": total_chapters - covered_chapters,
                "chapter_coverage_percentage": round((covered_chapters / max(1, total_chapters)) * 100, 2),
                "threshold_chunks_per_chapter": self.THRESHOLD_CHUNK_PER_CHAPTER
            },
            "distributions": {
                "by_board": dict(by_board),
                "by_grade": dict(sorted(by_grade.items(), key=lambda x: str(x[0]))),
                "by_subject": dict(sorted(by_subject.items(), key=lambda x: -x[1])),
                "by_content_type": dict(sorted(by_content_type.items(), key=lambda x: -x[1])),
                "by_coverage_status": dict(by_status)
            },
            "wbbse_core_curriculum_audit": core_audit,
            "thin_cells": thin_cells
        }

    def audit_wbbse_core_curriculum(self) -> Dict[str, Any]:
        """Audits WBBSE Class VI-X core syllabus coverage status."""
        results = {}
        all_passed = True

        for grade, subjects in self.WBBSE_CORE_CURRICULUM.items():
            results[f"Class_{grade}"] = {}
            for subject, required_chapters in subjects.items():
                subject_status = {
                    "required_chapters": len(required_chapters),
                    "covered_chapters": 0,
                    "thin_chapters": 0,
                    "missing_chapters": 0,
                    "chapters": {}
                }
                for ch in required_chapters:
                    rec = self.matrix.get("WBBSE", {}).get(grade, {}).get(subject, {}).get(ch)
                    if rec:
                        status = rec.status
                        count = rec.chunk_count
                    else:
                        status = CoverageStatus.MISSING
                        count = 0

                    if status == CoverageStatus.COVERED:
                        subject_status["covered_chapters"] += 1
                    elif status == CoverageStatus.THIN:
                        subject_status["thin_chapters"] += 1
                        all_passed = False
                    else:
                        subject_status["missing_chapters"] += 1
                        all_passed = False

                    subject_status["chapters"][ch] = {
                        "count": count,
                        "status": status.value
                    }
                results[f"Class_{grade}"][subject] = subject_status

        results["all_core_curriculum_covered"] = all_passed
        return results

    def generate_audit_json(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generates and writes corpus_distribution_audit.json."""
        out_file = output_path or self.audit_json_path
        audit_data = self.get_coverage_summary()

        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(audit_data, f, ensure_ascii=False, indent=2)

        return audit_data

    def generate_manifest_v1_1(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generates updated wbbse_grounding_v1_manifest.json with v1.1.0 and SHA-256."""
        out_file = output_path or self.manifest_json_path

        source_files_meta = {}
        for fname, desc in [
            ("grounding_chunks_text_only.jsonl", "All-board text-only grounding chunks with filled thin cells (v1.1)"),
            ("grounding_chunks_visual_context.jsonl", "Visual grounding chunks with paired bounding boxes"),
            ("wbbse_sft_source_pool.jsonl", "Frozen verified WBBSE textbook text chunks across Classes 6-10"),
            ("excluded_archive.jsonl", "Quarantined noisy or excluded structural units")
        ]:
            fpath = self.corpus_dir / fname
            if fpath.exists():
                sha256_hash = hashlib.sha256()
                rec_count = 0
                with open(fpath, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        sha256_hash.update(chunk)
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            rec_count += 1

                source_files_meta[fname] = {
                    "records": rec_count,
                    "sha256": sha256_hash.hexdigest(),
                    "description": desc
                }

        manifest = {
            "manifest_version": "1.1.0",
            "corpus_name": "wbbse_grounding_v1",
            "updated_at": "2026-09-22T12:00:00Z",
            "coverage_threshold_per_chapter": self.THRESHOLD_CHUNK_PER_CHAPTER,
            "source_files": source_files_meta
        }

        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        return manifest
