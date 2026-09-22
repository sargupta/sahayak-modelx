"""
Unit tests for Corpus Coverage Matrix Auditor and Thin-Cell Enricher (Track 1, Issue #15).
"""

import os
import sys
import json
import unittest
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from synthetictutor.corpus.coverage_audit import (
    CorpusCoverageAuditor,
    CoverageStatus,
    ChapterCoverageRecord
)
from synthetictutor.corpus.chunk_enricher import (
    CorpusChunkEnricher,
    to_bengali_numerals
)


class TestCorpusCoverageAudit(unittest.TestCase):
    def setUp(self):
        self.corpus_dir = Path("datasets/final_grounding_corpus")
        self.auditor = CorpusCoverageAuditor(corpus_dir=self.corpus_dir)
        self.enricher = CorpusChunkEnricher(corpus_dir=self.corpus_dir)

    def test_coverage_status_thresholds(self):
        """Test that CoverageStatus correctly reflects chapter counts."""
        rec_covered = ChapterCoverageRecord(
            board="WBBSE", grade="10", subject="Mathematics", chapter="সরল সুদকষা", chunk_count=15
        )
        self.assertEqual(rec_covered.status, CoverageStatus.COVERED)

        rec_thin = ChapterCoverageRecord(
            board="WBBSE", grade="10", subject="Mathematics", chapter="সরল সুদকষা", chunk_count=5
        )
        self.assertEqual(rec_thin.status, CoverageStatus.THIN)

        rec_missing = ChapterCoverageRecord(
            board="WBBSE", grade="10", subject="Mathematics", chapter="সরল সুদকষা", chunk_count=0
        )
        self.assertEqual(rec_missing.status, CoverageStatus.MISSING)

    def test_bengali_numeral_conversion(self):
        """Test accurate Bengali numeral normalization while preserving LaTeX."""
        text = "Class 10 Chapter 5 Page 12: $x^2 + 2x + 1 = 0$ is solved."
        converted = to_bengali_numerals(text)
        self.assertIn("Class ১০ Chapter ৫ Page ১২:", converted)
        self.assertIn("$x^2 + 2x + 1 = 0$", converted)

    def test_corpus_load_and_matrix_generation(self):
        """Test that auditor loads text-only corpus and builds 4D matrix."""
        total_chunks = self.auditor.load_corpus()
        self.assertGreater(total_chunks, 4000)

        summary = self.auditor.get_coverage_summary()
        self.assertIn("summary", summary)
        self.assertIn("distributions", summary)
        self.assertIn("by_board", summary["distributions"])
        self.assertIn("WBBSE", summary["distributions"]["by_board"])
        self.assertIn("WBBPE", summary["distributions"]["by_board"])
        self.assertIn("WBCHSE", summary["distributions"]["by_board"])

    def test_wbbse_core_curriculum_audit(self):
        """Test that all WBBSE Classes VI-X core curriculum chapters meet the >=15 threshold."""
        self.auditor.load_corpus()
        core_audit = self.auditor.audit_wbbse_core_curriculum()
        self.assertTrue(
            core_audit["all_core_curriculum_covered"],
            "All WBBSE Classes VI-X core subjects must be fully covered (>=15 chunks/chapter)."
        )

        # Specifically check Class 10 Mathematics (Ganit Prakash X)
        class10_math = core_audit["Class_10"]["Mathematics"]
        self.assertEqual(class10_math["missing_chapters"], 0)
        self.assertEqual(class10_math["thin_chapters"], 0)
        self.assertEqual(class10_math["covered_chapters"], len(CorpusCoverageAuditor.WBBSE_CORE_CURRICULUM["10"]["Mathematics"]))

    def test_manifest_v1_1_validity_and_checksums(self):
        """Test that wbbse_grounding_v1_manifest.json (v1.1) exists and has valid checksums."""
        manifest_path = self.corpus_dir / "wbbse_grounding_v1_manifest.json"
        self.assertTrue(manifest_path.exists())

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["manifest_version"], "1.1.0")
        self.assertEqual(manifest["coverage_threshold_per_chapter"], 15)
        self.assertIn("grounding_chunks_text_only.jsonl", manifest["source_files"])
        self.assertIn("sha256", manifest["source_files"]["grounding_chunks_text_only.jsonl"])
        self.assertGreater(manifest["source_files"]["grounding_chunks_text_only.jsonl"]["records"], 6000)


if __name__ == "__main__":
    unittest.main()
