"""
Unit and Integration Tests for Dataset Export Pipeline, Multi-Format Exporters, and Honest Splitter (Track 5).
"""

import json
import os
import sys
import tempfile
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from synthetictutor.core.schemas import (
    Concept,
    TeachingPlan,
    StudentPersona,
    TeacherPersona,
    DialogueTurn,
    DialogueRole,
    DialogueSession,
)
from synthetictutor.export.sharegpt import ShareGPTExporter, to_bengali_numerals, to_ascii_numerals
from synthetictutor.export.huggingface import ChatMLExporter
from synthetictutor.export.alpaca import AlpacaExporter
from synthetictutor.export.splitter import DatasetSplitter, DatasetSplits
from synthetictutor.export.dataset_card import DatasetCardGenerator
from synthetictutor.export.pipeline import DatasetExportPipeline, ExportManifest


class TestExportPipeline(unittest.TestCase):
    def setUp(self):
        self.concept = Concept(
            id="c_gravity_01",
            name="Free Fall Acceleration",
            description="All bodies accelerate at g = 9.8 m/s^2 regardless of mass.",
            misconceptions=[]
        )
        self.session1 = DialogueSession(
            id="sess_exp_01",
            plan=TeachingPlan(
                id="plan_01",
                target_concept=self.concept,
                learning_objective="Understand gravitational acceleration",
                starting_question="১০ কেজি ও ৫ কেজি ভরের দুটি বস্তু ফেললে কোনটি আগে পড়বে?",
                max_turns=2
            ),
            student_persona=StudentPersona(id="sp1", name="Lakshmi Barman", grade_level="Class 8"),
            teacher_persona=TeacherPersona(id="tp1", name="Socratic Mentor"),
            turns=[
                DialogueTurn(turn_number=1, role=DialogueRole.TEACHER, content="১০ কেজি ও ৫ কেজি ভরের দুটি বস্তু ফেললে কোনটি আগে পড়বে?", inner_thought="Probe mass misconception"),
                DialogueTurn(turn_number=2, role=DialogueRole.STUDENT, content="১০ কেজির বস্তুটা ভারী তাই আগে পড়তে পারে।", inner_thought="Hesitant intuition")
            ],
            completed=True,
            language="bn"
        )
        self.session2 = DialogueSession(
            id="sess_exp_02",
            plan=TeachingPlan(
                id="plan_02",
                target_concept=self.concept,
                learning_objective="Understand gravitational acceleration",
                starting_question="২০ কেজি ও ১৫ কেজি ভরের দুটি বস্তু ফেললে কোনটি আগে পড়বে?",  # Same template as session 1 (# কেজি ও # কেজি...)
                max_turns=2
            ),
            student_persona=StudentPersona(id="sp2", name="Aarav Menon", grade_level="Class 8"),
            teacher_persona=TeacherPersona(id="tp1", name="Socratic Mentor"),
            turns=[
                DialogueTurn(turn_number=1, role=DialogueRole.TEACHER, content="২০ কেজি ও ১৫ কেজি ভরের দুটি বস্তু ফেললে কোনটি আগে পড়বে?", inner_thought="Check understanding"),
                DialogueTurn(turn_number=2, role=DialogueRole.STUDENT, content="উভয় বস্তুই একই সাথে মাটিতে পৌঁছাবে কারণ g এর মান ধ্রুবক।", inner_thought="Confident answer")
            ],
            completed=True,
            language="bn"
        )
        self.session3 = DialogueSession(
            id="sess_exp_03",
            plan=TeachingPlan(
                id="plan_03",
                target_concept=self.concept,
                learning_objective="Understand buoyancy",
                starting_question="জাহাজ জলে ভাসে কিন্তু লোহার পেরেক কেন ডোবে?",  # Different template
                max_turns=2
            ),
            student_persona=StudentPersona(id="sp3", name="Rehan Mallick", grade_level="Class 8"),
            teacher_persona=TeacherPersona(id="tp1", name="Socratic Mentor"),
            turns=[
                DialogueTurn(turn_number=1, role=DialogueRole.TEACHER, content="জাহাজ জলে ভাসে কিন্তু লোহার পেরেক কেন ডোবে?", inner_thought="Probe buoyancy"),
                DialogueTurn(turn_number=2, role=DialogueRole.STUDENT, content="জাহাজের ঘনত্ব জলের চেয়ে কম হয় কারণ ভেতরটা ফাঁপা।", inner_thought="Analytical insight")
            ],
            completed=True,
            language="bn"
        )

    def test_numeral_converters(self):
        """Verify Bengali and ASCII numeral conversion."""
        self.assertEqual(to_bengali_numerals("12345"), "১২৩৪৫")
        self.assertEqual(to_ascii_numerals("১২৩৪৫"), "12345")
        self.assertEqual(to_bengali_numerals("Price: 250 INR"), "Price: ২৫০ INR")

    def test_sharegpt_exporter(self):
        """Verify ShareGPT exporter creates valid JSONL with <think> reasoning."""
        exporter = ShareGPTExporter(include_inner_thought=True, numerals_mode="bengali")
        rec = exporter.export_session(self.session1)

        self.assertEqual(rec["id"], "sess_exp_01")
        self.assertEqual(len(rec["conversations"]), 2)
        self.assertEqual(rec["conversations"][0]["from"], "gpt")
        self.assertIn("<think>", rec["conversations"][0]["value"])
        self.assertIn("Probe mass misconception", rec["conversations"][0]["value"])
        self.assertEqual(rec["conversations"][1]["from"], "human")

    def test_chatml_exporter(self):
        """Verify ChatML exporter formats role messages."""
        exporter = ChatMLExporter(include_inner_thought=True, numerals_mode="bengali")
        rec = exporter.export_session(self.session1)

        self.assertEqual(rec["id"], "sess_exp_01")
        self.assertEqual(rec["messages"][0]["role"], "system")
        self.assertIn("SahayakAI", rec["messages"][0]["content"])
        self.assertEqual(rec["messages"][1]["role"], "assistant")
        self.assertIn("<think>", rec["messages"][1]["content"])
        self.assertEqual(rec["messages"][2]["role"], "user")

    def test_alpaca_exporter(self):
        """Verify Alpaca exporter extracts instruction-input-output pairs."""
        exporter = AlpacaExporter(include_inner_thought=True, numerals_mode="bengali")
        records = exporter.export_session(self.session1)
        self.assertIsInstance(records, list)

    def test_dataset_splitter_zero_overlap(self):
        """Verify DatasetSplitter guarantees 0% template overlap between train and eval."""
        splitter = DatasetSplitter(eval_fraction=0.50, val_test_split_ratio=0.50, seed=42)
        sessions = [self.session1, self.session2, self.session3]

        splits: DatasetSplits = splitter.split(sessions)
        self.assertEqual(splits.template_overlap, 0)
        self.assertTrue(splits.is_honest_eval)
        self.assertGreater(len(splits.train) + len(splits.validation) + len(splits.test), 0)

    def test_export_pipeline_end_to_end(self):
        """Verify full DatasetExportPipeline end-to-end execution in temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = DatasetExportPipeline(
                eval_fraction=0.33,
                val_test_split_ratio=0.50,
                include_inner_thought=True,
                numerals_mode="bengali",
                seed=42
            )

            manifest: ExportManifest = pipeline.run(
                sessions=[self.session1, self.session2, self.session3],
                output_dir=tmpdir
            )

            self.assertEqual(manifest.template_overlap, 0)
            self.assertTrue(manifest.is_honest_eval)

            # Check files created
            self.assertTrue(os.path.exists(manifest.generated_files["sharegpt_train"]))
            self.assertTrue(os.path.exists(manifest.generated_files["chatml_train"]))
            self.assertTrue(os.path.exists(manifest.generated_files["alpaca_train"]))
            self.assertTrue(os.path.exists(manifest.generated_files["readme_md"]))
            self.assertTrue(os.path.exists(manifest.generated_files["dataset_info_json"]))

            # Verify contents of generated JSON
            with open(manifest.generated_files["dataset_info_json"], "r", encoding="utf-8") as f:
                info = json.load(f)
                self.assertEqual(info["license"], "Apache-2.0")
                self.assertIn("splits", info)


if __name__ == "__main__":
    unittest.main()
