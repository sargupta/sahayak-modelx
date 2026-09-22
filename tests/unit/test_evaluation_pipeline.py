"""
Unit and Integration Tests for Quality Verification Pipeline, Judges, and Diversity Evaluator (Track 4).
"""

import asyncio
import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from synthetictutor.core.schemas import (
    Concept,
    Misconception,
    TeachingPlan,
    StudentPersona,
    TeacherPersona,
    DialogueTurn,
    DialogueRole,
    DialogueSession,
)
from synthetictutor.evaluation.math_judge import MathVerificationJudge
from synthetictutor.evaluation.locale_grounding_judge import LocaleGroundingJudge
from synthetictutor.evaluation.bengali_linguistic_judge import BengaliLinguisticJudge
from synthetictutor.evaluation.diversity_evaluator import DiversityEvaluator
from synthetictutor.evaluation.quality_pipeline import QualityVerificationPipeline, QualityReport
from synthetictutor.llm.router import MockLLMClient


class TestEvaluationPipeline(unittest.TestCase):
    def setUp(self):
        self.concept = Concept(
            id="c_pressure_01",
            name="Hydrostatic Pressure & Depth (p = h * d * g)",
            description="Liquid pressure increases proportionally with depth, fluid density, and gravitational acceleration.",
            misconceptions=[]
        )
        self.plan = TeachingPlan(
            id="plan_p1",
            target_concept=self.concept,
            learning_objective="Understand depth dependency of fluid pressure",
            starting_question="নদীর বাঁধের নিচের অংশ বেশি চওড়া কেন করা হয়?",
            max_turns=4
        )
        self.session = DialogueSession(
            id="sess_eval_01",
            plan=self.plan,
            student_persona=StudentPersona(id="sp1", name="Lakshmi Barman", grade_level="Class 8"),
            teacher_persona=TeacherPersona(id="tp1", name="Socratic Teacher"),
            turns=[
                DialogueTurn(turn_number=1, role=DialogueRole.TEACHER, content="নদীর বাঁধের নিচের অংশ উপরের চেয়ে বেশি চওড়া করা হয় কেন?"),
                DialogueTurn(turn_number=2, role=DialogueRole.STUDENT, content="নিচে জলের চাপ বেশি থাকে তাই?"),
                DialogueTurn(turn_number=3, role=DialogueRole.TEACHER, content="একদম ঠিক। গভীরে গেলে জলের চাপ কেন বাড়ে বলে তোমার মনে হয়?"),
                DialogueTurn(turn_number=4, role=DialogueRole.STUDENT, content="উপরের সমস্ত জলের স্তরের ওজন নিচের জলের উপর পড়ে, তাই চাপ বাড়ে (p = h d g)।")
            ],
            completed=True,
            language="bn"
        )

    def test_math_verification_judge(self):
        """Verify MathVerificationJudge assesses mathematical expressions and units."""
        mock_response = {
            "contains_math_or_physics": True,
            "calculations_correct": True,
            "units_consistent": True,
            "score": 0.95,
            "passed": True,
            "feedback": "Hydrostatic pressure equation p = h * d * g applied correctly."
        }
        mock_llm = MockLLMClient(predefined_responses=mock_response)
        judge = MathVerificationJudge(mock_llm)

        res = asyncio.run(judge.evaluate(self.session))
        self.assertTrue(res.passed)
        self.assertEqual(res.score, 0.95)
        self.assertEqual(res.name, "math_verification")

    def test_locale_grounding_judge(self):
        """Verify LocaleGroundingJudge cross-references regional references."""
        mock_response = {
            "contains_regional_references": True,
            "entities_factually_accurate": True,
            "quantitative_facts_verified": True,
            "score": 0.92,
            "passed": True,
            "feedback": "Sundarbans earthen dyke context matches verified physical properties in locale.json."
        }
        mock_llm = MockLLMClient(predefined_responses=mock_response)
        judge = LocaleGroundingJudge(mock_llm)

        res = asyncio.run(judge.evaluate(self.session, target_district="south24parganas"))
        self.assertTrue(res.passed)
        self.assertEqual(res.score, 0.92)
        self.assertEqual(res.name, "locale_grounding")

    def test_bengali_linguistic_judge(self):
        """Verify BengaliLinguisticJudge evaluates phrasing and persona register."""
        mock_response = {
            "is_natural_bengali": True,
            "grammatically_sound": True,
            "persona_register_aligned": True,
            "score": 0.90,
            "passed": True,
            "feedback": "Fluent, authentic Bengali phrasing with natural conversational tone."
        }
        mock_llm = MockLLMClient(predefined_responses=mock_response)
        judge = BengaliLinguisticJudge(mock_llm)

        res = asyncio.run(judge.evaluate(self.session))
        self.assertTrue(res.passed)
        self.assertEqual(res.score, 0.90)

    def test_diversity_evaluator(self):
        """Verify DiversityEvaluator calculates distinct-1, distinct-2 and self-BLEU."""
        evaluator = DiversityEvaluator()
        sessions = [
            self.session,
            DialogueSession(
                id="sess_02",
                plan=self.plan,
                student_persona=self.session.student_persona,
                teacher_persona=self.session.teacher_persona,
                turns=[
                    DialogueTurn(turn_number=1, role=DialogueRole.TEACHER, content="সুন্দরবনের মাটির বাঁধ কেন ভেঙে যায়?"),
                    DialogueTurn(turn_number=2, role=DialogueRole.STUDENT, content="জোয়ারের সময় অতিরিক্ত জল এসে ধাক্কা দেয়।")
                ],
                completed=True,
                language="bn"
            )
        ]

        metrics = evaluator.evaluate_batch_diversity(sessions)
        self.assertGreater(metrics.distinct_1, 0.0)
        self.assertGreater(metrics.distinct_2, 0.0)
        self.assertLessEqual(metrics.self_bleu_score, 1.0)
        self.assertEqual(metrics.total_dialogues_evaluated, 2)

    def test_quality_verification_pipeline_end_to_end(self):
        """Verify unified QualityVerificationPipeline aggregates scores into QualityReport."""
        mock_response = {
            "score": 0.92,
            "passed": True,
            "feedback": "Exemplary Socratic dialog and rigorous grounding.",
            "contains_math_or_physics": True,
            "calculations_correct": True,
            "units_consistent": True,
            "contains_regional_references": True,
            "entities_factually_accurate": True,
            "quantitative_facts_verified": True,
            "is_natural_bengali": True,
            "grammatically_sound": True,
            "persona_register_aligned": True
        }
        mock_llm = MockLLMClient(predefined_responses=mock_response)
        pipeline = QualityVerificationPipeline(llm_client=mock_llm)

        report: QualityReport = asyncio.run(pipeline.evaluate_dialogue(
            session=self.session,
            target_district="south24parganas"
        ))

        self.assertIsInstance(report, QualityReport)
        self.assertTrue(report.passed)
        self.assertGreaterEqual(report.composite_score, 0.85)
        self.assertTrue(report.anti_leakage_passed)
        self.assertEqual(len(report.rejection_reasons), 0)
        self.assertIn("math_verification", report.detailed_metrics)
        self.assertIn("locale_grounding", report.detailed_metrics)


if __name__ == "__main__":
    unittest.main()
