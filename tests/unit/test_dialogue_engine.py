"""
Unit and Integration Tests for Socratic Dialogue Engine, Intent Router, and Benchmark Personas (Track 3).
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
    TeacherPersona,
    DialogueTurn,
    DialogueRole,
    DialogueSession,
)
from synthetictutor.simulation.intent_router import (
    IntentRouter,
    PedagogicalIntent,
    StudentMentalState,
)
from synthetictutor.planning.personas import (
    BENCHMARK_PERSONAS,
    get_benchmark_persona,
    get_all_benchmark_personas,
)
from synthetictutor.simulation.grounded_teacher import GroundedTeacherAgent
from synthetictutor.simulation.dialogue_engine import SocraticDialogueEngine
from synthetictutor.llm.router import MockLLMClient


class TestDialogueEngineAndRouter(unittest.TestCase):
    def setUp(self):
        self.router = IntentRouter()
        self.m1 = Misconception(
            id="misc_heavier_falls_faster",
            description="Heavier objects fall faster due to having more mass",
            typical_trigger="Dropping a stone and a feather",
            correct_conception="All masses experience identical gravitational acceleration in vacuum (g = 9.8 m/s^2)"
        )
        self.concept = Concept(
            id="concept_gravity_01",
            name="Universal Gravitation & Free Fall",
            description="Gravitational acceleration independent of falling mass in vacuum",
            misconceptions=[self.m1]
        )
        self.plan = TeachingPlan(
            id="plan_01",
            target_concept=self.concept,
            target_misconceptions=[self.m1],
            learning_objective="Deduce that gravitational acceleration is independent of mass",
            starting_question="একটি ভারী পাথর ও একটি হালকা কাঠ একই সাথে ফেলা হলে কোনটি আগে পড়বে?",
            max_turns=6
        )

    def test_intent_router_misconception_active(self):
        """Verify router produces counter-example or probe when misconception is active."""
        history = [
            DialogueTurn(turn_number=1, role=DialogueRole.TEACHER, content="Which falls faster?"),
            DialogueTurn(turn_number=2, role=DialogueRole.STUDENT, content="The heavy stone will fall faster because it has more weight.")
        ]
        decision = self.router.route_intent(
            history=history,
            active_misconceptions=[self.m1],
            student_state=StudentMentalState.MISCONCEPTION_ACTIVE
        )
        self.assertIn(
            decision.intent,
            [PedagogicalIntent.COUNTER_EXAMPLE, PedagogicalIntent.PROBE_MISCONCEPTION]
        )
        self.assertTrue(decision.use_local_analogy or decision.target_misconception_id)

    def test_intent_router_confusion_and_breakthrough(self):
        """Verify analogical bridge on confusion, and verify_reasoning on breakthrough."""
        # Test Confusion
        decision_confused = self.router.route_intent(
            history=[],
            active_misconceptions=[],
            student_state=StudentMentalState.CONFUSED
        )
        self.assertEqual(decision_confused.intent, PedagogicalIntent.ANALOGICAL_BRIDGE)
        self.assertTrue(decision_confused.use_local_analogy)

        # Test Breakthrough
        decision_bt = self.router.route_intent(
            history=[],
            active_misconceptions=[],
            student_state=StudentMentalState.BREAKTHROUGH
        )
        self.assertEqual(decision_bt.intent, PedagogicalIntent.VERIFY_REASONING)

    def test_benchmark_personas_integrity(self):
        """Verify all 5 benchmark personas are present with complete metadata."""
        personas = get_all_benchmark_personas()
        self.assertEqual(len(personas), 5)

        expected_ids = ["aarav_topper", "lakshmi_firstgen", "rehan_curious", "meera_rote", "karthik_distracted"]
        for pid in expected_ids:
            p = get_benchmark_persona(pid)
            self.assertEqual(p.id, pid)
            self.assertGreater(len(p.name_bengali), 2)
            self.assertGreater(len(p.voice_signature), 10)
            self.assertGreater(len(p.error_profile), 10)
            self.assertGreater(len(p.bengali_register_notes), 5)

            # Test conversion to StudentPersona
            sp = p.to_student_persona([self.m1])
            self.assertEqual(sp.name, p.name)
            self.assertEqual(len(sp.active_misconceptions), 1)

    def test_grounded_teacher_turn_generation(self):
        """Verify teacher generates turn with inner_thought and pedagogical intent."""
        mock_response = {
            "inner_thought": "Student believes mass causes faster fall. I will present a vacuum tube counter-example.",
            "pedagogical_intent": "counter_example",
            "response_text": "যদি আমরা বাতাসশূন্য একটি কাচের নলে দুটি জিনিস ফেলি, তাহলে কী ঘটবে বলে তোমার মনে হয়?",
            "scaffold_level": 2,
            "grounding_entity_used": "wb_ent_kheya_nouka"
        }
        mock_llm = MockLLMClient(predefined_responses=mock_response)
        teacher = GroundedTeacherAgent(
            persona=TeacherPersona(id="t1", name="Socratic Teacher", teaching_style="Socratic Elicitation"),
            llm_client=mock_llm,
            target_district="purulia"
        )

        history = [
            DialogueTurn(turn_number=1, role=DialogueRole.TEACHER, content=self.plan.starting_question),
            DialogueTurn(turn_number=2, role=DialogueRole.STUDENT, content="ভারী পাথরটাই আগে পড়বে।")
        ]

        turn = asyncio.run(teacher.generate_turn(
            plan=self.plan,
            history=history,
            student_state=StudentMentalState.MISCONCEPTION_ACTIVE,
            language="bn"
        ))

        self.assertEqual(turn.role, DialogueRole.TEACHER)
        self.assertEqual(turn.turn_number, 3)
        self.assertIn("বাতাসশূন্য", turn.content)
        self.assertIsNotNone(turn.inner_thought)
        self.assertEqual(turn.pedagogical_intent, "counter_example")

    def test_dialogue_engine_end_to_end_simulation(self):
        """Verify full multi-turn simulation lifecycle."""
        mock_responses = {
            "inner_thought": "Reasoning step",
            "pedagogical_intent": "counter_example",
            "response_text": "বাতাসে পালক বাধা পায়, কিন্তু ভ্যাকুয়ামে দুটিই একসাথে পড়বে।",
            "understanding_state": "breakthrough",
            "student_state": "breakthrough",
            "concept_mastered": True,
            "dialogue_stalled": False,
            "reasoning": "Student realized air resistance was the differentiating factor."
        }
        mock_llm = MockLLMClient(predefined_responses=mock_responses)
        engine = SocraticDialogueEngine(llm_client=mock_llm)

        session: DialogueSession = asyncio.run(engine.generate_dialogue(
            plan=self.plan,
            benchmark_persona_id="lakshmi_firstgen",
            target_district="coochbehar",
            language="bn",
            max_turns=4
        ))

        self.assertIsInstance(session, DialogueSession)
        self.assertGreaterEqual(len(session.turns), 2)
        self.assertTrue(session.completed)
        self.assertEqual(session.student_persona.name, "Lakshmi Barman")


if __name__ == "__main__":
    unittest.main()
