"""
Socratic Dialogue Engine for SahayakAI Synthetic Dialogue Generation.
Orchestrates multi-turn grounded dialogues between GroundedTeacherAgent and StudentAgent
with turn-by-turn learning progress tracking and early convergence detection.
"""

import uuid
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from synthetictutor.core.schemas import (
    DialogueSession,
    TeachingPlan,
    StudentPersona,
    TeacherPersona,
    DialogueTurn,
    DialogueRole,
)
from synthetictutor.simulation.grounded_teacher import GroundedTeacherAgent
from synthetictutor.simulation.student_agent import StudentAgent
from synthetictutor.simulation.intent_router import StudentMentalState
from synthetictutor.planning.personas import BenchmarkPersona, get_benchmark_persona
from synthetictutor.knowledge.locale_retriever import LocaleRetriever
from synthetictutor.knowledge.entities import EntityRegistry
from synthetictutor.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class DialogueStateAssessment(BaseModel):
    student_state: StudentMentalState = Field(..., description="Assessed student understanding state")
    concept_mastered: bool = Field(..., description="True if student has clearly articulated and understood the concept")
    dialogue_stalled: bool = Field(..., description="True if dialogue has made no progress for 2 consecutive turns")
    reasoning: str = Field(..., description="Assessment justification")


class SocraticDialogueEngine:
    """
    High-level orchestrator for generating curriculum-aligned, regionally-grounded Socratic dialogues.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        locale_retriever: Optional[LocaleRetriever] = None,
        entity_registry: Optional[EntityRegistry] = None,
    ):
        self.llm_client = llm_client
        self.locale_retriever = locale_retriever or LocaleRetriever()
        self.entity_registry = entity_registry or EntityRegistry()

    async def assess_student_state(
        self,
        plan: TeachingPlan,
        history: List[DialogueTurn],
    ) -> DialogueStateAssessment:
        """Evaluates student progress, misconception resolution, and mastery."""
        if len(history) < 2:
            return DialogueStateAssessment(
                student_state=StudentMentalState.CONFUSED,
                concept_mastered=False,
                dialogue_stalled=False,
                reasoning="Initial turn"
            )

        recent_turns = "\n".join([f"{t.role.value.upper()}: {t.content}" for t in history[-4:]])
        prompt = (
            f"Target Concept: {plan.target_concept.name} ({plan.target_concept.description})\n"
            f"Learning Objective: {plan.learning_objective}\n"
            f"Misconceptions: {[m.description for m in plan.target_misconceptions]}\n\n"
            f"Recent Dialogue:\n{recent_turns}\n\n"
            f"Evaluate the student's current understanding state, whether they have articulated mastery, or if stalled."
        )

        return await self.llm_client.generate_structured(
            prompt=prompt,
            response_schema=DialogueStateAssessment,
            temperature=0.2
        )

    async def generate_dialogue(
        self,
        plan: TeachingPlan,
        student_persona: Optional[StudentPersona] = None,
        benchmark_persona_id: Optional[str] = None,
        teacher_persona: Optional[TeacherPersona] = None,
        target_district: Optional[str] = None,
        language: str = "bn",
        max_turns: Optional[int] = None
    ) -> DialogueSession:
        """
        Executes a complete Socratic tutoring conversation session.
        """
        session_id = f"socratic_{uuid.uuid4().hex[:8]}"

        # Resolve student persona
        if benchmark_persona_id:
            bench = get_benchmark_persona(benchmark_persona_id)
            student_p = bench.to_student_persona(plan.target_misconceptions)
        elif student_persona:
            student_p = student_persona
        else:
            student_p = get_benchmark_persona("lakshmi_firstgen").to_student_persona(plan.target_misconceptions)

        # Resolve teacher persona
        teacher_p = teacher_persona or TeacherPersona(
            id="t_socratic_master",
            name="Socratic Mentor (সক্রেটিক শিক্ষক)",
            teaching_style="Guided Discovery & Grounded Scaffolding",
            scaffolding_patience=0.9,
            tone="warm, curious, encouraging, and rigorous"
        )

        teacher_agent = GroundedTeacherAgent(
            persona=teacher_p,
            llm_client=self.llm_client,
            locale_retriever=self.locale_retriever,
            entity_registry=self.entity_registry,
            target_district=target_district
        )
        student_agent = StudentAgent(student_p, self.llm_client)

        history: List[DialogueTurn] = []
        turn_limit = max_turns or plan.max_turns or 8

        # 1. Turn 1: Initial teacher question from plan
        initial_turn = DialogueTurn(
            turn_number=1,
            role=DialogueRole.TEACHER,
            content=plan.starting_question,
            inner_thought="Opening dialogue with initial Socratic question to gauge student perspective.",
            pedagogical_intent="initial_probe"
        )
        history.append(initial_turn)

        completed = False
        current_state = StudentMentalState.MISCONCEPTION_ACTIVE if plan.target_misconceptions else StudentMentalState.CONFUSED

        # Multi-turn exchange loop
        for turn_idx in range(2, turn_limit + 1):
            if turn_idx % 2 == 0:
                # Student turn
                student_turn = await student_agent.generate_turn(plan, history)
                history.append(student_turn)

                # Assess state after student answers
                assessment = await self.assess_student_state(plan, history)
                current_state = assessment.student_state

                if assessment.concept_mastered:
                    logger.info(f"Session {session_id} achieved mastery: {assessment.reasoning}")
                    completed = True
                    # Optional final affirmation turn from teacher
                    final_teacher_turn = await teacher_agent.generate_turn(
                        plan=plan,
                        history=history,
                        student_state=StudentMentalState.MASTERED,
                        language=language
                    )
                    history.append(final_teacher_turn)
                    break

                if assessment.dialogue_stalled and turn_idx >= 6:
                    logger.info(f"Session {session_id} stalled: {assessment.reasoning}")
                    break
            else:
                # Teacher turn
                teacher_turn = await teacher_agent.generate_turn(
                    plan=plan,
                    history=history,
                    student_state=current_state,
                    language=language
                )
                history.append(teacher_turn)

        return DialogueSession(
            id=session_id,
            plan=plan,
            student_persona=student_p,
            teacher_persona=teacher_p,
            turns=history,
            completed=completed,
            language=language
        )
