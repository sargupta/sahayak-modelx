"""
Mathematical and Formal Reasoning Judge for Socratic Dialogues.
Verifies algebraic manipulations, arithmetic calculations, and dimensional unit consistency.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from synthetictutor.core.schemas import DialogueSession, MetricResult
from synthetictutor.llm.base import BaseLLMClient


class MathEvaluationSchema(BaseModel):
    contains_math_or_physics: bool = Field(..., description="True if dialogue involves math, calculations, or physical equations")
    calculations_correct: bool = Field(..., description="True if all arithmetic and algebraic steps are mathematically sound")
    units_consistent: bool = Field(..., description="True if units (kg, m/s^2, INR, Bigha, etc.) are dimensionally consistent")
    score: float = Field(..., ge=0.0, le=1.0, description="Mathematical accuracy score")
    passed: bool = Field(..., description="Overall pass flag (score >= 0.85)")
    feedback: str = Field(..., description="Detailed feedback on mathematical validity")


class MathVerificationJudge:
    """Evaluates mathematical, numerical, and physical equation accuracy in dialogues."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    async def evaluate(self, session: DialogueSession) -> MetricResult:
        dialogue_text = "\n".join([f"{t.role.value.upper()}: {t.content}" for t in session.turns])

        prompt = (
            f"Target Concept: {session.plan.target_concept.name}\n"
            f"Learning Objective: {session.plan.learning_objective}\n\n"
            f"Dialogue to Evaluate:\n{dialogue_text}\n\n"
            f"Evaluate the mathematical and formal reasoning:\n"
            f"1. Are all equations, substitutions, and arithmetic steps correct?\n"
            f"2. Are units and dimensional analyses accurate?\n"
            f"3. If no math is present, rate 1.0 (pass).\n"
        )
        system = "You are an expert mathematical and physics verification auditor. Rigorously check calculations, formulas, and units."

        res: MathEvaluationSchema = await self.llm_client.generate_structured(
            prompt=prompt,
            response_schema=MathEvaluationSchema,
            system_instruction=system,
            temperature=0.1
        )

        return MetricResult(
            name="math_verification",
            score=res.score,
            passed=res.passed and res.score >= 0.85,
            feedback=res.feedback
        )
