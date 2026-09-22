"""
Bengali Linguistic Register and Naturalness Judge for Synthetic Dialogues.
Evaluates grammar, natural idioms, code-mixing appropriateness, and alignment with target student persona registers.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from synthetictutor.core.schemas import DialogueSession, MetricResult
from synthetictutor.llm.base import BaseLLMClient


class BengaliLinguisticEvaluationSchema(BaseModel):
    is_natural_bengali: bool = Field(..., description="True if text flows naturally without awkward machine translation artifacts")
    grammatically_sound: bool = Field(..., description="True if grammar and spelling follow standard Bengali orthography")
    persona_register_aligned: bool = Field(..., description="True if student dialogue matches their intended persona style")
    score: float = Field(..., ge=0.0, le=1.0, description="Linguistic quality score")
    passed: bool = Field(..., description="Pass flag (score >= 0.80)")
    feedback: str = Field(..., description="Linguistic critique and feedback on Bengali phrasing")


class BengaliLinguisticJudge:
    """Evaluates the linguistic quality, register authenticity, and fluency of Bengali dialogues."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    async def evaluate(self, session: DialogueSession) -> MetricResult:
        dialogue_text = "\n".join([f"{t.role.value.upper()}: {t.content}" for t in session.turns])
        student_p = session.student_persona

        prompt = (
            f"Student Persona Profile: {student_p.name} ({student_p.communication_style})\n"
            f"Target Language: {session.language}\n\n"
            f"Dialogue to Evaluate:\n{dialogue_text}\n\n"
            f"Evaluate the Bengali linguistic quality:\n"
            f"1. Is the Bengali fluent, idiomatic, and natural (স্বাভাবিক বাংলা প্রয়োগ)?\n"
            f"2. Are there any awkward transliteration or translation artifacts?\n"
            f"3. Does the student's register match their persona background?\n"
        )
        system = "You are a native Bengali linguist and educational content reviewer. Evaluate linguistic fluency and authentic persona register."

        res: BengaliLinguisticEvaluationSchema = await self.llm_client.generate_structured(
            prompt=prompt,
            response_schema=BengaliLinguisticEvaluationSchema,
            system_instruction=system,
            temperature=0.1
        )

        return MetricResult(
            name="bengali_linguistic_quality",
            score=res.score,
            passed=res.passed and res.score >= 0.80,
            feedback=res.feedback
        )
