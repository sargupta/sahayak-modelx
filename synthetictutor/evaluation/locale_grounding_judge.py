"""
Locale Grounding and Fact Verification Judge for Socratic Dialogues.
Cross-references cited geographical entities, crops, rivers, and quantitative facts against locale.json.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from synthetictutor.core.schemas import DialogueSession, MetricResult
from synthetictutor.knowledge.locale_retriever import LocaleRetriever
from synthetictutor.llm.base import BaseLLMClient


class GroundingEvaluationSchema(BaseModel):
    contains_regional_references: bool = Field(..., description="True if dialogue mentions West Bengal districts, crops, landmarks, or local units")
    entities_factually_accurate: bool = Field(..., description="True if all local entities and geographical features cited are authentic")
    quantitative_facts_verified: bool = Field(..., description="True if any numbers (prices, rainfall, wages) match verified facts without hallucination")
    score: float = Field(..., ge=0.0, le=1.0, description="Grounding score")
    passed: bool = Field(..., description="Overall grounding pass flag (score >= 0.80)")
    feedback: str = Field(..., description="Evaluation notes on regional authenticity and grounding accuracy")


class LocaleGroundingJudge:
    """Evaluates adherence to verified West Bengal regional grounding and guards against regional hallucinations."""

    def __init__(self, llm_client: BaseLLMClient, locale_retriever: Optional[LocaleRetriever] = None):
        self.llm_client = llm_client
        self.locale_retriever = locale_retriever or LocaleRetriever()

    async def evaluate(self, session: DialogueSession, target_district: Optional[str] = None) -> MetricResult:
        dialogue_text = "\n".join([f"{t.role.value.upper()}: {t.content}" for t in session.turns])

        # Retrieve relevant locale facts for context
        district = target_district or self.locale_retriever.detect_target_district(dialogue_text)
        locale_facts = self.locale_retriever.retrieve_locale_facts(dialogue_text, target_district=district, max_facts=6)
        facts_text = "\n".join([f"- [{f.category}] {f.district}: {f.fact_key} = {f.fact_value} (Source: {f.source})" for f in locale_facts])

        prompt = (
            f"Target Concept: {session.plan.target_concept.name}\n"
            f"Target District/Region: {district or 'General West Bengal'}\n\n"
            f"Verified Grounding Reference Facts (from locale.json):\n{facts_text or 'Standard West Bengal curriculum context'}\n\n"
            f"Dialogue to Evaluate:\n{dialogue_text}\n\n"
            f"Evaluate if regional claims, local analogies, landmarks, crops, rivers, and prices cited in the dialogue are authentic and unhallucinated."
        )
        system = "You are a regional grounding auditor for West Bengal educational materials. Enforce factual geographical, agricultural, and socio-economic accuracy."

        res: GroundingEvaluationSchema = await self.llm_client.generate_structured(
            prompt=prompt,
            response_schema=GroundingEvaluationSchema,
            system_instruction=system,
            temperature=0.1
        )

        return MetricResult(
            name="locale_grounding",
            score=res.score,
            passed=res.passed and res.score >= 0.80,
            feedback=res.feedback
        )
