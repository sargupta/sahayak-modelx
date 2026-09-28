"""
Unified Quality Verification Pipeline for SahayakAI Synthetic Dialogues.
Orchestrates multi-judge validation (Pedagogy, Math, Grounding, Linguistic, Factual)
and enforces gate thresholds for admission to the training dataset.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

from synthetictutor.core.schemas import DialogueSession, MetricResult
from synthetictutor.llm.base import BaseLLMClient
from synthetictutor.knowledge.locale_retriever import LocaleRetriever
from synthetictutor.evaluation.pedagogical_judge import PedagogicalQualityJudge, AnswerLeakageJudge
from synthetictutor.evaluation.factual_judge import FactualAccuracyJudge
from synthetictutor.evaluation.math_judge import MathVerificationJudge
from synthetictutor.evaluation.locale_grounding_judge import LocaleGroundingJudge
from synthetictutor.evaluation.bengali_linguistic_judge import BengaliLinguisticJudge


class QualityReport(BaseModel):
    session_id: str
    target_concept: str
    target_district: Optional[str] = None
    pedagogical_score: float
    math_score: float
    grounding_score: float
    linguistic_score: float
    factual_score: float
    anti_leakage_passed: bool
    composite_score: float
    passed: bool
    detailed_metrics: Dict[str, MetricResult]
    rejection_reasons: List[str] = Field(default_factory=list)


class QualityVerificationPipeline:
    """
    Comprehensive multi-judge quality assurance pipeline for validating synthetic educational dialogues.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        locale_retriever: Optional[LocaleRetriever] = None,
        min_composite_score: float = 0.80
    ):
        self.llm_client = llm_client
        self.locale_retriever = locale_retriever or LocaleRetriever()
        self.min_composite_score = min_composite_score

        # Initialize individual judges
        self.pedagogy_judge = PedagogicalQualityJudge(llm_client)
        self.leakage_judge = AnswerLeakageJudge(llm_client)
        self.factual_judge = FactualAccuracyJudge(llm_client)
        self.math_judge = MathVerificationJudge(llm_client)
        self.grounding_judge = LocaleGroundingJudge(llm_client, self.locale_retriever)
        self.linguistic_judge = BengaliLinguisticJudge(llm_client)

    async def evaluate_dialogue(
        self,
        session: DialogueSession,
        source_textbook_context: Optional[str] = None,
        target_district: Optional[str] = None
    ) -> QualityReport:
        """
        Runs all quality judges and produces a comprehensive validation report.
        """
        rejection_reasons: List[str] = []
        metrics: Dict[str, MetricResult] = {}

        # 1. Pedagogical Quality & Anti-Leakage
        ped_res = await self.pedagogy_judge.evaluate(session)
        metrics["pedagogical_quality"] = ped_res
        if not ped_res.passed:
            rejection_reasons.append(f"Failed Pedagogy: {ped_res.feedback}")

        leak_res = await self.leakage_judge.evaluate(session)
        metrics["anti_leakage"] = leak_res
        if not leak_res.passed:
            rejection_reasons.append(f"Answer Leakage Detected: {leak_res.feedback}")

        # 2. Math & Dimensional Verification
        math_res = await self.math_judge.evaluate(session)
        metrics["math_verification"] = math_res
        if not math_res.passed:
            rejection_reasons.append(f"Failed Math Verification: {math_res.feedback}")

        # 3. Regional Grounding & Locale Verification
        ground_res = await self.grounding_judge.evaluate(session, target_district=target_district)
        metrics["locale_grounding"] = ground_res
        if not ground_res.passed:
            rejection_reasons.append(f"Failed Locale Grounding: {ground_res.feedback}")

        # 4. Bengali Linguistic Naturalness
        ling_res = await self.linguistic_judge.evaluate(session)
        metrics["bengali_linguistic_quality"] = ling_res
        if not ling_res.passed:
            rejection_reasons.append(f"Failed Linguistic Quality: {ling_res.feedback}")

        # 5. Factual Accuracy against textbook context
        fact_context = source_textbook_context or session.plan.target_concept.description
        fact_res = await self.factual_judge.evaluate(session, source_context=fact_context)
        metrics["factual_accuracy"] = fact_res
        if not fact_res.passed:
            rejection_reasons.append(f"Failed Factual Accuracy: {fact_res.feedback}")

        # Compute weighted composite score
        # Pedagogy (0.25) + Math (0.20) + Grounding (0.20) + Factuality (0.20) + Linguistic (0.15)
        composite = (
            0.25 * ped_res.score +
            0.20 * math_res.score +
            0.20 * ground_res.score +
            0.20 * fact_res.score +
            0.15 * ling_res.score
        )

        overall_passed = (
            composite >= self.min_composite_score and
            leak_res.passed and
            len(rejection_reasons) == 0
        )

        return QualityReport(
            session_id=session.id,
            target_concept=session.plan.target_concept.name,
            target_district=target_district,
            pedagogical_score=round(ped_res.score, 3),
            math_score=round(math_res.score, 3),
            grounding_score=round(ground_res.score, 3),
            linguistic_score=round(ling_res.score, 3),
            factual_score=round(fact_res.score, 3),
            anti_leakage_passed=leak_res.passed,
            composite_score=round(composite, 3),
            passed=overall_passed,
            detailed_metrics=metrics,
            rejection_reasons=rejection_reasons
        )
