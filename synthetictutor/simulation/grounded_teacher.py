"""
Locale-Grounded Socratic Teacher Agent for SyntheticTutor.
Integrates WBBSE curriculum goals, Track 2 regional affordance contracts,
and pedagogical intent routing with strict Socratic non-directiveness guarantees.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from synthetictutor.core.schemas import TeacherPersona, TeachingPlan, DialogueTurn, DialogueRole
from synthetictutor.llm.base import BaseLLMClient
from synthetictutor.knowledge.locale_retriever import LocaleRetriever
from synthetictutor.knowledge.entities import EntityRegistry
from synthetictutor.simulation.intent_router import IntentRouter, PedagogicalIntent, StudentMentalState


class GroundedTeacherTurnOutput(BaseModel):
    inner_thought: str = Field(..., description="Pedagogical chain-of-thought analysis before speaking")
    pedagogical_intent: str = Field(..., description="The chosen strategy e.g. probe_misconception, scaffold_hint, counter_example")
    response_text: str = Field(..., description="Spoken teacher utterance in target language (concise, <= 3 sentences)")
    scaffold_level: int = Field(default=1, description="Scaffold depth from 1 (subtle) to 4 (high assistance)")
    grounding_entity_used: Optional[str] = Field(default=None, description="Local entity ID used if analogical bridge applied")


class GroundedTeacherAgent:
    """
    Simulates an expert Socratic educator grounded in West Bengal curriculum and local agro-cultural contexts.
    Guarantees strict non-directiveness: helps the student discover insights without blurting out definitions.
    """

    def __init__(
        self,
        persona: TeacherPersona,
        llm_client: BaseLLMClient,
        locale_retriever: Optional[LocaleRetriever] = None,
        entity_registry: Optional[EntityRegistry] = None,
        target_district: Optional[str] = None
    ):
        self.persona = persona
        self.llm_client = llm_client
        self.locale_retriever = locale_retriever or LocaleRetriever()
        self.entity_registry = entity_registry or EntityRegistry()
        self.target_district = target_district
        self.intent_router = IntentRouter()

    async def generate_turn(
        self,
        plan: TeachingPlan,
        history: List[DialogueTurn],
        student_state: Optional[StudentMentalState] = None,
        language: str = "bn"
    ) -> DialogueTurn:
        """
        Generates the next grounded Socratic teacher turn with <think> reasoning.
        """
        # 1. Determine pedagogical intent
        routing = self.intent_router.route_intent(
            history=history,
            active_misconceptions=plan.target_misconceptions,
            student_state=student_state
        )

        # 2. Retrieve local grounding facts and entity substitutions if applicable
        zone_key = self.locale_retriever.get_zone_for_district(self.target_district) if self.target_district else None
        grounding_context_text = ""
        substitutions_text = ""

        if zone_key:
            zone_ctx = self.locale_retriever.get_zone_context(zone_key)
            if zone_ctx:
                grounding_context_text = (
                    f"Local Zone: {zone_ctx.get('name_bengali', zone_key)}\n"
                    f"Common Crops/Features: {zone_ctx.get('primary_crops', [])}\n"
                    f"Geographic Traits: {zone_ctx.get('terrain_type', '')}"
                )

            # Check for relevant substitution rules
            subs = self.locale_retriever.get_substitutions_for_zone(zone_key)
            if subs:
                subs_formatted = []
                for s in subs[:2]:
                    disflags = f" [Disanalogy caveat: {', '.join(s.get('disanalogy_flags', []))}]" if s.get('disanalogy_flags') else ""
                    subs_formatted.append(f"- Concept: {s.get('canonical_concept')} -> Analogy: {s.get('description')}{disflags}")
                substitutions_text = "\n".join(subs_formatted)

        # 3. Format history
        history_text = "\n".join([f"{t.role.value.upper()}: {t.content}" for t in history])

        # 4. Construct System Instruction with strict Socratic rules
        lang_instruction = "Respond in natural, fluent Bengali (বাংলা) with standard terminology." if language == "bn" else "Respond in English."

        system_instruction = (
            f"You are {self.persona.name}, an expert Socratic educator in a West Bengal school.\n"
            f"Teaching Style: {self.persona.teaching_style}\n"
            f"Tone: {self.persona.tone}\n"
            f"Language: {lang_instruction}\n\n"
            f"PEDAGOGICAL INTENT FOR THIS TURN: {routing.intent.value.upper()}\n"
            f"Intent Rationale: {routing.rationale}\n\n"
            f"STRICT SOCRATIC RULES:\n"
            f"1. NON-DIRECTIVENESS: NEVER state the final definition, formula, or solution directly to the student.\n"
            f"2. Always guide via a probing question, intermediate cue, or concrete physical scenario.\n"
            f"3. Keep your spoken response concise: MAXIMUM 2-3 sentences.\n"
            f"4. If introducing a local analogy, ensure the physical analogy is clear without obscuring scientific rigor.\n"
        )

        prompt = (
            f"Target Concept: {plan.target_concept.name} ({plan.target_concept.description})\n"
            f"Learning Objective: {plan.learning_objective}\n"
            f"Target Misconceptions: {[m.description for m in plan.target_misconceptions]}\n\n"
            f"Regional Grounding Context:\n{grounding_context_text or 'Standard West Bengal school setting'}\n\n"
            f"Relevant Local Analogies:\n{substitutions_text or 'None'}\n\n"
            f"Dialogue History:\n{history_text or 'Starting initial question.'}\n\n"
            f"Generate your next teacher turn fulfilling the intent '{routing.intent.value}'."
        )

        resp: GroundedTeacherTurnOutput = await self.llm_client.generate_structured(
            prompt=prompt,
            response_schema=GroundedTeacherTurnOutput,
            system_instruction=system_instruction,
            temperature=0.65
        )

        turn_num = len(history) + 1
        return DialogueTurn(
            turn_number=turn_num,
            role=DialogueRole.TEACHER,
            content=resp.response_text,
            inner_thought=resp.inner_thought,
            pedagogical_intent=resp.pedagogical_intent
        )
