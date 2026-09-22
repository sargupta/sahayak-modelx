"""
Pedagogical Intent Router for Socratic Dialogue Simulation.
Selects and adapts pedagogical strategies (probing, scaffolding, analogical bridging, counter-examples).
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from synthetictutor.core.schemas import DialogueTurn, DialogueRole, Misconception


class PedagogicalIntent(str, Enum):
    PROBE_MISCONCEPTION = "probe_misconception"
    SCAFFOLD_HINT = "scaffold_hint"
    ANALOGICAL_BRIDGE = "analogical_bridge"
    COUNTER_EXAMPLE = "counter_example"
    VERIFY_REASONING = "verify_reasoning"
    AFFIRM_AND_EXTEND = "affirm_and_extend"


class StudentMentalState(str, Enum):
    CONFUSED = "confused"
    MISCONCEPTION_ACTIVE = "misconception_active"
    PARTIAL_UNDERSTANDING = "partial_understanding"
    BREAKTHROUGH = "breakthrough"
    MASTERED = "mastered"
    DISENGAGED = "disengaged"


class IntentRoutingDecision(BaseModel):
    intent: PedagogicalIntent
    rationale: str
    suggested_scaffold_depth: int = Field(default=1, ge=1, le=4)
    target_misconception_id: Optional[str] = None
    use_local_analogy: bool = False


class IntentRouter:
    """
    Determines the optimal next pedagogical move for a Socratic teacher
    based on student mental state, misconception persistence, and dialogue progress.
    """

    def __init__(self):
        pass

    def route_intent(
        self,
        history: List[DialogueTurn],
        active_misconceptions: List[Misconception],
        student_state: Optional[StudentMentalState] = None,
        force_intent: Optional[PedagogicalIntent] = None
    ) -> IntentRoutingDecision:
        """
        Calculates the next pedagogical intent.
        """
        if force_intent:
            return IntentRoutingDecision(
                intent=force_intent,
                rationale="Forced intent override by lesson planner.",
                suggested_scaffold_depth=1
            )

        # 1. Check explicit state conditions first
        if student_state in [StudentMentalState.CONFUSED, StudentMentalState.DISENGAGED]:
            return IntentRoutingDecision(
                intent=PedagogicalIntent.ANALOGICAL_BRIDGE,
                rationale="Student is confused or struggling with abstract terms; ground reasoning in a local physical analogy.",
                suggested_scaffold_depth=2,
                use_local_analogy=True
            )

        if student_state == StudentMentalState.BREAKTHROUGH:
            return IntentRoutingDecision(
                intent=PedagogicalIntent.VERIFY_REASONING,
                rationale="Student reached a breakthrough; ask them to explain 'why' to consolidate the reasoning.",
                suggested_scaffold_depth=1
            )

        if student_state == StudentMentalState.MASTERED:
            return IntentRoutingDecision(
                intent=PedagogicalIntent.AFFIRM_AND_EXTEND,
                rationale="Student demonstrated conceptual mastery; affirm and extend to a higher-order application.",
                suggested_scaffold_depth=1
            )

        if not history:
            return IntentRoutingDecision(
                intent=PedagogicalIntent.PROBE_MISCONCEPTION if active_misconceptions else PedagogicalIntent.SCAFFOLD_HINT,
                rationale="Initial turn: establish student baseline understanding.",
                suggested_scaffold_depth=1
            )

        last_student_turn = next((t for t in reversed(history) if t.role == DialogueRole.STUDENT), None)
        teacher_turns = [t for t in history if t.role == DialogueRole.TEACHER]
        scaffold_depth = min(len(teacher_turns) + 1, 4)

        # 2. Detect active misconception in student's utterance or state
        if student_state == StudentMentalState.MISCONCEPTION_ACTIVE or (active_misconceptions and scaffold_depth <= 2):
            # Alternate between counter-example and probing
            prev_intents = [t.pedagogical_intent for t in teacher_turns]
            if PedagogicalIntent.COUNTER_EXAMPLE.value not in prev_intents:
                return IntentRoutingDecision(
                    intent=PedagogicalIntent.COUNTER_EXAMPLE,
                    rationale="Student holds an active misconception; present a concrete physical counter-example.",
                    suggested_scaffold_depth=scaffold_depth,
                    target_misconception_id=active_misconceptions[0].id if active_misconceptions else None,
                    use_local_analogy=True
                )
            else:
                return IntentRoutingDecision(
                    intent=PedagogicalIntent.PROBE_MISCONCEPTION,
                    rationale="Misconception was challenged; probe if student perceives the contradiction.",
                    suggested_scaffold_depth=scaffold_depth,
                    target_misconception_id=active_misconceptions[0].id if active_misconceptions else None
                )

        # 2. Handle confusion or disengagement
        if student_state in [StudentMentalState.CONFUSED, StudentMentalState.DISENGAGED]:
            return IntentRoutingDecision(
                intent=PedagogicalIntent.ANALOGICAL_BRIDGE,
                rationale="Student is confused or struggling with abstract terms; ground reasoning in a local physical analogy.",
                suggested_scaffold_depth=scaffold_depth,
                use_local_analogy=True
            )

        # 3. Handle partial understanding or recent breakthrough
        if student_state == StudentMentalState.PARTIAL_UNDERSTANDING:
            return IntentRoutingDecision(
                intent=PedagogicalIntent.SCAFFOLD_HINT,
                rationale="Student has partial insight; provide a targeted leading cue to bridge the remaining gap.",
                suggested_scaffold_depth=scaffold_depth
            )

        if student_state == StudentMentalState.BREAKTHROUGH:
            return IntentRoutingDecision(
                intent=PedagogicalIntent.VERIFY_REASONING,
                rationale="Student reached a breakthrough; ask them to explain 'why' to consolidate the reasoning.",
                suggested_scaffold_depth=1
            )

        # 4. Mastery reached or default progression
        if student_state == StudentMentalState.MASTERED:
            return IntentRoutingDecision(
                intent=PedagogicalIntent.AFFIRM_AND_EXTEND,
                rationale="Student demonstrated conceptual mastery; affirm and extend to a higher-order application.",
                suggested_scaffold_depth=1
            )

        # Default fallback: structured scaffold hint
        return IntentRoutingDecision(
            intent=PedagogicalIntent.SCAFFOLD_HINT,
            rationale="Progressive Socratic guidance step.",
            suggested_scaffold_depth=scaffold_depth,
            use_local_analogy=(scaffold_depth >= 2)
        )
