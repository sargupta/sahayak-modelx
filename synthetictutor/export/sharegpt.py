"""
ShareGPT SFT Dataset Exporter for SyntheticTutor & Sarvam-30B.
Exports multi-turn dialogues with optional <think> inner-thought reasoning and Bengali numeral normalization.
"""

import json
import os
import re
from typing import List, Optional, Dict, Any
from synthetictutor.core.schemas import DialogueSession


BMAP = {'০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4', '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9'}
BINV = {v: k for k, v in BMAP.items()}


def to_bengali_numerals(text: str) -> str:
    """Converts ASCII digits (0-9) to Bengali digits (০-৯)."""
    return ''.join(BINV.get(c, c) for c in text)


def to_ascii_numerals(text: str) -> str:
    """Converts Bengali digits (০-৯) to ASCII digits (0-9)."""
    return ''.join(BMAP.get(c, c) for c in text)


def normalize_numerals(text: str, mode: str = "keep") -> str:
    if mode == "bengali":
        return to_bengali_numerals(text)
    elif mode == "ascii":
        return to_ascii_numerals(text)
    return text


class ShareGPTExporter:
    """Exports evaluated DialogueSessions to standard ShareGPT SFT JSONL format."""

    def __init__(
        self,
        include_inner_thought: bool = True,
        numerals_mode: str = "bengali"
    ):
        self.include_inner_thought = include_inner_thought
        self.numerals_mode = numerals_mode

    def export_session(self, session: DialogueSession) -> Dict[str, Any]:
        """Converts a single DialogueSession to a ShareGPT dictionary record."""
        conversations = []
        for t in session.turns:
            from_role = "human" if t.role.value == "student" else "gpt"
            content = t.content

            # Embed <think> CoT in GPT turns if requested
            if from_role == "gpt" and self.include_inner_thought and t.inner_thought:
                content = f"<think>\n{t.inner_thought.strip()}\n</think>\n{content}"

            content = normalize_numerals(content, self.numerals_mode)

            conversations.append({
                "from": from_role,
                "value": content
            })

        return {
            "id": session.id,
            "target_concept": session.plan.target_concept.name,
            "language": session.language,
            "conversations": conversations,
            "metadata": {
                "concept_id": session.plan.target_concept.id,
                "grade_level": session.student_persona.grade_level,
                "student_name": session.student_persona.name,
                "teaching_style": session.teacher_persona.teaching_style,
                "learning_objective": session.plan.learning_objective,
                "completed": session.completed
            }
        }

    def export(self, sessions: List[DialogueSession], output_path: str) -> str:
        """Exports sessions to output_path JSONL file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        records = [self.export_session(s) for s in sessions]

        with open(output_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        return output_path
