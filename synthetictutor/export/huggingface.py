"""
ChatML Dataset Exporter for SyntheticTutor & Sarvam-30B.
Exports multi-turn dialogues with role tokens (<|im_start|> / <|im_end|>) and system instruction scaffolding.
"""

import json
import os
from typing import List, Optional, Dict, Any
from synthetictutor.core.schemas import DialogueSession
from synthetictutor.export.sharegpt import normalize_numerals


DEFAULT_SYSTEM_BN = (
    "তুমি SahayakAI — পশ্চিমবঙ্গের সরকারি স্কুলের ছাত্রছাত্রীদের জন্য বাংলা মাধ্যমে "
    "ধাপে ধাপে অঙ্ক ও বিজ্ঞান শেখানোর একটি শিক্ষা-সহায়ক। সবসময় সহজ বাংলায়, ধাপ ধরে বোঝাও।"
)


class ChatMLExporter:
    """Exports evaluated DialogueSessions to ChatML standard format."""

    def __init__(
        self,
        system_instruction: str = DEFAULT_SYSTEM_BN,
        include_inner_thought: bool = True,
        numerals_mode: str = "bengali"
    ):
        self.system_instruction = system_instruction
        self.include_inner_thought = include_inner_thought
        self.numerals_mode = numerals_mode

    def export_session(self, session: DialogueSession) -> Dict[str, Any]:
        """Converts a single session into a ChatML messages object."""
        messages = [
            {
                "role": "system",
                "content": self.system_instruction
            }
        ]

        for t in session.turns:
            role = "user" if t.role.value == "student" else "assistant"
            content = t.content

            if role == "assistant" and self.include_inner_thought and t.inner_thought:
                content = f"<think>\n{t.inner_thought.strip()}\n</think>\n{content}"

            content = normalize_numerals(content, self.numerals_mode)

            messages.append({
                "role": role,
                "content": content
            })

        return {
            "id": session.id,
            "messages": messages,
            "concept_id": session.plan.target_concept.id,
            "language": session.language,
            "metadata": {
                "grade_level": session.student_persona.grade_level,
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
