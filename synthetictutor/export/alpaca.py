"""
Alpaca Instruction-Tuning Dataset Exporter for SyntheticTutor.
Converts single-turn Q&A or key dialogue turns into {"instruction": ..., "input": ..., "output": ...} format.
"""

import json
import os
from typing import List, Dict, Any, Optional
from synthetictutor.core.schemas import DialogueSession
from synthetictutor.export.sharegpt import normalize_numerals


class AlpacaExporter:
    """Exports educational sessions to standard Alpaca instruction-tuning format."""

    def __init__(
        self,
        include_inner_thought: bool = True,
        numerals_mode: str = "bengali"
    ):
        self.include_inner_thought = include_inner_thought
        self.numerals_mode = numerals_mode

    def export_session(self, session: DialogueSession) -> List[Dict[str, Any]]:
        """
        Extracts teacher-student turn pairs as instruction-input-output records.
        """
        records = []
        turns = session.turns

        for i in range(len(turns) - 1):
            if turns[i].role.value == "teacher" and turns[i + 1].role.value == "student":
                # Teacher prompt -> Student response
                pass
            elif turns[i].role.value == "student" and turns[i + 1].role.value == "teacher":
                # Student question/response -> Teacher Socratic guidance
                instruction = (
                    f"তুমি একজন দক্ষ শিক্ষক। পশ্চিমবঙ্গ বোর্ডের {session.student_persona.grade_level} "
                    f"শ্রেণির শিক্ষার্থীকে সরাসরি উত্তর না বলে সক্রেটিক পদ্ধতিতে চিন্তা করতে সাহায্য করো।"
                )
                student_input = normalize_numerals(turns[i].content, self.numerals_mode)
                teacher_output = turns[i + 1].content

                if self.include_inner_thought and turns[i + 1].inner_thought:
                    teacher_output = f"<think>\n{turns[i + 1].inner_thought.strip()}\n</think>\n{teacher_output}"

                teacher_output = normalize_numerals(teacher_output, self.numerals_mode)

                records.append({
                    "instruction": instruction,
                    "input": student_input,
                    "output": teacher_output,
                    "concept": session.plan.target_concept.name,
                    "language": session.language
                })

        return records

    def export(self, sessions: List[DialogueSession], output_path: str) -> str:
        """Exports all extracted Alpaca records to JSONL file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        all_records = []
        for s in sessions:
            all_records.extend(self.export_session(s))

        with open(output_path, "w", encoding="utf-8") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        return output_path
