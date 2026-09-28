"""
batch_50k_generator.py — Master 50k Dataset Generation & Orchestration Engine for SahayakAI.

Features:
1. 4D Quota Allocation (Educational Stages, Boards, Subjects, Task Families)
2. Request-First Curriculum & Multi-Chunk Textbook Retrieval
3. Locale Context Retrieval with Affordance Check
4. Grounding Compatibility Gate
5. 8-Stage Deterministic Validation Suite (Bengali Numerals, Anti-Visual Leak, Format Discipline)
6. Transactional Sharded Checkpointing (JSONL / SQLite) with Resume Support
7. Multi-Engine Dispatcher (Gemini Batch, SageMaker Sarvam-30B vLLM, OpenAI Compatible, Mock)
8. Full Provenance & Licensing Metadata on every record.
"""

import os
import json
import sqlite3
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass, asdict

from synthetictutor.knowledge.textbook_retriever import TextbookRetriever
from synthetictutor.knowledge.locale_retriever import LocaleRetriever
from synthetictutor.knowledge.grounding_compatibility_gate import GroundingCompatibilityGate
from synthetictutor.pipeline.intent_generator import IntentGenerator, UserIntent
from synthetictutor.pipeline.grounded_generator import GroundedGenerator
from synthetictutor.evaluation.grounding_first_validator import GroundingFirstValidator


@dataclass
class GenerationConfig50k:
    total_target: int = 50000
    output_dir: str = "output/50k_dataset"
    checkpoint_db: str = "output/50k_dataset/checkpoint.db"
    shard_size: int = 5000
    source_pool_path: str = "datasets/grounding_corpus_clean_master.jsonl"
    locale_path: str = "locale.json"
    generator_model: str = "gemini-1.5-pro-002"
    provider: str = "google_batch"
    license_class: str = "clean_research_sft"
    min_fidelity_score: int = 3
    teacher_ratio: float = 0.85
    student_ratio: float = 0.15


# 4D Quotas defined in the 50k Master Plan
STAGE_TARGETS = {
    "PRIMARY_1_5": {"quota": 10000, "share": 0.20, "grades": [1, 2, 3, 4, 5], "board": "WBBPE"},
    "MIDDLE_6_8": {"quota": 17500, "share": 0.35, "grades": [6, 7, 8], "board": "WBBSE"},
    "SECONDARY_9_10": {"quota": 15000, "share": 0.30, "grades": [9, 10], "board": "WBBSE"},
    "HIGHER_SECONDARY_11_12": {"quota": 7500, "share": 0.15, "grades": [11, 12], "board": "WBCHSE"}
}

TASK_TARGETS = {
    "CONCEPT_EXPLANATION": {"quota": 17500, "share": 0.35},
    "GUIDED_PROBLEM_SOLVING": {"quota": 12500, "share": 0.25},
    "LESSON_PLAN": {"quota": 7500, "share": 0.15},
    "QUIZ_GENERATION": {"quota": 7500, "share": 0.15},
    "WORKSHEET_PRACTICE": {"quota": 5000, "share": 0.10}
}


def format_grade_bengali(grade: Any) -> str:
    mapping = {
        1: "প্রথম শ্রেণি", 2: "দ্বিতীয় শ্রেণি", 3: "তৃতীয় শ্রেণি", 4: "চতুর্থ শ্রেণি",
        5: "পঞ্চম শ্রেণি", 6: "ষষ্ঠ শ্রেণি", 7: "সপ্তম শ্রেণি", 8: "অষ্টম শ্রেণি",
        9: "নবম শ্রেণি", 10: "দশম শ্রেণি", 11: "একাদশ শ্রেণি", 12: "দ্বাদশ শ্রেণি"
    }
    try:
        g_int = int(str(grade).replace("Class", "").replace("class", "").strip())
        return mapping.get(g_int, f"{grade}ম শ্রেণি")
    except Exception:
        return f"{grade}ম শ্রেণি"


def build_system_prompt(
    board: str,
    grade: int,
    subject: str,
    topic: str,
    target_role: str = "STUDENT",
    region: str = "পশ্চিমবঙ্গ সাধারণ পাঠ্যক্রম",
    locale_fact: Optional[str] = None
) -> str:
    board_full = {
        "WBBPE": "পশ্চিমবঙ্গ প্রাথমিক শিক্ষা পর্ষদ (WBBPE)",
        "WBBSE": "পশ্চিমবঙ্গ মধ্যশিক্ষা পর্ষদ (WBBSE)",
        "WBCHSE": "পশ্চিমবঙ্গ উচ্চমাধ্যমিক শিক্ষা সংসদ (WBCHSE)"
    }.get(board, board if "পশ্চিমবঙ্গ" in board else "পশ্চিমবঙ্গ মধ্যশিক্ষা পর্ষদ (WBBSE)")

    grade_bengali = format_grade_bengali(grade)

    addressing_rule = (
        "১. শিক্ষক বা অভিভাবকের উদ্দেশ্যে সম্মানসূচক 'আপনি' ভাষা ব্যবহার করবে।"
        if target_role == "TEACHER"
        else "১. শিক্ষার্থীর উদ্দেশ্যে সহজ, বন্ধুভাবাপন্ন 'তুমি/তোমরা' ভাষা ব্যবহার করবে।"
    )

    locale_section = f"\n- আঞ্চলিক প্রেক্ষাপট (যাচাইকৃত তথ্য): {locale_fact}" if locale_fact else ""

    return f"""তুমি 'সহায়কএআই' (SahayakAI) — {board_full} বেঙ্গলি-মিডিয়াম স্কুলের শিক্ষার্থী ও শিক্ষক মহাশয়দের জন্য তৈরি একজন অভিজ্ঞ, বিশেষায়িত অ্যাকাডেমিক AI টিউটর। তোমার কাজ হলো পাঠ্যক্রম অনুযায়ী সহজ, সাবলীল, সঠিক ও শিক্ষার্থী-উপযোগী বাংলা ভাষায় পাঠদান, ধারণা ব্যাখ্যা, সমস্যা সমাধান এবং অধ্যয়ন-সংক্রান্ত সহায়তা প্রদান করা।

নিয়মাবলী ও সম্বোধন:
{addressing_rule}
২. স্পষ্ট, প্রাঞ্জল ও স্বাভাবিক ভাষায় সরাসরি শিক্ষামূলক উত্তরে প্রবেশ করবে; অপ্রয়োজনীয় অভিবাদন, প্রশংসা বা motivational filler এড়িয়ে চলবে।
৩. পাঠ্যক্রম-উপযোগী প্রমিত বাংলা পরিভাষা ব্যবহার করবে।
৪. সাধারণ বাংলা গদ্যে সব সংখ্যা বাংলা অঙ্কে লিখবে: ০, ১, ২, ৩, ৪, ৫, ৬, ৭, ৮, ৯।
৫. বৈধ mathematical/scientific notation যেমন x², x^2, H₂O, CO₂, a₁, ∠ABC বা অনুরূপ notation বিকৃত করবে না।

শিক্ষাদান ও পাঠ্যক্রম-নির্ভরতা:
৬. পাঠ্যক্রম ও বিষয়বস্তুর ধারণা সঠিকভাবে ব্যাখ্যা করবে এবং কোনো বিভ্রান্তিকর তথ্য প্রদান করবে না।
৭. সাধারণ বাংলা গদ্যে 'প্রদত্ত পাঠ্যাংশে' বা 'উক্ত পাঠ্যাংশ অনুসারে' জাতীয় কৃত্রিম বাক্য পরিহার করে সরাসরি বিষয়ের ধারণা সহজভাবে বুঝিয়ে দেবে।
৮. গণিত ও বিজ্ঞানের ক্ষেত্রে সূত্র, হিসাব, একক এবং intermediate steps সঠিক রাখবে।
৯. উত্তর সম্পূর্ণ রাখবে এবং যথাযথ বিরামচিহ্ন (। বা ?) দিয়ে শেষ করবে।

সহায়কএআই (SahayakAI) পাঠ্যসূচি বিবরণী:
- পর্ষদ/সংসদ: {board_full}
- বিষয়: {subject}
- শ্রেণি: {grade_bengali}
- বিষয়বস্তু: {topic}
- অঞ্চল/জেলা: {region}{locale_section}"""



class Batch50kGenerator:
    """Production batch orchestrator for 50k SFT dataset generation."""

    def __init__(self, config: Optional[GenerationConfig50k] = None):
        self.config = config or GenerationConfig50k()
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Initialize knowledge retrieval and gate components
        self.retriever = TextbookRetriever(self.config.source_pool_path)
        self.locale_retriever = LocaleRetriever(self.config.locale_path)
        self.gate = GroundingCompatibilityGate(min_overlap_threshold=0.2)
        self.validator = GroundingFirstValidator()
        self.intent_gen = IntentGenerator()
        self.generator = GroundedGenerator()
        
        self._init_db()

    def _init_db(self):
        """Initializes transactional SQLite checkpoint database."""
        conn = sqlite3.connect(self.config.checkpoint_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generated_records (
                id TEXT PRIMARY KEY,
                stage TEXT,
                grade INTEGER,
                subject TEXT,
                task_family TEXT,
                fidelity_score INTEGER,
                validation_passed INTEGER,
                record_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def get_completed_count(self) -> int:
        """Returns total successfully validated records in checkpoint db."""
        conn = sqlite3.connect(self.config.checkpoint_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM generated_records WHERE validation_passed = 1")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_stage_counts(self) -> Dict[str, int]:
        """Returns record counts per educational stage."""
        conn = sqlite3.connect(self.config.checkpoint_db)
        cursor = conn.cursor()
        cursor.execute("SELECT stage, COUNT(*) FROM generated_records WHERE validation_passed = 1 GROUP BY stage")
        rows = cursor.fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}

    def save_record(self, record: Dict[str, Any], validation_passed: bool) -> bool:
        """Saves a generated record transactionally."""
        conn = sqlite3.connect(self.config.checkpoint_db)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO generated_records 
                (id, stage, grade, subject, task_family, fidelity_score, validation_passed, record_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get("id"),
                record.get("stage", "UNKNOWN"),
                record.get("grade", 0),
                record.get("subject", "General"),
                record.get("task_family", "CONCEPT_EXPLANATION"),
                record.get("curriculum_fidelity_score", 0),
                1 if validation_passed else 0,
                json.dumps(record, ensure_ascii=False)
            ))
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            return False
        finally:
            conn.close()

    def synthesize_record(
        self,
        intent: UserIntent,
        custom_id: str,
        llm_response_fn: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Executes complete pipeline for one unit:
        1. Retrieval
        2. Compatibility Gate Check
        3. LLM Response Generation
        4. 8-Stage Deterministic Validation
        5. Provenance Injection
        """
        # 1. Retrieval
        is_complex = intent.task_type in ["LESSON_PLAN", "QUIZ_GENERATION", "WORKSHEET_PRACTICE"]
        if is_complex:
            chunks = self.retriever.retrieve_multi_chunk(
                query=intent.query,
                grade=intent.target_grade,
                subject=intent.target_subject,
                topic=intent.target_topic,
                max_chunks=3
            )
        else:
            chunks = self.retriever.retrieve(
                query=intent.query,
                grade=intent.target_grade,
                subject=intent.target_subject,
                topic=intent.target_topic,
                top_k=1
            )

        # 2. Locale Retrieval
        locale_snips = []
        if self.locale_retriever.is_localization_requested(intent.query):
            locale_snips = self.locale_retriever.retrieve_locale_facts(intent.query)

        # 3. Gate Verification
        primary_chunk = chunks[0] if chunks else {}
        compat = self.gate.evaluate(
            user_query=intent.query,
            retrieved_chunk=primary_chunk,
            task_type=intent.task_type,
            target_subject=intent.target_subject,
            target_grade=intent.target_grade,
            locale_snippets=locale_snips
        )
        fidelity_score = compat.curriculum_fidelity_score
        if not compat.is_compatible or fidelity_score < self.config.min_fidelity_score:
            return None

        # 4. Generation
        if llm_response_fn:
            generated_output = llm_response_fn(intent, chunks, locale_snips)
        else:
            generated_output = self.generator.generate_response(
                intent=intent,
                textbook_chunks=chunks,
                locale_snippets=locale_snips,
                model_name=self.config.generator_model
            )

        # 5. Assemble candidate record in standard ChatML/ShareGPT format
        board_name = compat.board_authority or self._get_board_for_grade(intent.target_grade)
        locale_fact_str = locale_snips[0].text if (locale_snips and hasattr(locale_snips[0], "text")) else (str(locale_snips[0]) if locale_snips else None)

        system_prompt = build_system_prompt(
            board=board_name,
            grade=intent.target_grade,
            subject=intent.target_subject,
            topic=intent.target_topic,
            target_role=intent.requester_role,
            locale_fact=locale_fact_str
        )

        chunk_ids = []
        for c in chunks:
            if isinstance(c, dict):
                chunk_ids.append(c.get("chunk_id", ""))
            elif hasattr(c, "chunk_id"):
                chunk_ids.append(c.chunk_id)

        record = {
            "custom_id": custom_id,
            "id": custom_id,
            "stage": self._resolve_stage(intent.target_grade),
            "grade": intent.target_grade,
            "subject": intent.target_subject,
            "task_family": intent.task_type,
            "topic": intent.target_topic,
            "curriculum_fidelity_score": fidelity_score,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": intent.query},
                {"role": "assistant", "content": generated_output}
            ],
            "metadata": {
                "board": compat.board_authority,
                "grade": intent.target_grade,
                "subject": intent.target_subject,
                "topic": intent.target_topic,
                "task_type": intent.task_type,
                "requester_role": intent.requester_role,
                "instructional_target": intent.instructional_target,
                "grounding_mode": "TEXTBOOK_INTERNAL",
                "curriculum_fidelity_score": fidelity_score,
                "provenance": {
                    "generator_model": self.config.generator_model,
                    "provider": self.config.provider,
                    "license_class": self.config.license_class,
                    "chunk_ids": chunk_ids,
                    "locale_keys": [s.fact_key for s in locale_snips] if locale_snips else [],
                    "validator_version": "v2.4"
                }
            }
        }

        # 6. 8-Stage Deterministic Validation
        validation_res = self.validator.validate_record(record)
        is_valid = validation_res.passed
        record["validation"] = {
            "passed": validation_res.passed,
            "errors": validation_res.errors,
            "warnings": validation_res.warnings,
            "tier_failures": validation_res.tier_failures
        }

        # 7. Save to DB
        self.save_record(record, is_valid)
        return record if is_valid else None

    def _resolve_stage(self, grade: Any) -> str:
        try:
            g = int(str(grade).replace("Class", "").replace("class", "").replace("Gr", "").strip() or "8")
        except Exception:
            g = 8

        if g <= 5:
            return "PRIMARY_1_5"
        elif g <= 8:
            return "MIDDLE_6_8"
        elif g <= 10:
            return "SECONDARY_9_10"
        else:
            return "HIGHER_SECONDARY_11_12"

    def _get_board_for_grade(self, grade: Any) -> str:
        try:
            g = int(str(grade).replace("Class", "").replace("class", "").replace("Gr", "").strip() or "8")
        except Exception:
            g = 8
        if g <= 5:
            return "WBBPE"
        elif g <= 10:
            return "WBBSE"
        else:
            return "WBCHSE"

    def export_shards(self, shard_size: Optional[int] = None, messages_only: bool = True) -> List[str]:
        """Exports approved records from SQLite into clean JSONL shards with only 'messages' for SFT training."""
        shard_size = shard_size or self.config.shard_size
        conn = sqlite3.connect(self.config.checkpoint_db)
        cursor = conn.cursor()
        cursor.execute("SELECT record_json FROM generated_records WHERE validation_passed = 1 ORDER BY id ASC")
        
        exported_files = []
        shard_idx = 1
        current_shard = []
        
        row = cursor.fetchone()
        while row:
            rec = json.loads(row[0])
            out_item = {"messages": rec.get("messages", [])} if messages_only else rec
            current_shard.append(out_item)
            if len(current_shard) >= shard_size:
                shard_path = os.path.join(self.config.output_dir, f"sahayak_50k_shard_{shard_idx:02d}.jsonl")
                with open(shard_path, "w", encoding="utf-8") as f:
                    for item in current_shard:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                exported_files.append(shard_path)
                shard_idx += 1
                current_shard = []
            row = cursor.fetchone()
            
        if current_shard:
            shard_path = os.path.join(self.config.output_dir, f"sahayak_50k_shard_{shard_idx:02d}.jsonl")
            with open(shard_path, "w", encoding="utf-8") as f:
                for item in current_shard:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            exported_files.append(shard_path)
            
        conn.close()
        return exported_files
