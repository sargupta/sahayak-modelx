# -*- coding: utf-8 -*-
"""
generate_api_sft_2000.py — High-fidelity, API-driven dataset generation engine for SahayakAI SFT.
Pulls directly from 19,500+ textbook grounding chunks across Classes 1-12 (WBBPE, WBBSE, WBCHSE)
and verified district context, generating authentic, high-quality Bengali pedagogical dialogues.
Heavily focused on Lesson Plans and Comprehensive Quiz Sets (~60%+ distribution).
Primary Provider: Sarvam AI Indic LLM (sarvam-105b-conversations & sarvam-105b).
Outputs strictly in the exact original schema:
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

import os
import sys
import json
import random
import re
import asyncio
import time
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Tuple

import aiohttp
from export_api_sft_formats import export_all

# API Configuration
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "sk_t3i8crml_elALafNJUw1DyJJZoqK6iS2r")
SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"
SARVAM_EXHAUSTED = False

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_KEYS = [k.strip() for k in GROQ_API_KEY.split(",") if k.strip()] if GROQ_API_KEY else []
GEMINI_API_KEYS = [k.strip() for k in GEMINI_API_KEY.split(",") if k.strip()] if GEMINI_API_KEY else []
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

OUTPUT_DIR = Path("datasets/separated_datasets/batch_500_expansion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE_500 = OUTPUT_DIR / "sft_500_expansion.jsonl"
PART_A_FILE = OUTPUT_DIR / "part_a_pedagogy_200.jsonl"
PART_B_FILE = OUTPUT_DIR / "part_b_all_grades_curriculum_papers_200.jsonl"
PART_C_FILE = OUTPUT_DIR / "part_c_historic_naturalised_papers_100.jsonl"

MASTER_2655_FILE = Path("datasets/gold_standard_sft_2655_master.jsonl")
CHATML_2655_FILE = Path("datasets/gold_standard_sft_2655_chatml.jsonl")
SHAREGPT_2655_FILE = Path("datasets/gold_standard_sft_2655_sharegpt.json")
EXACT_3FIELD_2655_FILE = Path("datasets/gold_standard_sft_2655_3field.jsonl")


PROGRESS_MD = Path("GENERATION_500_PROGRESS.md")
PROGRESS_JSON = Path("GENERATION_500_PROGRESS.json")

# Concurrency & Rate Limiting
CONCURRENCY_LIMIT = 32
REQUEST_TIMEOUT_SECONDS = 45

# Load Regional Locale Facts
LOCALE_PATH = Path("locale.json")
VERIFIED_LOCALE_FACTS = [
    "দার্জিলিং জেলার পাহাড়ি ঢালে অম্লীয় দোআঁশ মাটিতে চা ও দার্জিলিং কমলার চাষ প্রধান (গড় বার্ষিক বৃষ্টিপাত ৩০৯২ মিমি, উৎস: IMD)।",
    "জলপাইগুড়ি ও আলিপুরদুয়ার জেলার ডুয়ার্স সমভূমিতে তিস্তা, তোর্সা ও জলঢাকা নদীর পলিমাটি উর্বর হওয়ায় ধান, পাট ও চা বাগান বিস্তৃত।",
    "মালদা জেলার মহানন্দা ও গঙ্গা তীরবর্তী সমতল পলিভূমিতে ফজলি আম ও পাট চাষ অত্যন্ত উন্নত।",
    "কোচবিহার জেলার কৃষিপ্রধান অর্থনীতিতে ধান ও তামাক চাষের বিশেষ ভূমিকা রয়েছে (মনরেগা দৈনিক মজুরি ₹২৫০)।",
    "উত্তর ও দক্ষিণ দিনাজপুর জেলায় বরো ধান, সরিষা ও পাট চাষ প্রধান গ্রামীণ জীবিকা।"
]

BENGALI_GRADES = {
    1: "প্রথম শ্রেণি", 2: "দ্বিতীয় শ্রেণি", 3: "তৃতীয় শ্রেণি", 4: "চতুর্থ শ্রেণি",
    5: "পঞ্চম শ্রেণি", 6: "ষষ্ঠ শ্রেণি", 7: "সপ্তম শ্রেণি", 8: "অষ্টম শ্রেণি",
    9: "নবম শ্রেণি", 10: "দশম শ্রেণি", 11: "একাদশ শ্রেণি", 12: "দ্বাদশ শ্রেণি"
}

def get_board_for_grade(grade: int) -> str:
    if 1 <= grade <= 5:
        return "WBBPE"
    elif 6 <= grade <= 10:
        return "WBBSE"
    else:
        return "WBCHSE"

def get_board_full_bengali(board: str) -> str:
    return {
        "WBBPE": "পশ্চিমবঙ্গ প্রাথমিক শিক্ষা পর্ষদ (WBBPE)",
        "WBBSE": "পশ্চিমবঙ্গ মধ্যশিক্ষা পর্ষদ (WBBSE)",
        "WBCHSE": "পশ্চিমবঙ্গ উচ্চমাধ্যমিক শিক্ষা সংসদ (WBCHSE)"
    }.get(board, "পশ্চিমবঙ্গ মধ্যশিক্ষা পর্ষদ (WBBSE)")

def load_all_textbook_chunks() -> List[Dict[str, Any]]:
    chunks = []
    chunk_paths = list(Path("datasets/extracted_textbooks_by_class").rglob("grounding_chunks.jsonl"))
    for p in chunk_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        c = json.loads(line)
                        if c.get("source_text") and len(c["source_text"].strip()) > 30:
                            chunks.append(c)
                    except Exception:
                        pass
    print(f"Loaded {len(chunks)} valid textbook grounding chunks from disk.", flush=True)
    return chunks

def load_historical_questions() -> List[Dict[str, Any]]:
    q_file = Path("datasets/question_papers_10_12_clean/historical_questions_10_12.jsonl")
    questions = []
    if q_file.exists():
        with open(q_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        q = json.loads(line)
                        if q.get("question_text") and len(q["question_text"].strip()) > 15:
                            questions.append(q)
                    except Exception:
                        pass
    print(f"Loaded {len(questions)} authentic historical board questions.", flush=True)
    return questions

def load_historical_papers() -> List[Dict[str, Any]]:
    p_file = Path("datasets/question_papers_10_12_clean/historical_papers_10_12.jsonl")
    papers = []
    if p_file.exists():
        with open(p_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        p = json.loads(line)
                        papers.append(p)
                    except Exception:
                        pass
    print(f"Loaded {len(papers)} historical board papers.", flush=True)
    return papers

def build_system_prompt(board: str, grade: int, subject: str, topic: str, target_role: str = "TEACHER", locale_fact: Optional[str] = None) -> str:
    board_full = get_board_full_bengali(board)
    grade_bengali = BENGALI_GRADES.get(grade, f"{grade}ম শ্রেণি")

    addressing_rule = (
        "১. শিক্ষক মহাশয়ের উদ্দেশ্যে সম্মানসূচক 'আপনি' সম্বোধন করবে এবং শিক্ষণ ও মূল্যায়ন-পদ্ধতি অনুযায়ী প্রাতিষ্ঠানিক প্রমিত কাঠামোতে উত্তর দেবে।"
        if target_role == "TEACHER"
        else "১. শিক্ষার্থীর উদ্দেশ্যে স্নেহপূর্ণ ও সহজবোধ্য 'তুমি/তোমরা' সম্বোধন করবে।"
    )

    locale_section = f"\n- আঞ্চলিক প্রেক্ষাপট (যাচাইকৃত তথ্য): {locale_fact}" if locale_fact else ""

    return f"""তুমি 'সহায়কএআই' (SahayakAI) — {board_full} বেঙ্গলি-মিডিয়াম স্কুলের শিক্ষক মহাশয় ও শিক্ষার্থীদের জন্য তৈরি একজন অভিজ্ঞ, বিশেষায়িত অ্যাকাডেমিক AI টিউটর ও প্রশ্নপত্র প্রণয়ন বিশেষজ্ঞ। তোমার কাজ হলো পাঠ্যক্রম ও প্রাতিষ্ঠানিক ব্লুপ্রিন্ট অনুযায়ী সহজ, নির্ভুল, সাবলীল বাংলা ভাষায় প্রমিত প্রশ্নপত্র, উত্তরমালা, পাঠ-পরিকল্পনা ও মূল্যায়ন সহায়তা তৈরি করা।

নিয়মাবলী ও নির্দেশিকা:
{addressing_rule}
২. স্পষ্ট, প্রাঞ্জল ও প্রমিত বাংলায় সরাসরি শিক্ষামূলক উত্তরে প্রবেশ করবে; অপ্রয়োজনীয় অভিবাদন বা চাটুকারিতা পরিহার করবে।
৩. সাধারণ বাংলা গদ্যে সংখ্যা বাংলা অঙ্কে (০, ১, ২, ৩, ৪, ৫, ৬, ৭, ৮, ৯) লিখবে।
৪. গাণিতিক ও বৈজ্ঞানিক সমীকরণ ও প্রতীকে প্রমিত LaTeX ও বৈজ্ঞানিক সংকেত (যেমন $x^2 + y^2 = r^2$, $H_2O$, $CO_2$, $\\Delta T$) যথাযথ বজায় রাখবে।
৫. কোনো কৃত্রিম সোর্স-রেফারেন্স বা মেটাডাটা উল্লেখ না করে স্বাভাবিকভাবে প্রমিত অ্যাকাডেমিক উত্তর উপস্থাপন করবে।

সহায়কএআই (SahayakAI) বিবরণী:
- পর্ষদ/সংসদ: {board_full}
- শ্রেণি: {grade_bengali}
- বিষয়: {subject}
- বিষয়বস্তু: {topic}{locale_section}"""

def build_part_a_work_items(textbook_chunks: List[Dict[str, Any]], count: int = 200) -> List[Dict[str, Any]]:
    tasks_cycle = [
        "lesson_plan", "quiz_generation", "lesson_plan", "quiz_generation",
        "concept_explanation", "word_problem", "socratic_dialogue", "error_spotting"
    ]
    items = []
    for i in range(count):
        slot_id = i + 1
        task_type = tasks_cycle[i % len(tasks_cycle)]
        chunk = random.choice(textbook_chunks) if textbook_chunks else {
            "grade": random.randint(1, 12), "subject": "Mathematics", "chapter": "সাধারণ পাঠ্যক্রম", "source_text": "পশ্চিমবঙ্গ পাঠ্যক্রমের প্রমিত ধারণা।"
        }
        grade = chunk.get("grade", random.randint(6, 10))
        board = chunk.get("board") or get_board_for_grade(grade)
        subject = chunk.get("subject", "Science")
        chapter = chunk.get("chapter") or chunk.get("topic") or "পাঠ্যক্রমের অধ্যায়"
        source_text = chunk.get("source_text", "").strip()

        locale_fact = VERIFIED_LOCALE_FACTS[slot_id % len(VERIFIED_LOCALE_FACTS)] if (i % 4 == 0) else None
        target_role = "TEACHER" if task_type in ["lesson_plan", "quiz_generation"] else "STUDENT"
        system_prompt = build_system_prompt(board, grade, subject, chapter, target_role, locale_fact)

        instructions = {
            "lesson_plan": f"শিক্ষক মহাশয়ের জন্য এই অধ্যায়ের ওপর একটি ক্লাসরুম-উপযোগী পূর্ণাঙ্গ পাঠ-পরিকল্পনা (Detailed Lesson Plan) তৈরি করুন। এতে থাকবে: (১) সাধারণ ও বিশেষ শিখন উদ্দেশ্য, (২) স্বল্পমূল্যের স্থানীয় শিক্ষাপ্রদীপ/উপকরণ (Low-Cost TLM), (৩) ৪০ মিনিটের সুবিন্যস্ত পর্যায়ক্রমিক শ্রেণি কার্যক্রম (Hook/আকর্ষণীয় সূচনা, মূল ধারণা উপস্থাপন, দলগত ও একক কাজ), (৪) শিখনে পিছিয়ে পড়া শিক্ষার্থীদের জন্য বিশেষ কৌশল, (৫) তাৎক্ষণিক গঠনমূলক মূল্যায়ন ও বাড়ির কাজ।",
            "quiz_generation": f"শিক্ষার্থীদের মূল্যায়নের জন্য এই পাঠ্যাংশের ওপর ভিত্তি করে একটি পূর্ণাঙ্গ প্রশ্নসেট (Quiz Paper) প্রস্তুত করুন: (১) ৩টি মানসম্মত MCQ (বাস্তবসম্মত distractors সহ), (২) ২টি সংক্ষিপ্ত ধারণামূলক প্রশ্ন, (৩) ১টি উচ্চতর চিন্তন দক্ষতাভিত্তিক সমস্যা (HOTS Problem), (৪) সম্পূর্ণ উত্তরমালা (Answer Key) ও পুঙ্খানুপুঙ্খ ব্যাখ্যা।",
            "concept_explanation": f"{BENGALI_GRADES.get(grade, f'{grade}ম শ্রেণি')}-র শিক্ষার্থীর জন্য এই ধারণার ওপর একটি প্রাঞ্জল, চিত্তাকর্ষক ও বাস্তব জীবনের উদাহরণসমৃদ্ধ কনসেপ্ট ব্যাখ্যা তৈরি করো।",
            "word_problem": f"এই পাঠ্যাংশের ওপর ভিত্তি করে একটি বাস্তবমুখী গাণিতিক/বিজ্ঞানভিত্তিক সমস্যা এবং তার ধাপে ধাপে বিস্তারিত সমাধান (LaTeX সমীকরণ সহ) প্রস্তুত করো।",
            "socratic_dialogue": f"শিক্ষার্থী এবং শিক্ষকের মধ্যে একটি আকর্ষণীয় সক্রেটিক প্রশ্নোত্তর সংলাপ (৩-৪ রাউন্ড) তৈরি করো, যেখানে শিক্ষক চিন্তাশীল প্রশ্নের মাধ্যমে শিক্ষার্থীকে সঠিক উত্তরের দিকে নিয়ে যান।",
            "error_spotting": f"এই ধারণায় শিক্ষার্থীরা সচরাচর যে সাধারণ ভুল বা ভ্রান্ত ধারণা (misconception) করে, তার একটি ভুল সমাধানের উদাহরণ তুলে ধরে বুঝিয়ে দাও কেন এটি ভুল এবং কীভাবে সঠিক সমাধান করতে হবে।"
        }
        task_desc = instructions.get(task_type, instructions["concept_explanation"])

        user_prompt = f"""পাঠ্যাংশ:
\"\"\"
{source_text[:1200]}
\"\"\"

প্রশ্ন / নির্দেশ:
{task_desc}"""

        primary_model = "sarvam-105b" if (grade >= 9 and any(s in subject.lower() for s in ["math", "physic", "chem", "গণিত"])) else "sarvam-105b-conversations"

        items.append({
            "part": "PART_A",
            "sub_type": task_type,
            "slot_id": slot_id,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "grade": grade,
            "board": board,
            "subject": subject,
            "chapter": chapter,
            "chosen_model": primary_model
        })
    return items

def build_part_b_work_items(textbook_chunks: List[Dict[str, Any]], count: int = 200) -> List[Dict[str, Any]]:
    grade_targets = [
        (1, 2, "WBBPE", 40, "Foundational Language & Numeracy", 15, "৪৫ মিনিট"),
        (3, 5, "WBBPE", 50, "Primary Summative Evaluation", 30, "১ ঘণ্টা"),
        (6, 8, "WBBSE", 60, "Middle School Summative Model Paper", 50, "১ ঘণ্টা ৩০ মিনিট"),
        (9, 9, "WBBSE", 30, "Pre-Madhyamik 5-Group Summative Paper", 70, "২ ঘণ্টা"),
        (11, 11, "WBCHSE", 20, "Higher Secondary Semester Exam Paper", 40, "১ ঘণ্টা ৩০ মিনিট")
    ]
    items = []
    slot_id = 200

    for min_g, max_g, board, num_papers, paper_type_name, total_marks, duration in grade_targets:
        stage_chunks = [c for c in textbook_chunks if min_g <= c.get("grade", 0) <= max_g]
        for i in range(num_papers):
            slot_id += 1
            chunk = random.choice(stage_chunks) if stage_chunks else random.choice(textbook_chunks)
            grade = chunk.get("grade") or random.randint(min_g, max_g)
            grade_bengali = BENGALI_GRADES.get(grade, f"{grade}ম শ্রেণি")
            subject = chunk.get("subject", "Mathematics")
            chapter = chunk.get("chapter") or chunk.get("topic") or "পাঠ্যক্রমের সম্পূর্ণ সিলেবাস"
            source_text = chunk.get("source_text", "").strip()

            system_prompt = build_system_prompt(board, grade, subject, chapter, "TEACHER")

            if grade <= 2:
                struct_desc = f"- পূর্ণমান: {total_marks} | সময়: {duration}\n- বিভাগ ক: সঠিক উত্তরে টিক চিহ্ন / বহু বিকল্পভিত্তিক প্রশ্ন (MCQ - ৩ নম্বর)\n- বিভাগ খ: শূন্যস্থান পূরণ ও এক কথায় উত্তর (VSA - ৪ নম্বর)\n- বিভাগ গ: সংক্ষিপ্ত বর্ণনামূলক ও চিত্র/গণনাভিত্তিক সমস্যা (SA - ৮ নম্বর)"
            elif grade <= 5:
                struct_desc = f"- পূর্ণমান: {total_marks} | সময়: {duration}\n- বিভাগ ক: বহুনির্বাচনী প্রশ্ন (MCQ - ৫ নম্বর)\n- বিভাগ খ: অতি সংক্ষিপ্ত উত্তরভিত্তিক প্রশ্ন ও সত্য/মিথ্যা (VSA - ৬ নম্বর)\n- বিভাগ গ: সংক্ষিপ্ত ব্যাখ্যামূলক ও গাণিতিক সমস্যা (SA - ১৪ নম্বর)\n- বিভাগ ঘ: দীর্ঘ উত্তরভিত্তিক সমস্যা (LA - ৫ নম্বর)"
            elif grade <= 8:
                struct_desc = f"- পূর্ণমান: {total_marks} | সময়: {duration}\n- গ্রুপ 'ক': সঠিক উত্তর নির্বাচন করো (MCQ - ৮ নম্বর)\n- গ্রুপ 'খ': অতি সংক্ষিপ্ত উত্তরভিত্তিক প্রশ্ন (VSA - ১০ নম্বর)\n- গ্রুপ 'গ': সংক্ষিপ্ত ব্যাখ্যামূলক প্রশ্ন ও গাণিতিক সমস্যা (SA ২ নম্বর করে - ১৬ নম্বর)\n- গ্রুপ 'ঘ': দীর্ঘ উত্তরভিত্তিক ও বিশ্লেষণধর্মী সমস্যা (LA ৪-৫ নম্বর করে - ১৬ নম্বর)"
            elif grade == 9:
                struct_desc = f"- পূর্ণমান: {total_marks} | সময়: {duration}\n- Group A: বহুনির্বাচনী প্রশ্ন (MCQ - ১০ নম্বর)\n- Group B: অতি সংক্ষিপ্ত উত্তরভিত্তিক প্রশ্ন (VSA - ১৫ নম্বর)\n- Group C: সংক্ষিপ্ত ধারণামূলক প্রশ্ন (SA ২ নম্বর করে - ২০ নম্বর)\n- Group D: দীর্ঘ বিশ্লেষণধর্মী ও প্রমাণ/উপপাদ্য/সমস্যা (LA ৩-৫ নম্বর করে - ২৫ নম্বর)"
            else:
                struct_desc = f"- পূর্ণমান: {total_marks} | সময়: {duration}\n- Part B: বহুনির্বাচনী প্রশ্ন (MCQ - ১৫ নম্বর)\n- Part A: সংক্ষিপ্ত ও দীর্ঘ ব্যাখ্যামূলক সমস্যা ও উত্তর (Descriptive - ২৫ নম্বর)"

            user_prompt = f"""পাঠ্যক্রম বিষয়বস্তু ও মূল সূত্র:
\"\"\"
{source_text[:1200]}
\"\"\"

নির্দেশ:
{get_board_full_bengali(board)}-র পাঠ্যক্রম এবং ব্লুপ্রিন্ট অনুযায়ী {grade_bengali}-র {subject} বিষয়ের উপর একটি সম্পূর্ণ মডেল প্রশ্নপত্র এবং তার পুঙ্খানুপুঙ্খ উত্তরমালা ও মার্কিং স্কিম প্রস্তুত করে দিন।

প্রশ্নপত্রের কাঠামো ও নম্বর বিভাজন:
{struct_desc}

উত্তরে স্পষ্টভাবে নিচের ৩টি মূল বিভাগ রাখবেন:
## ১. [প্রশ্নপত্র] (স্পষ্ট নির্দেশাবলী ও নম্বর সহ)
## ২. [উত্তর নির্দেশিকা ও সম্পূর্ণ সমাধানমালা] (প্রতিটি প্রশ্নের বিশদ ও নির্ভুল সমাধান)
## ৩. [নম্বর বণ্টন ও মার্কিং স্কিম] (ধাপভিত্তিক মূল্যায়ন নির্দেশিকা)"""

            items.append({
                "part": "PART_B",
                "sub_type": "curriculum_question_paper",
                "slot_id": slot_id,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "grade": grade,
                "board": board,
                "subject": subject,
                "chapter": chapter,
                "chosen_model": "sarvam-105b" if grade >= 9 else "sarvam-105b-conversations"
            })
    return items

def build_part_c_work_items(hist_questions: List[Dict[str, Any]], hist_papers: List[Dict[str, Any]], count: int = 100) -> List[Dict[str, Any]]:
    items = []
    slot_id = 400

    madhyamik_qs = [q for q in hist_questions if str(q.get("class_level")) in ["10", "X", "Madhyamik"]]
    hs_qs = [q for q in hist_questions if str(q.get("class_level")) in ["12", "XII", "HS", "Higher Secondary"]]

    for i in range(count):
        slot_id += 1
        is_class_10 = (i % 2 == 0)

        if is_class_10:
            grade = 10
            board = "WBBSE"
            subject = random.choice(["Mathematics", "Physical Science", "Life Science", "History", "Geography"])
            grade_bengali = "দশম শ্রেণি"
            pool = [q for q in madhyamik_qs if subject.lower() in q.get("subject", "").lower()] or madhyamik_qs
            sampled_qs = random.sample(pool, min(6, len(pool))) if pool else []
            q_texts = "\n".join([f"- ({q.get('question_type', 'প্রশ্ন')} / {q.get('marks', 1)} নম্বর): {q.get('question_text', '')}" for q in sampled_qs])

            system_prompt = build_system_prompt(board, grade, subject, "মাধ্যমিক পরীক্ষা পূর্ণাঙ্গ পাঠ্যসূচি", "TEACHER")
            user_prompt = f"""প্রমাণিত পর্ষদ প্রশ্ন-নমুনা ও ধারণাসমূহ:
\"\"\"
{q_texts[:1400]}
\"\"\"

নির্দেশ:
পশ্চিমবঙ্গ মধ্যশিক্ষা পর্ষদ (WBBSE) দশম শ্রেণির {subject} বিষয়ের মাধ্যমিক পরীক্ষার পূর্ণাঙ্গ বোর্ড-ধাঁচের মডেল প্রশ্নপত্র, বিস্তারিত সমাধান ও স্টেপ-বাই-স্টেপ মার্কিং স্কিম প্রস্তুত করে দিন।

কাঠামো:
- Group A: বহুনির্বাচনী প্রশ্ন (MCQ) — সঠিক বিকল্প নির্বাচন
- Group B: অতি সংক্ষিপ্ত উত্তরভিত্তিক প্রশ্ন (VSA) — শূন্যস্থান পূরণ, সত্য/মিথ্যা ও সংক্ষিপ্ত উত্তর
- Group C: সংক্ষিপ্ত উত্তরভিত্তিক প্রশ্ন (SA - ২ নম্বর করে)
- Group D: দীর্ঘ বিশ্লেষণধর্মী, গাণিতিক বা প্রয়োগমূলক সমস্যা (LA - ৩-৫ নম্বর করে)

উত্তরে স্পষ্টভাবে নিচের বিভাগগুলো অন্তর্ভুক্ত থাকবে:
## ১. [মডেল প্রশ্নপত্র]
## ২. [সম্পূর্ণ সমাধানমালা ও উত্তর নির্দেশিকা]
## ৩. [ধাপভিত্তিক নম্বর বণ্টন ও মূল্যায়ন নীতি]"""

        else:
            grade = 12
            board = "WBCHSE"
            subject = random.choice(["Physics", "Chemistry", "Mathematics", "Biological Sciences", "Bengali", "English", "History", "Geography"])
            grade_bengali = "দ্বাদশ শ্রেণি"
            pool = [q for q in hs_qs if subject.lower() in q.get("subject", "").lower()] or hs_qs
            sampled_qs = random.sample(pool, min(6, len(pool))) if pool else []
            q_texts = "\n".join([f"- ({q.get('question_type', 'প্রশ্ন')} / {q.get('marks', 1)} নম্বর): {q.get('question_text', '')}" for q in sampled_qs])

            system_prompt = build_system_prompt(board, grade, subject, "উচ্চমাধ্যমিক পরীক্ষা পূর্ণাঙ্গ পাঠ্যসূচি", "TEACHER")
            user_prompt = f"""প্রমাণিত উচ্চমাধ্যমিক প্রশ্ন-নমুনা ও বিষয়বস্তু:
\"\"\"
{q_texts[:1400]}
\"\"\"

নির্দেশ:
পশ্চিমবঙ্গ উচ্চমাধ্যমিক শিক্ষা সংসদ (WBCHSE) দ্বাদশ শ্রেণির {subject} বিষয়ের উচ্চমাধ্যমিক পরীক্ষার অফিসিয়াল ব্লুপ্রিন্ট অনুযায়ী একটি সমৃদ্ধ মডেল প্রশ্নপত্র, পূর্ণাঙ্গ সমাধানমালা ও নম্বর বিভাজন প্রস্তুত করে দিন।

কাঠামো:
- Part B (Objective): বহুনির্বাচনী প্রশ্ন (MCQ) ও অতি সংক্ষিপ্ত প্রশ্ন (VSA)
- Part A (Subjective): সংক্ষিপ্ত ব্যাখ্যামূলক, গাণিতিক এবং দীর্ঘ বিশ্লেষণধর্মী ও বর্ণনামূলক সমস্যা

উত্তরে স্পষ্টভাবে অন্তর্ভুক্ত থাকবে:
## ১. [উচ্চমাধ্যমিক মডেল প্রশ্নপত্র]
## ২. [উত্তর নির্দেশিকা ও সম্পূর্ণ সমাধানমালা]
## ৩. [মার্কিং স্কিম ও নম্বর বিভাজন]"""

        items.append({
            "part": "PART_C",
            "sub_type": "historic_naturalised_board_paper",
            "slot_id": slot_id,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "grade": grade,
            "board": board,
            "subject": subject,
            "chapter": "বোর্ড পরীক্ষা পূর্ণাঙ্গ সিলেবাস",
            "chosen_model": "sarvam-105b"
        })
    return items

async def call_sarvam_api(session: aiohttp.ClientSession, prompt_pkg: Dict[str, Any], semaphore: asyncio.Semaphore) -> Tuple[Optional[str], Optional[str]]:
    global SARVAM_EXHAUSTED
    if SARVAM_EXHAUSTED:
        return None, None

    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }

    model = prompt_pkg["chosen_model"]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt_pkg["system_prompt"]},
            {"role": "user", "content": prompt_pkg["user_prompt"]}
        ],
        "temperature": 0.7,
        "max_tokens": 1500
    }

    async with semaphore:
        try:
            async with session.post(SARVAM_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    return content, model
                elif resp.status in (402, 403):
                    SARVAM_EXHAUSTED = True
                    return None, None
        except Exception:
            pass
    return None, None

async def call_gemini_api(session: aiohttp.ClientSession, prompt_pkg: Dict[str, Any], semaphore: asyncio.Semaphore) -> Tuple[Optional[str], Optional[str]]:
    slot_id = prompt_pkg.get("slot_id", 0)
    for k_idx in range(len(GEMINI_API_KEYS)):
        api_key = GEMINI_API_KEYS[(slot_id + k_idx) % len(GEMINI_API_KEYS)]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        payload = {
            "systemInstruction": {
                "parts": [{"text": prompt_pkg["system_prompt"]}]
            },
            "contents": [
                {"role": "user", "parts": [{"text": prompt_pkg["user_prompt"]}]}
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048
            }
        }
        headers = {"Content-Type": "application/json"}

        async with semaphore:
            try:
                async with session.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        candidates = data.get("candidates", [])
                        if candidates and "content" in candidates[0]:
                            parts = candidates[0]["content"].get("parts", [])
                            if parts and "text" in parts[0]:
                                return parts[0]["text"].strip(), "gemini-2.5-flash"
                    elif resp.status == 429:
                        continue
            except Exception:
                pass
    return None, None

async def call_groq_api(session: aiohttp.ClientSession, prompt_pkg: Dict[str, Any], semaphore: asyncio.Semaphore) -> Tuple[Optional[str], Optional[str]]:
    slot_id = prompt_pkg.get("slot_id", 0)
    models_to_try = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
    if slot_id % 2 != 0:
        models_to_try.reverse()

    for m_name in models_to_try:
        for k_idx in range(len(GROQ_API_KEYS)):
            api_key = GROQ_API_KEYS[(slot_id + k_idx) % len(GROQ_API_KEYS)]
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0"
            }

            payload = {
                "model": m_name,
                "messages": [
                    {"role": "system", "content": prompt_pkg["system_prompt"]},
                    {"role": "user", "content": prompt_pkg["user_prompt"]}
                ],
                "temperature": 0.7,
                "max_tokens": 1500
            }

            async with semaphore:
                try:
                    async with session.post(GROQ_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            content = data["choices"][0]["message"]["content"].strip()
                            if len(content) > 50:
                                return content, m_name
                        elif resp.status == 429:
                            continue
                except Exception:
                    pass
    return None, None

async def call_llm_pipeline(session: aiohttp.ClientSession, prompt_pkg: Dict[str, Any], semaphore: asyncio.Semaphore) -> Tuple[Optional[str], str]:
    # 1. Primary: Sarvam AI Indic LLM
    text, model_used = await call_sarvam_api(session, prompt_pkg, semaphore)
    if text and len(text.strip()) > 50:
        return text, model_used

    # 2. Secondary: Groq 4-Key Pool
    text, model_used = await call_groq_api(session, prompt_pkg, semaphore)
    if text and len(text.strip()) > 50:
        return text, model_used

    # 3. Tertiary: Google Gemini Pool
    text, model_used = await call_gemini_api(session, prompt_pkg, semaphore)
    if text and len(text.strip()) > 50:
        return text, model_used

    return None, prompt_pkg["chosen_model"]

def update_progress_files(total_saved: int, target_records: int, start_time: float, part_counts: Dict[str, int], model_counts: Dict[str, int]):
    elapsed = time.time() - start_time
    rate = total_saved / elapsed if elapsed > 0 else 0
    remaining_records = max(0, target_records - total_saved)
    eta_seconds = remaining_records / rate if rate > 0 else 0
    eta_min = eta_seconds / 60.0
    percent = (total_saved / target_records) * 100.0 if target_records > 0 else 0

    status_data = {
        "status": "RUNNING" if total_saved < target_records else "COMPLETED",
        "completed": total_saved,
        "target": target_records,
        "percent": round(percent, 2),
        "speed_records_per_sec": round(rate, 2),
        "speed_records_per_min": round(rate * 60, 2),
        "elapsed_seconds": round(elapsed, 1),
        "eta_minutes": round(eta_min, 1),
        "parts": part_counts,
        "models": model_counts,
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    with open(PROGRESS_JSON, "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2, ensure_ascii=False)

    md_content = f"""# 📊 SahayakAI 500-Expansion Batch Generation Monitor

**Status**: `{"🟢 RUNNING" if total_saved < target_records else "✅ COMPLETED"}`  
**Progress**: **{total_saved} / {target_records}** records (**{percent:.1f}%**)  
**Format**: `Pure ChatML messages (Zero Metadata Wrappers)`  
**Throughput**: `{rate:.2f}` records/sec (`{rate*60:.1f}` records/min)  
**Elapsed**: `{elapsed/60:.1f}` mins | **Estimated Remaining (ETA)**: `{eta_min:.1f}` mins  
**Last Updated**: `{status_data["last_updated"]}`  

---

### 📦 Part Breakdown
| Part | Category | Target | Completed | Progress |
| :--- | :--- | :--- | :--- | :--- |
| **Part A** | Pedagogical Dialogues (Lesson Plans, Quizzes, Q/A) | 200 | `{part_counts.get("PART_A", 0)}` | `{(part_counts.get("PART_A", 0)/200)*100:.1f}%` |
| **Part B** | All-Grade Curriculum Blueprint Papers (Classes 1–9 & 11) | 200 | `{part_counts.get("PART_B", 0)}` | `{(part_counts.get("PART_B", 0)/200)*100:.1f}%` |
| **Part C** | Historic Grounded & Naturalised Board Papers (Classes 10 & 12) | 100 | `{part_counts.get("PART_C", 0)}` | `{(part_counts.get("PART_C", 0)/100)*100:.1f}%` |

---

### 🤖 LLM Engine Usage
"""
    for model_name, m_count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
        md_content += f"- `{model_name}`: **{m_count}** records\n"

    with open(PROGRESS_MD, "w", encoding="utf-8") as f:
        f.write(md_content)

def merge_and_export_all():
    print("\n--- Merging 500 Expansion with Base 2,155 Records ---", flush=True)
    base_file = Path("datasets/separated_datasets/combined_2155/sft_chatml_2155.jsonl")
    if not base_file.exists():
        base_file = Path("datasets/gold_standard_sft_2000_api.jsonl")

    all_2655_records = []
    
    # 1. Load Base 2,155
    if base_file.exists():
        with open(base_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if "messages" in rec:
                            all_2655_records.append({"messages": rec["messages"]})
                    except Exception:
                        pass
    print(f"Loaded {len(all_2655_records)} base records.", flush=True)

    # 2. Load 500 Expansion
    expansion_records = []
    if OUTPUT_FILE_500.exists():
        with open(OUTPUT_FILE_500, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if "messages" in rec:
                            expansion_records.append({"messages": rec["messages"]})
                    except Exception:
                        pass
    print(f"Loaded {len(expansion_records)} expansion records.", flush=True)

    all_2655_records.extend(expansion_records)
    print(f"Total Master Records: {len(all_2655_records)}", flush=True)

    # 3. Write Master File
    with open(MASTER_2655_FILE, "w", encoding="utf-8") as f:
        for r in all_2655_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ Saved Master 2655 File to {MASTER_2655_FILE}", flush=True)

    # 4. Write ChatML Format
    with open(CHATML_2655_FILE, "w", encoding="utf-8") as f:
        for r in all_2655_records:
            f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    print(f"✅ Exported ChatML 2655 to {CHATML_2655_FILE}", flush=True)

    # 5. Write ShareGPT Format
    sharegpt_list = []
    for i, r in enumerate(all_2655_records):
        conversations = []
        for m in r["messages"]:
            from_role = "system" if m["role"] == "system" else ("human" if m["role"] == "user" else "gpt")
            conversations.append({"from": from_role, "value": m["content"]})
        sharegpt_list.append({"id": f"sahayak_sft_{i+1:05d}", "conversations": conversations})

    with open(SHAREGPT_2655_FILE, "w", encoding="utf-8") as f:
        json.dump(sharegpt_list, f, ensure_ascii=False, indent=2)
    print(f"✅ Exported ShareGPT 2655 to {SHAREGPT_2655_FILE}", flush=True)

    # 6. Write Exact 3-Field Format
    with open(EXACT_3FIELD_2655_FILE, "w", encoding="utf-8") as f:
        for r in all_2655_records:
            system_txt = r["messages"][0]["content"]
            user_txt = r["messages"][1]["content"]
            assistant_txt = r["messages"][2]["content"]
            f.write(json.dumps({
                "instruction": system_txt,
                "input": user_txt,
                "output": assistant_txt
            }, ensure_ascii=False) + "\n")
    print(f"✅ Exported Exact 3-Field 2655 to {EXACT_3FIELD_2655_FILE}", flush=True)

async def generate_500_dataset():
    print("Preparing 500 specialized expansion work items...", flush=True)
    textbook_chunks = load_all_textbook_chunks()
    hist_questions = load_historical_questions()
    hist_papers = load_historical_papers()

    part_a_items = build_part_a_work_items(textbook_chunks, count=200)
    part_b_items = build_part_b_work_items(textbook_chunks, count=200)
    part_c_items = build_part_c_work_items(hist_questions, hist_papers, count=100)

    all_work_items = part_a_items + part_b_items + part_c_items
    target_records = len(all_work_items) # 500

    print(f"Total Expansion Items: {target_records} (Part A: {len(part_a_items)}, Part B: {len(part_b_items)}, Part C: {len(part_c_items)})", flush=True)

    # Check for existing generated files
    existing_records = []
    part_counts = defaultdict(int)
    model_counts = defaultdict(int)

    if OUTPUT_FILE_500.exists():
        with open(OUTPUT_FILE_500, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if "messages" in rec and len(rec["messages"]) == 3:
                            existing_records.append(rec)
                    except Exception:
                        pass

    total_saved = len(existing_records)
    for i in range(min(total_saved, len(all_work_items))):
        item = all_work_items[i]
        part_counts[item["part"]] += 1
        model_counts[item["chosen_model"]] += 1

    print(f"Existing records found in 500 expansion: {total_saved}. Resuming from index {total_saved}...", flush=True)

    remaining_items = all_work_items[total_saved:]
    queue = deque(remaining_items)

    start_time = time.time()
    update_progress_files(total_saved, target_records, start_time, part_counts, model_counts)

    if total_saved >= target_records:
        print("All 500 expansion records already generated! Proceeding to merge and export.", flush=True)
        merge_and_export_all()
        return

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY_LIMIT * 2)

    async with aiohttp.ClientSession(connector=connector) as session:
        batch_size = 32
        while total_saved < target_records and queue:
            batch = []
            for _ in range(min(batch_size, len(queue))):
                batch.append(queue.popleft())

            tasks = [call_llm_pipeline(session, item, semaphore) for item in batch]
            results = await asyncio.gather(*tasks)

            new_saved = 0
            # Open output file handles
            with open(OUTPUT_FILE_500, "a", encoding="utf-8") as out_500_f, \
                 open(PART_A_FILE, "a", encoding="utf-8") as pa_f, \
                 open(PART_B_FILE, "a", encoding="utf-8") as pb_f, \
                 open(PART_C_FILE, "a", encoding="utf-8") as pc_f:

                for item, (response_text, model_used) in zip(batch, results):
                    if response_text and len(response_text.strip()) > 50:
                        clean_record = {
                            "messages": [
                                {"role": "system", "content": item["system_prompt"]},
                                {"role": "user", "content": item["user_prompt"]},
                                {"role": "assistant", "content": response_text}
                            ]
                        }
                        line_str = json.dumps(clean_record, ensure_ascii=False) + "\n"
                        out_500_f.write(line_str)

                        if item["part"] == "PART_A":
                            pa_f.write(line_str)
                        elif item["part"] == "PART_B":
                            pb_f.write(line_str)
                        elif item["part"] == "PART_C":
                            pc_f.write(line_str)

                        total_saved += 1
                        new_saved += 1
                        part_counts[item["part"]] += 1
                        model_counts[model_used or item["chosen_model"]] += 1
                    else:
                        queue.append(item)

            update_progress_files(total_saved, target_records, start_time, part_counts, model_counts)
            elapsed = time.time() - start_time
            rate = total_saved / elapsed if elapsed > 0 else 0
            print(f"Progress: [{total_saved}/{target_records}] ({total_saved/target_records*100:.1f}%) | Part A: {part_counts['PART_A']}/200 | Part B: {part_counts['PART_B']}/200 | Part C: {part_counts['PART_C']}/100 | Rate: {rate:.2f} rec/s", flush=True)

            if new_saved == 0:
                await asyncio.sleep(2.0)
            else:
                await asyncio.sleep(0.2)

    print(f"\nSuccessfully generated all {total_saved} specialized expansion records!", flush=True)
    merge_and_export_all()

if __name__ == "__main__":
    asyncio.run(generate_500_dataset())
