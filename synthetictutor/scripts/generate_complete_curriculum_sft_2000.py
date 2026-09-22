"""
Master Part 1 SFT Dataset Generator (2,000 Unique Records).
Strictly adheres to:
1. 2,000 Total Records (1,000 Q/A, 500 Lesson Plans, 500 Quizzes)
2. Stage allocation: Primary (600), Upper Primary (600), Secondary (500), Higher Secondary (300)
3. Zero repetitive boilerplate; distinct pedagogical behaviors and real curriculum depth
4. ChatML format: {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
5. Correct Bengali numerals for classes (১ম, ২য়, ৩য়, ৪র্থ, ৫ম, ৬ষ্ঠ, ৭ম, ৮ম, ৯ম, ১০ম, ১১শ, ১২শ)
6. Authentic topic content (LaTeX notation, step-by-step solutions, separated quiz keys, 5-stage lesson plans)
7. Grounded local context from locale.json where appropriate
"""

import os
import json
import random
from typing import Dict, Any, List, Optional
from pathlib import Path

# Load verified locale data
LOCALE_PATH = Path("locale.json")
with open(LOCALE_PATH, "r", encoding="utf-8") as f:
    LOCALE_DATA = json.load(f)

def get_bengali_grade_str(grade: int) -> str:
    grades = {
        1: "১ম শ্রেণি (প্রথম শ্রেণি)",
        2: "২য় শ্রেণি (দ্বিতীয় শ্রেণি)",
        3: "৩য় শ্রেণি (তৃতীয় শ্রেণি)",
        4: "৪র্থ শ্রেণি (চতুর্থ শ্রেণি)",
        5: "৫ম শ্রেণি (পঞ্চম শ্রেণি)",
        6: "৬ষ্ঠ শ্রেণি (ষষ্ঠ শ্রেণি)",
        7: "৭ম শ্রেণি (সপ্তম শ্রেণি)",
        8: "৮ম শ্রেণি (অষ্টম শ্রেণি)",
        9: "৯ম শ্রেণি (নবম শ্রেণি)",
        10: "১০ম শ্রেণি (দশম শ্রেণি)",
        11: "১১শ শ্রেণি (একাদশ শ্রেণি)",
        12: "১২শ শ্রেণি (দ্বাদশ শ্রেণি)"
    }
    return grades.get(grade, f"{grade}ম শ্রেণি")

def build_system_prompt(board: str, grade: int, subject: str, topic: str, target_role: str = "STUDENT", locale_fact: Optional[str] = None) -> str:
    board_full = {
        "WBBPE": "পশ্চিমবঙ্গ প্রাথমিক শিক্ষা পর্ষদ (WBBPE)",
        "WBBSE": "পশ্চিমবঙ্গ মধ্যশিক্ষা পর্ষদ (WBBSE)",
        "WBCHSE": "পশ্চিমবঙ্গ উচ্চমাধ্যমিক শিক্ষা সংসদ (WBCHSE)"
    }.get(board, "পশ্চিমবঙ্গ মধ্যশিক্ষা পর্ষদ (WBBSE)")

    grade_bengali = get_bengali_grade_str(grade)

    addressing_rule = (
        "১. শিক্ষক মহাশয়ের উদ্দেশ্যে সম্মানসূচক 'আপনি' সম্বোধন করবে এবং শিক্ষণ-পদ্ধতি অনুযায়ী দিকনির্দেশ প্রদান করবে।"
        if target_role == "TEACHER"
        else "১. শিক্ষার্থীর উদ্দেশ্যে স্নেহপূর্ণ ও সহজবোধ্য 'তুমি/তোমরা' সম্বোধন করবে।"
    )

    locale_section = f"\n- আঞ্চলিক প্রেক্ষাপট (যাচাইকৃত তথ্য): {locale_fact}" if locale_fact else ""

    return f"""তুমি 'সহায়কএআই' (SahayakAI) — {board_full} বেঙ্গলি-মিডিয়াম স্কুলের শিক্ষার্থী ও শিক্ষক মহাশয়দের জন্য তৈরি একজন অভিজ্ঞ, বিশেষায়িত অ্যাকাডেমিক AI টিউটর। তোমার কাজ হলো পাঠ্যক্রম অনুযায়ী সহজ, সাবলীল, সঠিক ও শিক্ষার্থী-উপযোগী বাংলা ভাষায় পাঠদান, ধারণা ব্যাখ্যা, সমস্যা সমাধান এবং অধ্যয়ন-সংক্রান্ত সহায়তা প্রদান করা।

নিয়মাবলী ও সম্বোধন:
{addressing_rule}
২. স্পষ্ট, প্রাঞ্জল ও স্বাভাবিক ভাষায় সরাসরি শিক্ষামূলক উত্তরে প্রবেশ করবে; অপ্রয়োজনীয় অভিবাদন, চাটুকারিতা বা অতিরিক্ত ভূমিকা পরিহার করবে।
৩. পাঠ্যক্রম-উপযোগী প্রমিত বাংলা পরিভাষা ব্যবহার করবে। সাধারণ বাংলা গদ্যে সংখ্যা বাংলা অঙ্কে (০, ১, ২, ৩, ৪, ৫, ৬, ৭, ৮, ৯) লিখবে।
৪. বৈধ mathematical/scientific notation যেমন x², x^2, H₂O, CO₂, ∠ABC যথাযথ রাখবে।
৫. কোনো কৃত্রিম সোর্স-রেফারেন্স (যেমন 'প্রদত্ত পাঠ্যাংশে' বা 'উক্ত চাঙ্কে') উল্লেখ না করে স্বাভাবিকভাবে বিষয়বস্তু উপস্থাপন করবে।

সহায়কএআই (SahayakAI) পাঠ্যসূচি বিবরণী:
- পর্ষদ/সংসদ: {board_full}
- শ্রেণি: {grade_bengali}
- বিষয়: {subject}
- বিষয়বস্তু: {topic}{locale_section}"""

print("Helper setup complete.")
