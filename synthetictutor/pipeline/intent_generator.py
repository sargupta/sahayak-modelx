"""
Intent and Query Synthesizer / Classifier with Dual-Role Modeling for SahayakAI SFT Pipeline.
Decouples requester_role (TEACHER 85%, STUDENT 15%) from instructional_target (STUDENT, TEACHER, BOTH).
"""

import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class UserIntent:
    query: str
    target_subject: str
    target_grade: Any
    target_topic: str
    task_type: str
    requester_role: str  # "TEACHER" or "STUDENT"
    instructional_target: str  # "STUDENT", "TEACHER", or "BOTH"
    audience: str  # Legacy compat string
    is_user_provided_source: bool = False
    provided_source_text: Optional[str] = None


class IntentGenerator:
    """Generates teacher-centric and student educational requests with dual-role modeling."""

    # Teacher-facing requests (85% target)
    TEACHER_TEMPLATES = {
        # Teacher -> Student target
        ("CONCEPT_EXPLANATION", "STUDENT"): [
            "{grade} শ্রেণির শিক্ষার্থীদের জন্য '{topic}' বিষয়টি সহজ ও আকর্ষণীয়ভাবে কীভাবে বুঝিয়ে বলব?",
            "শ্রেণিকক্ষে শিক্ষার্থীদের '{topic}' ধারণাটি স্পষ্ট করার জন্য একটি কার্যকর ও সহজবোধ্য ব্যাখ্যা দিন।",
            "সপ্তম শ্রেণির শিক্ষার্থীদের '{topic}' এমনভাবে বুঝিয়ে দিন যাতে আমি ক্লাসে ব্ল্যাকবোর্ডে সহজে বোঝাতে পারি।",
            "শিক্ষার্থীদের '{topic}' বিষয়ে প্রাথমিক ধারণা দেওয়ার জন্য একটি বাস্তবসম্মত উদাহরণসহ ব্যাখ্যা দিন।"
        ],
        ("GUIDED_PROBLEM_SOLVING", "STUDENT"): [
            "ক্লাসে শিক্ষার্থীদের '{topic}' সম্পর্কিত অঙ্কের সমাধান ধাপে ধাপে কীভাবে বোঝাব?",
            "{grade} শ্রেণির শিক্ষার্থীদের জন্য '{topic}'-এর নিয়ম ব্যবহার করে সমস্যা সমাধানের একটি সহজ পদ্ধতি বুঝিয়ে দিন।",
            "শিক্ষার্থীরা যাতে নিজে নিজে '{topic}'-এর সমস্যা সমাধান করতে পারে, তার জন্য একটি আদর্শ উদাহরণসহ সমাধান প্রস্তুত করুন।"
        ],
        ("MISCONCEPTION_CORRECTION", "STUDENT"): [
            "শিক্ষার্থীরা '{topic}' অধ্যায়ে সাধারণত কোন কোন জায়গায় ভুল করে এবং তাদের এই ভুল ধারণা কীভাবে সংশোধন করব?",
            "শ্রেণিকক্ষে '{topic}' পড়ানোর সময় শিক্ষার্থীদের প্রচলিত বিভ্রান্তিগুলো দূর করতে কী কী সতর্কতা অবলম্বন করা উচিত?",
            "অনেকে ভাবে '{topic}'-এ নিয়ম সবসময় একই থাকে। শিক্ষার্থীদের এই ভুল যুক্তিটি কীভাবে যুক্তি দিয়ে শুধরে দেব?"
        ],
        ("WORKSHEET_PRACTICE", "STUDENT"): [
            "{grade} শ্রেণির শিক্ষার্থীদের শ্রেণিকক্ষে ও বাড়িতে অনুশীলনের জন্য '{topic}' থেকে একটি সুন্দর ওয়ার্কশিট তৈরি করে দিন।",
            "শিক্ষার্থীদের শব্দার্থ, শূন্যস্থান পূরণ ও অনুশীলনের জন্য '{topic}' অংশ থেকে একটি অ্যাক্টিভিটি শিট প্রস্তুত করুন।",
            "জলপাইগুড়ি ও আলিপুরদুয়ার জেলার চা-বাগান অঞ্চলের শিক্ষার্থীদের জন্য '{topic}' বিষয়ে স্থানীয় বাস্তব উদাহরণসহ একটি অনুশীলনপত্র তৈরি করুন।",
            "সুন্দরবন উপকূলীয় অঞ্চলের শিক্ষার্থীদের পরিবেশ ও জীবিকার প্রেক্ষাপটে '{topic}' থেকে একটি কর্মপত্র (Worksheet) প্রস্তুত করুন।",
            "পুরুলিয়া ও বাঁকুড়া জেলার গ্রামীণ বিদ্যালয়ের শিক্ষার্থীদের উপযোগী করে '{topic}' বিষয়ের ওপর একটি অনুশীলনপত্র তৈরি করে দিন।",
            "পূর্ব বর্ধমান ও নদিয়া জেলার কৃষিকাজ ও স্থানীয় এককের (বিঘা, কাঠা) সাথে মিলিয়ে '{topic}' বিষয়ে একটি গণিত ওয়ার্কশিট তৈরি করুন।"
        ],
        ("QUIZ_GENERATION", "STUDENT"): [
            "{grade} শ্রেণির {subject} বিষয়ের '{topic}' থেকে শিক্ষার্থীদের জ্ঞান যাচাইয়ের জন্য একটি ১৫ নম্বরের কুইজ প্রশ্নপত্র তৈরি করে দিন।",
            "শ্রেণিকক্ষে দ্রুত মূল্যায়নের জন্য '{topic}' অধ্যায় থেকে ৫টি সংক্ষিপ্ত ও ৫টি বহুনির্বাচনী (MCQ) প্রশ্ন তৈরি করুন।"
        ],

        # Teacher -> Teacher target
        ("LESSON_PLAN", "TEACHER"): [
            "{grade} শ্রেণির {subject} বিষয়ের '{topic}' পাঠটির ওপর একটি পূর্ণাঙ্গ ৪০ মিনিটের শিক্ষণ পরিকল্পনা (Lesson Plan) তৈরি করে দিন।",
            "শ্রেণিকক্ষে '{topic}' পড়ানোর জন্য শিক্ষণ উদ্দেশ্য, প্রয়োজনীয় টিএলএম (TLM) ও মূল্যায়নসহ একটি বিস্তারিত পাঠ পরিকল্পনা দিন।",
            "শিক্ষার্থীদের সক্রিয় অংশগ্রহণের ভিত্তিতে '{topic}' অধ্যায়টি পড়ানোর জন্য একটি শিক্ষাদান নকশা প্রস্তুত করুন।"
        ],
        ("TEACHER_PEDAGOGY", "TEACHER"): [
            "{grade} শ্রেণিতে '{topic}' পড়ানোর আগে শিক্ষার্থীদের পূর্বজ্ঞান যাচাই করতে কোন কোন প্রশ্ন করা যেতে পারে?",
            "বিদ্যালয়ে দুর্বল শিক্ষার্থীদের '{topic}' ধারণাটি সহজে বোঝাতে কোন বিশেষ শিক্ষণ কৌশল বা টিএলএম ব্যবহার করা যায়?",
            "শ্রেণিকক্ষে '{topic}' পাঠদানের সময় পাঠের ধারাবাহিকতা বজায় রাখতে শিক্ষকের কী কী ধাপ অনুসরণ করা উচিত?"
        ]
    }

    # Direct Student requests (15% target)
    STUDENT_TEMPLATES = {
        ("CONCEPT_EXPLANATION", "STUDENT"): [
            "{topic} বিষয়টা একটু সহজ বাংলায় বুঝিয়ে দেবে?",
            "আমি পাঠ্যবইয়ের {topic} অংশটা বুঝতে পারছি না, একটু সহজ উদাহরণ দিয়ে বুঝিয়ে দাও না।",
            "{topic} কী এবং এটা কীভাবে কাজ করে, সহজ ভাষায় বলো।"
        ],
        ("GUIDED_PROBLEM_SOLVING", "STUDENT"): [
            "{topic}-এর সমস্যাগুলো কীভাবে ধাপে ধাপে সমাধান করব, একটা উদাহরণ দিয়ে দেখিয়ে দাও।",
            "আমাদের বইয়ের {topic}-এর এই নিয়মটা ব্যবহার করে কীভাবে উত্তর বের করব?"
        ],
        ("MISCONCEPTION_CORRECTION", "STUDENT"): [
            "আমার মনে হয় {topic}-এ সবসময় এই নিয়মটাই খাটে। আমার ধারণা কি ঠিক?",
            "{topic} নিয়ে আমার একটা খটকা আছে, এটা কি সবসময় সত্যি?"
        ]
    }

    @classmethod
    def synthesize_intent(
        cls,
        subject: str,
        grade: Any,
        topic: str,
        task_type: Optional[str] = None,
        requester_role: Optional[str] = None,
        instructional_target: Optional[str] = None
    ) -> UserIntent:
        """
        Synthesizes a realistic educational request enforcing Teacher-centric (85%) / Student (15%) distribution.
        """
        # 1. Determine requester_role (Default: 85% TEACHER, 15% STUDENT)
        if not requester_role:
            requester_role = "TEACHER" if random.random() < 0.85 else "STUDENT"

        # 2. Determine task_type and instructional_target based on requester_role
        if requester_role == "TEACHER":
            if not instructional_target:
                # 75% Student target, 25% Teacher target
                instructional_target = "STUDENT" if random.random() < 0.75 else "TEACHER"

            if not task_type:
                if instructional_target == "STUDENT":
                    task_type = random.choice([
                        "CONCEPT_EXPLANATION", "GUIDED_PROBLEM_SOLVING",
                        "MISCONCEPTION_CORRECTION", "QUIZ_GENERATION", "WORKSHEET_PRACTICE"
                    ])
                else:
                    task_type = random.choice(["LESSON_PLAN", "TEACHER_PEDAGOGY"])
            
            key = (task_type, instructional_target)
            templates = cls.TEACHER_TEMPLATES.get(key, cls.TEACHER_TEMPLATES.get(("CONCEPT_EXPLANATION", "STUDENT")))
            audience_str = "শিক্ষক মহাশয় / শিক্ষাবিদ"
        else:
            instructional_target = "STUDENT"
            if not task_type:
                task_type = random.choice(["CONCEPT_EXPLANATION", "GUIDED_PROBLEM_SOLVING", "MISCONCEPTION_CORRECTION"])
            key = (task_type, "STUDENT")
            templates = cls.STUDENT_TEMPLATES.get(key, cls.STUDENT_TEMPLATES.get(("CONCEPT_EXPLANATION", "STUDENT")))
            audience_str = "শিক্ষার্থী"

        template = random.choice(templates)
        clean_topic = topic.replace("অধ্যায়", "").replace("Lesson", "").replace("অধ্যায়", "").strip()
        if ":" in clean_topic:
            clean_topic = clean_topic.split(":", 1)[1].strip()

        grade_label = f"{grade}ম" if str(grade) in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"] else f"{grade}"
        query = template.format(
            topic=clean_topic or topic,
            subject=subject,
            grade=grade_label
        )

        return UserIntent(
            query=query,
            target_subject=subject,
            target_grade=grade,
            target_topic=topic,
            task_type=task_type,
            requester_role=requester_role,
            instructional_target=instructional_target,
            audience=audience_str
        )
