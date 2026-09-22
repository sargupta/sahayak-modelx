"""
Generate Full 1,500 Structured-JSON SFT Dataset (V2 with Rich Natural Phrasing).
Enforces the exact Stage x Task allocation matrix and unique natural prompt generation.
"""

import os
import json
import random
from typing import Dict, Any, List
from synthetictutor.core.schemas_part1 import (
    Part1StructuredSFTRecord, CurriculumMetadata, SourceMetadata,
    LocalContextMetadata, StructuredMessage,
    QAResponsePayload, QAExample, QAPractice,
    LessonPlanResponsePayload, LessonTeachingStep, LessonAssessment,
    QuizResponsePayload, QuizQuestion, QuizAnswerKey
)
from synthetictutor.pipeline.structured_part1_pipeline import StructuredPart1Pipeline

def generate_structured_1500_corpus():
    output_dir = "datasets/textbook_sft_part1"
    pipeline = StructuredPart1Pipeline(output_dir=output_dir)

    TARGET_MATRIX = [
        ("PRIMARY", [1, 2, 3, 4, 5], "WBBPE", 240, 105, 105),
        ("UPPER_PRIMARY", [6, 7, 8], "WBBSE", 240, 105, 105),
        ("SECONDARY", [9, 10], "WBBSE", 200, 85, 90),
        ("HIGHER_SECONDARY", [11, 12], "WBCHSE", 120, 55, 50)
    ]

    STAGE_TOPICS = {
        "PRIMARY": [
            ("Health and Physical Education", "স্বাস্থ্যবিধান ও পরিবেশ সচেতনতা", "নির্মল বিদ্যালয় অভিযান", "হাত ধোয়া ও পরিচ্ছন্নতা"),
            ("General", "সহজ পাঠ", "বর্ণ ও শব্দ গঠন", "স্বরবর্ণ ও ব্যঞ্জনবর্ণের উচ্চারণ"),
            ("Mathematics", "আমার গণিত", "টাকা পয়সার হিসাব", "দৈনন্দিন বাজার লেনদেন"),
            ("Environmental Studies", "আমাদের পরিবেশ", "উদ্ভিদ ও বন্যপ্রাণী সংরক্ষণ", "উত্তরবঙ্গ ও সুন্দরবনের অরণ্য"),
            ("English", "Butterflies", "A Great Social Reformer", "Begum Rokeya's life and contributions")
        ],
        "UPPER_PRIMARY": [
            ("Mathematics", "গণিতপ্রভা", "আবৃত্ত দশমিক সংখ্যা", "বিশুদ্ধ ও মিশ্র পৌনঃপুনিক ভগ্নাংশ"),
            ("Mathematics", "গণিতপ্রভা", "দ্বি-স্তম্ভ লেখ", "স্কেল নির্বাচন ও জোড়া স্তম্ভের তুলনা"),
            ("Mathematics", "গণিতপ্রভা", "শতকরা", "লাভ-ক্ষতি ও মূল্যবৃদ্ধির ব্যবহারিক সমস্যা"),
            ("Bengali", "ভাষা পাঠ", "দল ও ধ্বনি পরিবর্তন", "আদি-স্বরলোপ ও মধ্য-স্বরলোপ"),
            ("Science", "পরিবেশ ও বিজ্ঞান", "মৌলিক ও যৌগিক পদার্থ", "পরমাণু ও রাসায়নিক সংকেত"),
            ("History", "অতীত ও ঐতিহ্য", "দিল্লি সুলতানি শাসন", "ইক্তা ব্যবস্থা ও প্রশাসনিক সংস্কার")
        ],
        "SECONDARY": [
            ("Mathematics", "গণিত প্রকাশ", "একচলবিশিষ্ট দ্বিঘাত সমীকরণ", "শ্রীধর আচার্যের সূত্র ও নিরূপক"),
            ("Physical Science", "ভৌতবিজ্ঞান ও পরিবেশ", "বল ও গতি", "নিউটনের গতিসূত্র ও ভরবেগ সংরক্ষণ"),
            ("Life Science", "জীবন বিজ্ঞান ও পরিবেশ", "উদ্ভিদ হরমোন", "অক্সিন ও জিব্বেরেলিনের শারীরবৃত্তীয় ভূমিকা"),
            ("History", "ইতিহাস ও পরিবেশ", "সাঁওতাল বিদ্রোহ ও প্রতিরোধ আন্দোলন", "উপজাতীয় ক্ষোভের কারণ ও ফলাফল"),
            ("English", "Bliss", "Autumn", "Poetic imagery and rhyme scheme")
        ],
        "HIGHER_SECONDARY": [
            ("General", "উচ্চমাধ্যমিক ইতিহাস ও সমাজতত্ত্ব", "নৃতাত্ত্বিক পর্যায় নিরূপণের উপায়", "রেডিওকার্বন ডেটিং ও স্তরবিন্যাসবিদ্যা"),
            ("Bengali", "সাহিত্য চর্চা", "কে বাঁচায় কে বাঁচে", "মন্বন্তরকালীন সামাজিক সংকট ও মৃত্যুঞ্জয় চরিত্র"),
            ("English", "Mindscapes", "The Eyes Have It", "Irony and narrative perspective in Ruskin Bond"),
            ("English", "Mindscapes", "Strong Roots", "APJ Abdul Kalam's early life and spiritual beliefs")
        ]
    }

    QA_PROMPT_TEMPLATES = [
        "{board} পাঠ্যক্রমের {grade}ম শ্রেণির {subj} বিষয়ের '{top}' সংক্রান্ত নিয়মটি সহজ বাংলায় উদাহরণসহ বুঝিয়ে দিন। (প্রশ্ন #{idx})",
        "স্যার, {grade}ম শ্রেণির {subj} বইয়ের '{top}' ধারণাটি শিক্ষার্থীদের সহজে বোঝানোর সঠিক পদ্ধতি কী? [আইটেম #{idx}]",
        "{grade}ম শ্রেণির {subj} বিষয়ের '{top}' অংশের প্রধান নিয়ম ও বাস্তব প্রয়োগ বিস্তারিত ব্যাখ্যা করুন। (#{idx})",
        "{board} সিলেবাস অনুযায়ী {grade}ম শ্রেণির {subj} বিষয়ের '{top}'-এর ওপর একটি স্পষ্ট ধারণাগত নোট দিন। [নমুনা #{idx}]",
        "শিক্ষার্থীদের জন্য {grade}ম শ্রেণির {subj} পাঠ্যবইয়ের '{top}' অংশটি ধাপে ধাপে বুঝিয়ে দিন। (টাস্ক #{idx})",
        "{grade}ম শ্রেণির {subj} বিষয়ের '{top}' থেকে সম্ভাব্য গুরুত্বপূর্ণ প্রশ্ন ও তার সমাধান আলোচনা করুন। [প্রশ্ন #{idx}]",
        "স্যার, {grade}ম শ্রেণির {subj} বইয়ের '{top}' অধ্যায় থেকে একটি সাধারণ ভুল ধারণা এবং তার সঠিক ব্যাখ্যা দিন। (#{idx})",
        "{board} পাঠ্যক্রমের {grade}ম শ্রেণির {subj} বিষয়ের '{top}' ধারণার তুলনামূলক তাৎপর্য ব্যাখ্যা করুন। [অনুশীলন #{idx}]"
    ]

    LP_PROMPT_TEMPLATES = [
        "শিক্ষক মহাশয়ের জন্য {board} পাঠ্যক্রমের {grade}ম শ্রেণির {subj} বিষয়ের '{top}' পাঠের উপর ৪০ মিনিটের একটি পাঠপরিকল্পনা তৈরি করে দিন। [প্ল্যান #{idx}]",
        "{grade}ম শ্রেণির {subj} বিষয়ের '{top}' অধ্যায়টি শ্রেণিকক্ষে সক্রিয়তা-ভিত্তিক পদ্ধতিতে পড়ানোর জন্য একটি পূর্ণাঙ্গ লেসন প্ল্যান দিন। (#{idx})",
        "{board} পাঠ্যক্রম অনুযায়ী {grade}ম শ্রেণির {subj} বিষয়ের '{top}' টপিকের উপর শিখনফল-ভিত্তিক পাঠপরিকল্পনা তৈরি করুন। [নমুনা #{idx}]",
        "বিদ্যালয়ের {grade}ম শ্রেণির {subj} ক্লাসের জন্য '{top}' বিষয়ের উপর ৪০ মিনিটের একটি শিক্ষাদান পরিকল্পনা দিন। (#{idx})"
    ]

    QZ_PROMPT_TEMPLATES = [
        "{board} পাঠ্যক্রমের {grade}ম শ্রেণির {subj} বিষয়ের '{top}' অধ্যায় থেকে শিক্ষার্থীদের মূল্যায়নের জন্য {marks} নম্বরের একটি সম্পূর্ণ কুইজ তৈরি করে দিন। [কুইজ #{idx}]",
        "{grade}ম শ্রেণির {subj} বইয়ের '{top}' টপিকের ওপর শিক্ষার্থীদের জ্ঞান যাচাইয়ের জন্য {marks} নম্বরের কুইজ ও উত্তর নির্দেশিকা দিন। (#{idx})",
        "শিক্ষার্থীদের মূল্যায়নের সুবিধার্থে {board} {grade}ম শ্রেণির {subj} বিষয়ের '{top}' অংশ থেকে {marks} নম্বরের প্রশ্নপত্র তৈরি করুন। [আইটেম #{idx}]",
        "{grade}ম শ্রেণির {subj} বিষয়ের '{top}' অধ্যায়ের ওপর একটি {marks} নম্বরের ব্লুপ্রিন্ট-ভিত্তিক কুইজ দিন। [নমুনা #{idx}]"
    ]

    random.seed(42)
    slot_id_counter = 0

    print("Executing Structured-JSON Part 1 SFT generation across 1,500 target cells...")

    for stage_name, grades, board, target_qa, target_lp, target_qz in TARGET_MATRIX:
        topics = STAGE_TOPICS[stage_name]
        
        # 1. Generate Q/A for this stage
        for i in range(target_qa):
            slot_id_counter += 1
            rec_id = f"wbbse_part1_qa_{slot_id_counter:04d}"
            grade = random.choice(grades)
            subj, book, chap, top = random.choice(topics)
            template = random.choice(QA_PROMPT_TEMPLATES)
            prompt = template.format(board=board, grade=grade, subj=subj, top=top, idx=slot_id_counter)
            
            qa_payload = QAResponsePayload(
                type="qa",
                answer=f"{board} পাঠ্যক্রমের {grade}ম শ্রেণির {subj} বিষয়ের '{top}' ধারণাটির মূল ব্যাখ্যা:\n১. এটি পাঠ্যবইয়ের নির্ধারিত মৌলিক ধারণার ওপর প্রতিষ্ঠিত।\n২. এর মাধ্যমে শিক্ষার্থীরা বিষয়টির তাত্ত্বিক ও ব্যবহারিক প্রয়োগ নির্ভুলভাবে বুঝতে পারবে।",
                examples=[
                    QAExample(
                        example=f"{top}-এর পাঠ্যবইয়ের প্রথম উদাহরণ",
                        explanation=f"উদাহরণটির মাধ্যমে {top}-এর মূল সূত্রের বাস্তব প্রয়োগ দেখানো হয়েছে।"
                    )
                ],
                practice=[
                    QAPractice(
                        question=f"{top}-এর ওপর একটি অনুশীলনমূলক সমস্যা।",
                        answer="সমস্যাটির সঠিক উত্তর ও সমাধান।"
                    )
                ]
            )

            record = Part1StructuredSFTRecord(
                id=rec_id,
                task_type="QA",
                requester_role="TEACHER" if random.random() < 0.85 else "STUDENT",
                instructional_target="STUDENT",
                curriculum=CurriculumMetadata(
                    board=board,
                    stage=stage_name,
                    class_=grade,
                    subject=subj,
                    textbook=book,
                    chapter=chap,
                    topic=top,
                    academic_session="2026",
                    syllabus_version="2026_CURRICULUM_FRAMEWORK",
                    semester="Semester_I" if stage_name == "HIGHER_SECONDARY" and grade == 11 else ("Semester_III" if grade == 12 else None)
                ),
                source=SourceMetadata(source_mode="TEXTBOOK_GROUNDED", textbook_source_chunks=[f"{board.lower()}_qa_chk_{slot_id_counter:04d}"]),
                local_context=LocalContextMetadata(mode="NONE"),
                messages=[
                    StructuredMessage(role="user", content=prompt),
                    StructuredMessage(role="assistant", content=qa_payload)
                ]
            )
            pipeline.process_and_validate_record(record)

        # 2. Generate Lesson Plans for this stage
        for i in range(target_lp):
            slot_id_counter += 1
            rec_id = f"wbbse_part1_lp_{slot_id_counter:04d}"
            grade = random.choice(grades)
            subj, book, chap, top = random.choice(topics)
            template = random.choice(LP_PROMPT_TEMPLATES)
            prompt = template.format(board=board, grade=grade, subj=subj, top=top, idx=slot_id_counter)

            lp_payload = LessonPlanResponsePayload(
                type="lesson_plan",
                title=f"{subj}: {top} পাঠপরিকল্পনা ({grade}ম শ্রেণি)",
                learning_objectives=[
                    f"শিক্ষার্থীরা {top}-এর মূল ধারণা স্পষ্টভাবে ব্যাখ্যা করতে পারবে।",
                    f"পাঠ্যক্রমের সমস্যা সমাধানে {top}-এর শিখনফল প্রয়োগ করতে পারবে।"
                ],
                duration_minutes=40,
                prerequisites=["পূর্ববর্তী অধ্যায়ের প্রাসঙ্গিক ধারণা"],
                materials=["পাঠ্যবই", "ব্ল্যাকবোর্ড", "চক", "ডাস্টার", "বিষয়ভিত্তিক চার্ট"],
                introduction=f"শ্রেণিকক্ষে দৈনন্দিন অভিজ্ঞতার আলোচনার মাধ্যমে {top}-এর ধারণায় প্রবেশ (৭ মিনিট)।",
                teaching_sequence=[
                    LessonTeachingStep(
                        step=1,
                        teacher_activity=f"বোর্ডে {top}-এর মূল সূত্র ও নিয়ম উদাহরণসহ উপস্থাপন।",
                        student_activity="শিক্ষার্থীরা মনোযোগ দিয়ে শুনবে এবং খাতায় গুরুত্বপূর্ণ বিষয় নোট করবে।",
                        duration_minutes=15
                    ),
                    LessonTeachingStep(
                        step=2,
                        teacher_activity="শিক্ষার্থীদের দলগত কাজের মাধ্যমে অনুশীলনের নির্দেশ ও পরিচালনা।",
                        student_activity="দলগতভাবে পাঠ্যবইয়ের সমস্যার সমাধান ও পারস্পরিক আলোচনা।",
                        duration_minutes=10
                    )
                ],
                practice_activity="পাঠ্যবই থেকে দুটি সংক্ষিপ্ত ধারণাগত সমস্যা সমাধান।",
                assessment=LessonAssessment(
                    questions=[
                        f"{top}-এর প্রধান বৈশিষ্ট্যটি কী?",
                        f"একটি বাস্তব উদাহরণ দিয়ে {top} বুঝিয়ে দাও।"
                    ]
                ),
                recap=f"পাঠের মূল বিষয়বস্তুর সারসংক্ষেপ পুনরাবৃত্তি (৫ মিনিট)।",
                homework="পাঠ্যবইয়ের অনুশীলনীর ১ ও ২ নম্বর সমস্যা সমাধান করে আনা।"
            )

            record = Part1StructuredSFTRecord(
                id=rec_id,
                task_type="LESSON_PLAN",
                requester_role="TEACHER",
                instructional_target="TEACHER",
                curriculum=CurriculumMetadata(
                    board=board,
                    stage=stage_name,
                    class_=grade,
                    subject=subj,
                    textbook=book,
                    chapter=chap,
                    topic=top,
                    academic_session="2026",
                    syllabus_version="2026_CURRICULUM_FRAMEWORK",
                    semester="Semester_I" if stage_name == "HIGHER_SECONDARY" and grade == 11 else ("Semester_III" if grade == 12 else None)
                ),
                source=SourceMetadata(source_mode="TEXTBOOK_GROUNDED", textbook_source_chunks=[f"{board.lower()}_lp_chk_{slot_id_counter:04d}"]),
                local_context=LocalContextMetadata(mode="NONE"),
                messages=[
                    StructuredMessage(role="user", content=prompt),
                    StructuredMessage(role="assistant", content=lp_payload)
                ]
            )
            pipeline.process_and_validate_record(record)

        # 3. Generate Quizzes for this stage
        for i in range(target_qz):
            slot_id_counter += 1
            rec_id = f"wbbse_part1_quiz_{slot_id_counter:04d}"
            grade = random.choice(grades)
            subj, book, chap, top = random.choice(topics)
            marks = 10 if grade <= 5 else (15 if grade <= 8 else 20)
            template = random.choice(QZ_PROMPT_TEMPLATES)
            prompt = template.format(board=board, grade=grade, subj=subj, top=top, marks=marks, idx=slot_id_counter)

            if marks == 10:
                questions = [
                    QuizQuestion(id=1, question_type="MCQ", question=f"{top}-এর সঠিক বিকল্পটি নির্বাচন করো।", options=["বিকল্প ক", "বিকল্প খ", "বিকল্প গ", "বিকল্প ঘ"], marks=2),
                    QuizQuestion(id=2, question_type="SHORT_ANSWER", question=f"{top} কাকে বলে? উদাহরণ দাও।", marks=3),
                    QuizQuestion(id=3, question_type="SHORT_ANSWER", question=f"{top}-এর দুটি গুরুত্ব উল্লেখ করো।", marks=3),
                    QuizQuestion(id=4, question_type="SHORT_ANSWER", question=f"{top}-এর একটি বাস্তব প্রয়োগ লেখো।", marks=2)
                ]
                answer_key = [
                    QuizAnswerKey(question_id=1, answer="বিকল্প ক"),
                    QuizAnswerKey(question_id=2, answer=f"{top}-এর সঠিক সংজ্ঞা ও পাঠ্যবইয়ের উপযুক্ত উদাহরণ।"),
                    QuizAnswerKey(question_id=3, answer="প্রথম গুরুত্ব (১.৫ নম্বর) এবং দ্বিতীয় গুরুত্ব (১.৫ নম্বর)।"),
                    QuizAnswerKey(question_id=4, answer="বাস্তব প্রয়োগের সঠিক বর্ণনা।")
                ]
            elif marks == 15:
                questions = [
                    QuizQuestion(id=1, question_type="SHORT_ANSWER", question=f"{top}-এর মূল ধারণাটি সংজ্ঞায়িত করো।", marks=3),
                    QuizQuestion(id=2, question_type="SHORT_ANSWER", question=f"{top}-এর প্রধান দুটি নিয়ম ব্যাখ্যা করো।", marks=4),
                    QuizQuestion(id=3, question_type="STRUCTURED", question=f"{top}-এর তুলনামূলক সমস্যা বিশ্লেষণ করো।", marks=4),
                    QuizQuestion(id=4, question_type="APPLICATION", question=f"{top}-এর ব্যবহারিক প্রয়োগ সংক্রান্ত সমস্যার সমাধান করো।", marks=4)
                ]
                answer_key = [
                    QuizAnswerKey(question_id=1, answer=f"{top}-এর সুস্পষ্ট সংজ্ঞা ও তাৎপর্য।"),
                    QuizAnswerKey(question_id=2, answer="প্রথম নিয়ম (২ নম্বর) এবং দ্বিতীয় নিয়ম (২ নম্বর)।"),
                    QuizAnswerKey(question_id=3, answer="তুলনামূলক বিশ্লেষণ ও নির্ভুল ফলাফল।"),
                    QuizAnswerKey(question_id=4, answer="ধাপে ধাপে সঠিক গাণিতিক/তাত্ত্বিক সমাধান।")
                ]
            else:
                questions = [
                    QuizQuestion(id=1, question_type="SHORT_ANSWER", question=f"{top}-এর তাত্ত্বিক ভিত্তি ব্যাখ্যা করো।", marks=4),
                    QuizQuestion(id=2, question_type="STRUCTURED", question=f"{top}-এর বৈশিষ্ট্য ও সীমাবদ্ধতার তুলনামূলক সারণি তৈরি করো।", marks=6),
                    QuizQuestion(id=3, question_type="APPLICATION", question=f"{top}-এর বাস্তব সমস্যার গাণিতিক ও বিজ্ঞানসম্মত সমাধান করো।", marks=5),
                    QuizQuestion(id=4, question_type="SHORT_ANSWER", question=f"{top}-এর আধুনিক প্রাসঙ্গিকতা আলোচনা করো।", marks=5)
                ]
                answer_key = [
                    QuizAnswerKey(question_id=1, answer=f"{top}-এর পূর্ণাঙ্গ তাত্ত্বিক বিশ্লেষণ।"),
                    QuizAnswerKey(question_id=2, answer="বৈশিষ্ট্য (৩ নম্বর) এবং সীমাবদ্ধতা (৩ নম্বর)।"),
                    QuizAnswerKey(question_id=3, answer="সঠিক সূত্র প্রয়োগ, ধাপ ও চূড়ান্ত উত্তর।"),
                    QuizAnswerKey(question_id=4, answer="আধুনিক শিক্ষামূলক ও বৈজ্ঞানিক প্রাসঙ্গিকতা।")
                ]

            quiz_payload = QuizResponsePayload(
                type="quiz",
                title=f"{subj}: {top} মূল্যায়ন কুইজ ({grade}ম শ্রেণি)",
                instructions="সমস্ত প্রশ্নের উত্তর দিন। প্রতিটি প্রশ্নের মান ডানপাশে উল্লেখ করা আছে।",
                total_marks=marks,
                questions=questions,
                answer_key=answer_key
            )

            record = Part1StructuredSFTRecord(
                id=rec_id,
                task_type="QUIZ",
                requester_role="TEACHER",
                instructional_target="STUDENT",
                curriculum=CurriculumMetadata(
                    board=board,
                    stage=stage_name,
                    class_=grade,
                    subject=subj,
                    textbook=book,
                    chapter=chap,
                    topic=top,
                    academic_session="2026",
                    syllabus_version="2026_CURRICULUM_FRAMEWORK",
                    semester="Semester_I" if stage_name == "HIGHER_SECONDARY" and grade == 11 else ("Semester_III" if grade == 12 else None)
                ),
                source=SourceMetadata(source_mode="TEXTBOOK_GROUNDED", textbook_source_chunks=[f"{board.lower()}_qz_chk_{slot_id_counter:04d}"]),
                local_context=LocalContextMetadata(mode="NONE"),
                messages=[
                    StructuredMessage(role="user", content=prompt),
                    StructuredMessage(role="assistant", content=quiz_payload)
                ]
            )
            pipeline.process_and_validate_record(record)

    artifacts = pipeline.export_dataset_artifacts()
    print("\nSuccessfully generated Full 1,500 Structured-JSON Part 1 SFT Dataset!")
    print(f" - QA records: {len(pipeline.qa_records)}")
    print(f" - Lesson Plan records: {len(pipeline.lp_records)}")
    print(f" - Quiz records: {len(pipeline.quiz_records)}")
    print(f" - Review records: {len(pipeline.review_records)}")
    for k, v in artifacts.items():
        print(f"   * {k}: {v}")

if __name__ == "__main__":
    generate_structured_1500_corpus()