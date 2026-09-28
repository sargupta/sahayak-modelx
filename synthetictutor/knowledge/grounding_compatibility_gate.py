"""
Grounding Compatibility Gate with 0-5 Curriculum Fidelity Scoring and Verified Locale Validation.
Enforces strict alignment across User Request <-> Curriculum <-> Textbook <-> Locale.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from synthetictutor.knowledge.textbook_retriever import BengaliTokenizer
from synthetictutor.knowledge.locale_retriever import LocaleSnippet


@dataclass
class CompatibilityResult:
    is_compatible: bool
    status: str  # 'PASS', 'REJECT', 'FALLBACK_GENERAL_KNOWLEDGE'
    curriculum_fidelity_score: int  # 0 to 5 scale
    reasons: List[str] = field(default_factory=list)
    suggested_action: str = "ACCEPT"  # 'ACCEPT', 'RETRY_RETRIEVAL', 'FALLBACK', 'REJECT_TASK'
    board_authority: str = "WBBSE"


class GroundingCompatibilityGate:
    """Strict verification gate ensuring retrieved textbook chunks genuinely match user requests."""

    ADMIN_PATTERNS = [
        re.compile(r'পশ্চিমবঙ্গ\s+মধ্যশিক্ষা\s+পর্ষদ\s+কর্তৃক\s+প্রকাশিত', re.IGNORECASE),
        re.compile(r'সর্বস্বত্ব\s+সংরক্ষিত|copyright|all\s+rights\s+reserved', re.IGNORECASE),
        re.compile(r'isbn\s*[:\-\d]+', re.IGNORECASE),
        re.compile(r'মূল্য\s*[:ঃ]\s*টাকা|price\s*[:ঃ]', re.IGNORECASE),
        re.compile(r'মুদ্রক\s*[:ঃ]|মুদ্রণ\s*[:ঃ]|প্রকাশক\s*[:ঃ]', re.IGNORECASE),
        re.compile(r'পাঠ্যপুস্তক\s+রচনা\s+সমিতি|উপদেষ্টা\s+মণ্ডলী', re.IGNORECASE),
        re.compile(r'নিবেদনে\s*[:ঃ]|বিনীত\s*[:ঃ]', re.IGNORECASE)
    ]

    VISUAL_PATTERNS = [
        re.compile(r'পাশের\s+ছবি(তে|টি|গুলি)|পাশের\s+চিত্র(টি|টিতে)', re.IGNORECASE),
        re.compile(r'উপরের\s+ছবি(তে|টি)|উপরের\s+চিত্র(টি|টিতে)', re.IGNORECASE),
        re.compile(r'নিচের\s+মানচিত্র(টি|টিতে)|প্রদত্ত\s+মানচিত্র', re.IGNORECASE),
        re.compile(r'see\s+the\s+picture\s+above|look\s+at\s+the\s+given\s+diagram', re.IGNORECASE)
    ]

    OCR_CORRUPTION_PATTERNS = [
        re.compile(r'(?:^|\s)[\u09BE-\u09CD\u09D7]+(?:\s|$)'),
        re.compile(r'\b(অধ\s*:\s*বয|বয\s+অধ|ঐত\s+পট|রাজন\s+তর|উপন\s+সন|সম\s+রকদ|আধ\s+রতের)\b')
    ]

    def __init__(self, min_overlap_threshold: float = 0.2):
        self.min_overlap_threshold = min_overlap_threshold

    @classmethod
    def resolve_board_authority(cls, grade: Any) -> str:
        """Enforces correct educational authority based on grade level."""
        try:
            g = int(str(grade).replace("Class", "").replace("class", "").strip())
            if g <= 5:
                return "পশ্চিমবঙ্গ প্রাথমিক শিক্ষা পর্ষদ (WBBPE)"
            elif 6 <= g <= 10:
                return "পশ্চিমবঙ্গ মধ্যশিক্ষা পর্ষদ (WBBSE)"
            else:
                return "পশ্চিমবঙ্গ উচ্চমাধ্যমিক শিক্ষা সংসদ (WBCHSE)"
        except Exception:
            return "পশ্চিমবঙ্গ মধ্যশিক্ষা পর্ষদ (WBBSE)"

    def evaluate(
        self,
        user_query: str,
        retrieved_chunk: Dict[str, Any],
        task_type: Optional[str] = None,
        target_subject: Optional[str] = None,
        target_grade: Optional[Any] = None,
        locale_snippets: Optional[List[LocaleSnippet]] = None
    ) -> CompatibilityResult:
        """
        Evaluates compatibility across user query, curriculum, textbook chunk, and verified locale.
        Computes a 0-5 Curriculum Fidelity Score.
        """
        reasons = []
        source_text = retrieved_chunk.get("source_text", "")
        topic = retrieved_chunk.get("topic", "")
        chapter = retrieved_chunk.get("chapter", "")
        subject = retrieved_chunk.get("subject", "")
        grade = str(retrieved_chunk.get("grade", ""))

        target_board = self.resolve_board_authority(target_grade if target_grade is not None else grade)

        # 1. Administrative Debris & OCR Corruption Check (Score 0)
        admin_match_count = sum(1 for p in self.ADMIN_PATTERNS if p.search(source_text))
        ocr_corrupt_count = sum(1 for p in self.OCR_CORRUPTION_PATTERNS if p.search(source_text))
        if admin_match_count >= 2 or ocr_corrupt_count > 0 or len(source_text.strip()) < 35:
            return CompatibilityResult(
                is_compatible=False,
                status="REJECT",
                curriculum_fidelity_score=0,
                reasons=["Source chunk contains administrative debris, insufficient content, or broken OCR glyphs/orphaned matras."],
                suggested_action="RETRY_RETRIEVAL",
                board_authority=target_board
            )

        # 2. Subject Conflict Check (Score 0)
        if target_subject:
            t_sub = target_subject.lower().strip()
            c_sub = subject.lower().strip()
            if t_sub not in c_sub and c_sub not in t_sub:
                return CompatibilityResult(
                    is_compatible=False,
                    status="REJECT",
                    curriculum_fidelity_score=0,
                    reasons=[f"Subject mismatch: requested '{target_subject}', got '{subject}'."],
                    suggested_action="RETRY_RETRIEVAL",
                    board_authority=target_board
                )

        # 3. Grade Conflict Check (Score 1)
        if target_grade is not None:
            t_grade = str(target_grade).replace("Class", "").replace("class", "").strip()
            if t_grade and grade and t_grade != grade:
                return CompatibilityResult(
                    is_compatible=False,
                    status="REJECT",
                    curriculum_fidelity_score=1,
                    reasons=[f"Grade mismatch: requested Class {t_grade}, got Class {grade}."],
                    suggested_action="RETRY_RETRIEVAL",
                    board_authority=target_board
                )

        # 4. Lexical & Semantic Topic Alignment
        query_tokens = set(BengaliTokenizer.tokenize(user_query))
        chunk_tokens = set(BengaliTokenizer.tokenize(f"{topic} {chapter} {source_text}"))
        topic_tokens = set(BengaliTokenizer.tokenize(f"{topic} {chapter}"))

        if not query_tokens:
            return CompatibilityResult(
                is_compatible=False,
                status="REJECT",
                curriculum_fidelity_score=0,
                reasons=["User query contains no substantive educational tokens."],
                suggested_action="REJECT_TASK",
                board_authority=target_board
            )

        overlap = query_tokens.intersection(chunk_tokens)
        overlap_score = len(overlap) / len(query_tokens)
        topic_overlap = query_tokens.intersection(topic_tokens)

        # 5. Determine Curriculum Fidelity Score (0 to 5)
        if len(topic_overlap) == 0 and overlap_score < self.min_overlap_threshold:
            return CompatibilityResult(
                is_compatible=False,
                status="REJECT",
                curriculum_fidelity_score=1,
                reasons=[f"Topic/Concept mismatch: query tokens {list(query_tokens)[:4]} not found in chunk '{topic}'."],
                suggested_action="RETRY_RETRIEVAL",
                board_authority=target_board
            )
        elif len(topic_overlap) > 0 and overlap_score < 0.25:
            fidelity_score = 3
        elif len(topic_overlap) > 0 and overlap_score >= 0.25:
            fidelity_score = 4
        else:
            fidelity_score = 2

        if len(source_text.split()) > 60 and fidelity_score >= 4:
            fidelity_score = 5

        # 6. Visual Dependency Handling
        has_visual_dep = retrieved_chunk.get("visual_dependency", False)
        visual_text_hits = [p.pattern for p in self.VISUAL_PATTERNS if p.search(source_text)]
        if has_visual_dep or visual_text_hits:
            reasons.append("Chunk has visual dependencies (diagram/map referenced in text).")

        # 7. Verified Locale Grounding Evaluation
        if locale_snippets:
            reasons.append(f"Grounding verified with {len(locale_snippets)} verified local facts from locale.json.")

        return CompatibilityResult(
            is_compatible=True,
            status="PASS",
            curriculum_fidelity_score=fidelity_score,
            reasons=reasons,
            suggested_action="ACCEPT",
            board_authority=target_board
        )
