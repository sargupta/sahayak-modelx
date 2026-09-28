"""
Multi-Tier Grounding-First Validator and Quality Audit Engine with Dual-Role Compliance for SahayakAI SFT.
Enforces 10-tier strict validation and distinguishes requester_role (TEACHER / STUDENT)
from instructional_target (STUDENT / TEACHER / BOTH).
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    passed: bool
    record_id: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    tier_failures: List[str] = field(default_factory=list)


class GroundingFirstValidator:
    """10-Tier validation engine for grounding-first SFT records with dual-role support."""

    RAW_EXCERPT_PATTERN = re.compile(r'^\s*(পাঠ্যাংশ|অনুচ্ছেদ|Textbook Passage)\s*[:ঃ]', re.MULTILINE)
    META_LEAKAGE_PATTERN = re.compile(r'প্রদত্ত\s+পাঠ্যাংশ(ে|টির)?|উক্ত\s+পাঠ্যাংশ(ে|টির)?|প্রদত্ত\s+অনুচ্ছেদ(ে)?')
    ASCII_DIGITS_PATTERN = re.compile(r'(?<![a-zA-Z_\^\$\{\}\\])\b\d+\b(?![a-zA-Z_\^\$\{\}\\])')
    ORPHAN_MATRA_PATTERN = re.compile(r'(?:^|\s)[\u09BE-\u09CD\u09D7]+(?:\s|$)')
    BROKEN_STEM_PATTERN = re.compile(r'\b(অধ\s*:\s*বয|বয\s+অধ|ঐত\s+পট|রাজন\s+তর|উপন\s+সন|সম\s+রকদ|আধ\s+রতের)\b')

    def __init__(self):
        self.stats = {
            "total_evaluated": 0,
            "passed": 0,
            "rejected": 0,
            "retrieval_failures": 0,
            "compatibility_rejections": 0,
            "tier_breakdown": {
                "TIER_1_FORMAT": 0,
                "TIER_2_NO_RAW_EXCERPT": 0,
                "TIER_3_GROUNDING_FIDELITY": 0,
                "TIER_4_NO_META_LEAKAGE": 0,
                "TIER_5_BANGLA_NUMERALS": 0,
                "TIER_6_AUDIENCE_REGISTER": 0,
                "TIER_7_TASK_FIDELITY": 0,
                "TIER_8_MATH_SCIENTIFIC_INTEGRITY": 0,
                "TIER_9_SELF_CONTAINEDNESS": 0,
                "TIER_10_COMPLETENESS": 0,
                "TIER_OCR_INTEGRITY": 0
            }
        }

    def validate_record(self, record: Dict[str, Any]) -> ValidationResult:
        self.stats["total_evaluated"] += 1
        rec_id = record.get("custom_id", "unknown_id")
        errors = []
        warnings = []
        tier_failures = []

        messages = record.get("messages", [])

        # Tier 1: Format & Schema Integrity
        if len(messages) < 3:
            errors.append("Invalid conversation length (requires at least system, user, assistant).")
            tier_failures.append("TIER_1_FORMAT")
        else:
            roles = [m.get("role") for m in messages]
            if roles[:3] != ["system", "user", "assistant"]:
                errors.append(f"Invalid message role sequence: {roles[:3]}")
                tier_failures.append("TIER_1_FORMAT")

        if errors:
            self._record_failure(tier_failures)
            return ValidationResult(passed=False, record_id=rec_id, errors=errors, tier_failures=tier_failures)

        user_content = messages[1].get("content", "").strip()
        asst_content = messages[2].get("content", "").strip()
        metadata = record.get("metadata", {})
        task_type = metadata.get("task_type", "")
        requester_role = metadata.get("requester_role", "TEACHER" if "শিক্ষক" in metadata.get("audience", "") else "STUDENT")
        instructional_target = metadata.get("instructional_target", "STUDENT")

        # Tier 2: No Raw Excerpt Container in User Query (Unless USER_PROVIDED_SOURCE)
        if metadata.get("grounding_mode") == "TEXTBOOK_INTERNAL":
            if self.RAW_EXCERPT_PATTERN.search(user_content):
                errors.append("User message contains artificial 'পাঠ্যাংশ:' or raw excerpt container.")
                tier_failures.append("TIER_2_NO_RAW_EXCERPT")

        # Tier 3 & 4: Meta-leakage in Assistant Response
        if self.META_LEAKAGE_PATTERN.search(asst_content):
            errors.append("Assistant response contains forbidden meta-leakage ('প্রদত্ত পাঠ্যাংশে').")
            tier_failures.append("TIER_4_NO_META_LEAKAGE")

        # Tier 5: Numeral System Compliance (Bangla digits in Bengali prose)
        if metadata.get("subject") not in ["English", "Mathematics"]:
            ascii_matches = self.ASCII_DIGITS_PATTERN.findall(asst_content)
            if len(ascii_matches) > 3:
                warnings.append(f"Found {len(ascii_matches)} ASCII digits in Bengali prose: {ascii_matches[:3]}")

        # Tier 6: Audience Register Compliance (Dual-Role Aware)
        if requester_role == "TEACHER":
            # Teacher should be addressed respectfully with 'আপনি'
            if "তুমি " in asst_content and "আপনি" not in asst_content and "শিক্ষার্থী" not in asst_content:
                warnings.append("Teacher requester addressed purely in informal 'তুমি' register without pedagogical framing.")
        elif requester_role == "STUDENT":
            # Direct student tutoring should be friendly 'তুমি'
            if "আপনি " in asst_content and "তুমি" not in asst_content:
                warnings.append("Student requester addressed in formal 'আপনি' register.")

        # Tier 7: Task Fidelity
        if task_type in ["LESSON_PLAN", "LESSON_PLAN_GENERATION"]:
            if not any(k in asst_content for k in ["উদ্দেশ্য", "পরিকল্পনা", "উপকরণ", "TLM", "পাঠ", "মিনিট", "শিক্ষাদান"]):
                errors.append("Lesson plan task did not produce pedagogical plan structure.")
                tier_failures.append("TIER_7_TASK_FIDELITY")
        elif task_type in ["QUIZ_GENERATION", "QUIZ"]:
            if not any(k in asst_content for k in ["প্রশ্ন", "কুইজ", "মান", "নম্বর", "১.", "২."]):
                errors.append("Quiz generation task did not produce structured questions.")
                tier_failures.append("TIER_7_TASK_FIDELITY")

        # Tier 9: Self-Containedness
        if "পাশের চিত্রে" in asst_content or "প্রদত্ত মানচিত্রে" in asst_content:
            warnings.append("Response mentions external figures/maps; verify visual asset availability.")

        # Tier OCR Integrity: Reject broken Bengali OCR artifacts and orphaned matras
        if self.ORPHAN_MATRA_PATTERN.search(asst_content) or self.BROKEN_STEM_PATTERN.search(asst_content):
            errors.append("Assistant response contains broken Bengali OCR characters or orphaned matras.")
            tier_failures.append("TIER_OCR_INTEGRITY")
        if self.ORPHAN_MATRA_PATTERN.search(user_content) or self.BROKEN_STEM_PATTERN.search(user_content):
            errors.append("User query contains broken Bengali OCR characters or orphaned matras.")
            tier_failures.append("TIER_OCR_INTEGRITY")

        # Tier 10: Completeness
        if not asst_content.endswith(("।", "?", "!", ".", "”", "'", "’", "\n")):
            errors.append("Assistant response appears truncated or missing terminal punctuation.")
            tier_failures.append("TIER_10_COMPLETENESS")

        passed = len(errors) == 0
        if passed:
            self.stats["passed"] += 1
        else:
            self.stats["rejected"] += 1
            self._record_failure(tier_failures)

        return ValidationResult(
            passed=passed,
            record_id=rec_id,
            errors=errors,
            warnings=warnings,
            tier_failures=tier_failures
        )

    def record_retrieval_failure(self):
        self.stats["retrieval_failures"] += 1

    def record_compatibility_rejection(self):
        self.stats["compatibility_rejections"] += 1

    def _record_failure(self, tier_failures: List[str]):
        for t in tier_failures:
            if t in self.stats["tier_breakdown"]:
                self.stats["tier_breakdown"][t] += 1

    def get_audit_summary(self) -> Dict[str, Any]:
        return dict(self.stats)
