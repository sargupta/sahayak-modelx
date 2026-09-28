"""
Part 1 Dataset Schema Definitions (Structured JSON Edition).
Implements the 48-rule specification for Structured-JSON Part 1 SFT Dataset.
"""

from typing import Dict, Any, List, Optional, Literal, Union
from pydantic import BaseModel, Field

class CurriculumMetadata(BaseModel):
    board: Literal["WBBPE", "WBBSE", "WBCHSE"]
    stage: Literal["PRIMARY", "UPPER_PRIMARY", "SECONDARY", "HIGHER_SECONDARY"]
    class_: int = Field(alias="class")
    medium: str = "Bengali"
    subject: str
    textbook: str
    chapter: str
    topic: str
    subtopic: Optional[str] = None
    learning_outcomes: List[str] = Field(default_factory=list)
    academic_session: str = "2026"
    syllabus_version: str = "2026_CURRICULUM_FRAMEWORK"
    coverage_status: Literal["VERIFIED", "PARTIALLY_VERIFIED", "TEXTBOOK_ONLY", "AMBIGUOUS", "UNKNOWN"] = "VERIFIED"
    assessment_period: Optional[str] = None
    semester: Optional[str] = None

    class Config:
        populate_by_name = True

class SourceMetadata(BaseModel):
    source_mode: Literal["TEXTBOOK_GROUNDED", "TEXTBOOK_PLUS_LOCALE"] = "TEXTBOOK_GROUNDED"
    textbook_source_chunks: List[str] = Field(default_factory=list)
    visual_dependency: Literal["NONE", "SUPPORTIVE", "REQUIRED"] = "NONE"

class RecordProvenance(BaseModel):
    generator_model: str = Field(..., description="Model identifier used for generation (e.g., qwen2.5-72b-instruct)")
    provider: str = Field(..., description="API or inference provider (e.g., openrouter, groq, vllm, deepseek)")
    licence: str = Field(default="CC-BY-4.0", description="Data license or open license terms")
    chunk_ids: List[str] = Field(default_factory=list, description="IDs of textbook grounding chunks used")
    locale_keys: List[str] = Field(default_factory=list, description="Keys/facts extracted from locale.json")
    validator_version: str = Field(default="1.0.0", description="Automated validator pipeline version")
    teacher_id: Optional[str] = Field(default=None, description="Identifier of human educator if reviewed")
    quarantined: bool = Field(default=False, description="Flag if quarantined pending license resolution")
    quarantine_reason: Optional[str] = Field(default=None, description="Reason for quarantine if applicable")

class LocalContextMetadata(BaseModel):
    mode: Literal["NONE", "GENERIC_STATE", "REGION", "DISTRICT", "SCHOOL_SETTING"] = "NONE"
    region: Optional[str] = None
    district: Optional[str] = None
    school_setting: Optional[str] = None
    locale_source_chunks: List[str] = Field(default_factory=list)
    verified_local_facts: List[Dict[str, Any]] = Field(default_factory=list)

# --- STRUCTURED ASSISTANT RESPONSE PAYLOADS ---

class QAExample(BaseModel):
    example: str
    explanation: str

class QAPractice(BaseModel):
    question: str
    answer: str

class QAResponsePayload(BaseModel):
    type: Literal["qa"] = "qa"
    answer: str
    examples: Optional[List[QAExample]] = None
    practice: Optional[List[QAPractice]] = None

class LessonTeachingStep(BaseModel):
    step: int
    teacher_activity: str
    student_activity: str
    duration_minutes: int

class LessonAssessment(BaseModel):
    questions: List[str] = Field(default_factory=list)

class LessonPlanResponsePayload(BaseModel):
    type: Literal["lesson_plan"] = "lesson_plan"
    title: str
    learning_objectives: List[str] = Field(default_factory=list)
    duration_minutes: int = 40
    prerequisites: Optional[List[str]] = None
    materials: List[str] = Field(default_factory=list)
    introduction: str
    teaching_sequence: List[LessonTeachingStep] = Field(default_factory=list)
    practice_activity: Optional[str] = None
    assessment: Optional[LessonAssessment] = None
    recap: Optional[str] = None
    homework: Optional[str] = None

class QuizQuestion(BaseModel):
    id: int
    question_type: Literal["MCQ", "SHORT_ANSWER", "STRUCTURED", "TRUE_FALSE", "FILL_IN_BLANKS", "APPLICATION"]
    question: str
    options: Optional[List[str]] = None
    marks: int

class QuizAnswerKey(BaseModel):
    question_id: int
    answer: str

class QuizResponsePayload(BaseModel):
    type: Literal["quiz"] = "quiz"
    title: str
    instructions: str
    total_marks: int
    questions: List[QuizQuestion]
    answer_key: Optional[List[QuizAnswerKey]] = None

# Union of all supported structured response payloads
AssistantResponseContent = Union[QAResponsePayload, LessonPlanResponsePayload, QuizResponsePayload, Dict[str, Any]]

class StructuredMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, AssistantResponseContent]

class ValidationScores(BaseModel):
    schema_passed: bool = True
    curriculum_alignment: bool = True
    topic_alignment: bool = True
    source_groundedness: bool = True
    local_groundedness: bool = True
    factual_correctness: bool = True
    instruction_following: bool = True
    grade_appropriateness: bool = True
    duplicate_free: bool = True
    overall_passed: bool = True
    failure_reasons: List[str] = Field(default_factory=list)

class Part1StructuredSFTRecord(BaseModel):
    id: str
    task_type: Literal["QA", "LESSON_PLAN", "QUIZ"]
    requester_role: Literal["TEACHER", "STUDENT"] = "TEACHER"
    instructional_target: Literal["STUDENT", "TEACHER", "BOTH"] = "STUDENT"
    curriculum: CurriculumMetadata
    source: SourceMetadata
    local_context: LocalContextMetadata
    provenance: Optional[RecordProvenance] = None
    messages: List[StructuredMessage]
    validation: ValidationScores = Field(default_factory=ValidationScores)