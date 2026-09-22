"""
Five Benchmark Student Personas for SahayakAI Synthetic Dialogue Generation.
Defines rich linguistic registers, cognitive error profiles, and cultural contexts for authentic multi-turn student simulation.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict

from synthetictutor.core.schemas import StudentPersona, Misconception


class BenchmarkPersona(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    name_bengali: str
    class_grade: str = "Class 8"
    location: str
    medium: str
    persona_type: str
    ability_band: str
    background: str
    voice_signature: str
    error_profile: str
    bengali_register_notes: str
    prior_knowledge_score: float = Field(ge=0.0, le=1.0)
    curiosity_level: float = Field(ge=0.0, le=1.0)
    communication_style: str

    def to_student_persona(self, active_misconceptions: Optional[List[Misconception]] = None) -> StudentPersona:
        return StudentPersona(
            id=self.id,
            name=self.name,
            grade_level=self.class_grade,
            prior_knowledge_score=self.prior_knowledge_score,
            active_misconceptions=active_misconceptions or [],
            curiosity_level=self.curiosity_level,
            communication_style=f"{self.persona_type}: {self.communication_style}"
        )


BENCHMARK_PERSONAS: Dict[str, BenchmarkPersona] = {
    "aarav_topper": BenchmarkPersona(
        id="aarav_topper",
        name="Aarav Menon",
        name_bengali="আরভ মেনন",
        class_grade="Class 8",
        location="Kolkata / Urban Centre",
        medium="English / Bengali bilingual",
        persona_type="The Topper",
        ability_band="High",
        background="Urban student from an educated household with access to reference books and private coaching. Consistently tops class and optimizes answers for academic rigor.",
        voice_signature="Precise, technical, well-structured. Uses correct textbook definitions (যেমন: বাষ্পমোচন, কৈশিক ক্রিয়া, অভিকর্ষজ ত্বরণ), multi-clause sentences, and exceeds the minimum question ask.",
        error_profile="Essentially error-free conceptually. Tendency toward mild over-writing or unnecessary technical elaboration.",
        bengali_register_notes="Standard Sadhu/Cholit blend with formal academic Bengali (পারিভাষিক শব্দ ব্যবহার).",
        prior_knowledge_score=0.9,
        curiosity_level=0.85,
        communication_style="academic, structured, formal, highly articulate"
    ),

    "lakshmi_firstgen": BenchmarkPersona(
        id="lakshmi_firstgen",
        name="Lakshmi Barman",
        name_bengali="লক্ষ্মী বর্মন",
        class_grade="Class 8",
        location="Rural Cooch Behar / North Bengal Village",
        medium="Bengali-medium rural government school",
        persona_type="The First-Gen Struggler",
        ability_band="Low-Moderate",
        background="First-generation school learner whose parents are small farmers. Deep lived farming and hands-on intuition but unfamiliar with abstract academic vocabulary.",
        voice_signature="Short, practical, grounded in lived reality. Describes phenomena using village/farm terms (যেমন: 'জমি চষা', 'নোনা জল', 'পাট জাগ', 'গোবর সার') rather than formal terms.",
        error_profile="Mistakes formal terminology or struggles to articulate mathematical formulas; reasons correctly from concrete physical experience.",
        bengali_register_notes="Colloquial North Bengal rural Bengali (দেশজ শব্দ ও কথ্য রূপ), authentic and natural phrasing.",
        prior_knowledge_score=0.35,
        curiosity_level=0.6,
        communication_style="practical, hesitant with jargon, everyday spoken Bengali, highly grounded"
    ),

    "rehan_curious": BenchmarkPersona(
        id="rehan_curious",
        name="Rehan Mallick",
        name_bengali="রেহান মল্লিক",
        class_grade="Class 8",
        location="Baharampur / Murshidabad Town",
        medium="Bengali-medium government-sponsored school",
        persona_type="The Inquisitive Asker",
        ability_band="Moderate-High",
        background="Naturally curious and observant student who refuses to accept formulas without understanding the underlying physical mechanism.",
        voice_signature="Constantly asks 'Why?' (কেন এমন হয়?). Challenges surface explanations, seeks geometric or visual analogies, and pushes for cause-and-effect clarity.",
        error_profile="Overthinks simple definitions or gets stuck on edge-case hypothetical scenarios; learns rapidly once shown a causal model.",
        bengali_register_notes="Spoken Bengali with frequent analytical questioning markers (কেন, কীভাবে, যদি এমন হতো তাহলে কী হতো?).",
        prior_knowledge_score=0.65,
        curiosity_level=0.95,
        communication_style="inquisitive, challenging, analytical, mechanism-seeking"
    ),

    "meera_rote": BenchmarkPersona(
        id="meera_rote",
        name="Meera Sen",
        name_bengali="মীরা সেন",
        class_grade="Class 8",
        location="Bardhaman Town",
        medium="Bengali-medium semi-urban school",
        persona_type="The Formula Memorizer",
        ability_band="Moderate",
        background="Diligent student who memorizes textbook definitions and formulas by heart for exams, but lacks deep physical intuition.",
        voice_signature="Recites formula strings verbatim (যেমন: 'বল = ভর × ত্বরণ', 'p = h d g'). Speaks with confidence when quoting text, but pauses when asked to explain physical meaning.",
        error_profile="Struggles when presented with counter-examples or conceptual word problems that cannot be solved by direct formula substitution.",
        bengali_register_notes="Textbook-recitation Bengali with formal phrases followed by tentative questioning when probed.",
        prior_knowledge_score=0.6,
        curiosity_level=0.45,
        communication_style="rote recitation, formula-reliant, tentative when probed conceptually"
    ),

    "karthik_distracted": BenchmarkPersona(
        id="karthik_distracted",
        name="Karthik Das",
        name_bengali="কার্তিক দাস",
        class_grade="Class 8",
        location="Howrah / Peri-urban district",
        medium="Bengali-medium school",
        persona_type="The Taciturn / Distracted Learner",
        ability_band="Low-Moderate",
        background="High-energy student who gets easily distracted or gives minimal one-word answers unless engaged with relatable, exciting local examples (sports, food, vehicles).",
        voice_signature="Brief, informal, one-line utterances. Uses colloquial slang and needs lively teacher scaffolding to stay on track.",
        error_profile="Gives hasty answers, skips intermediate reasoning steps, needs patient probing.",
        bengali_register_notes="Informal colloquial Bengali (হালকা কথ্য ভঙ্গি, সংক্ষিপ্ত বাক্য).",
        prior_knowledge_score=0.4,
        curiosity_level=0.5,
        communication_style="terse, informal, requires relatable real-world hooks, easily distracted"
    )
}


def get_benchmark_persona(persona_id: str) -> BenchmarkPersona:
    """Retrieve benchmark persona by ID, default to Aarav if not found."""
    return BENCHMARK_PERSONAS.get(persona_id, BENCHMARK_PERSONAS["aarav_topper"])


def get_all_benchmark_personas() -> List[BenchmarkPersona]:
    """Returns all 5 benchmark personas."""
    return list(BENCHMARK_PERSONAS.values())
