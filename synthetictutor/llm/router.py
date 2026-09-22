"""
LLM Router and Mock LLM Client for testing and flexible provider routing.
"""

import json
import logging
from typing import Optional, Type, TypeVar, Dict, Any
from pydantic import BaseModel

from synthetictutor.llm.base import BaseLLMClient
from synthetictutor.llm.gemini import GeminiLLMClient
from synthetictutor.llm.openai_client import OpenAICompatibleClient
from synthetictutor.core.exceptions import LLMProviderError

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class MockLLMClient(BaseLLMClient):
    """Deterministic Mock LLM client for unit tests and local pipeline dry runs."""

    def __init__(self, predefined_responses: Optional[Dict[str, Any]] = None):
        self.predefined_responses = predefined_responses or {}

    async def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        for key, val in self.predefined_responses.items():
            if key in prompt or (system_instruction and key in system_instruction):
                if isinstance(val, str):
                    return val
                return json.dumps(val)
        return "Mock response: The concept is fundamental in physics and requires understanding prerequisites."

    async def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> T:
        if self.predefined_responses:
            try:
                return response_schema.model_validate(self.predefined_responses)
            except Exception:
                pass
            for key, val in self.predefined_responses.items():
                if key in prompt:
                    if isinstance(val, dict):
                        try:
                            return response_schema.model_validate(val)
                        except Exception:
                            pass

        # Fallback dummy construction based on schema name
        schema_name = response_schema.__name__
        if schema_name == "ExtractedConceptList":
            from synthetictutor.core.schemas import Concept, Misconception
            if "Comprehensive Curriculum" in prompt or "Motion" in prompt:
                m1 = Misconception(id="m1", description="Force is needed to maintain constant velocity", typical_trigger="Pushing a box", correct_conception="Newton's 1st Law: Constant velocity requires zero net force")
                m2 = Misconception(id="m2", description="Heavy objects have more acceleration under gravity", typical_trigger="Free fall", correct_conception="g is constant for all masses in vacuum")
                m3 = Misconception(id="m3", description="Electrons orbit like planets in fixed physical rings", typical_trigger="Bohr model", correct_conception="Electrons exist in probability clouds/orbitals")
                m4 = Misconception(id="m4", description="Energy is consumed and destroyed when used", typical_trigger="Car burning fuel", correct_conception="Conservation of energy: Energy transforms from chemical to thermal/kinetic")

                concepts = [
                    Concept(id="c_velocity", name="Velocity & Acceleration", description="Speed with direction and rate of velocity change.", domain="Physics", grade_level="Grade 9", prerequisite_ids=[]),
                    Concept(id="c_newton1", name="Newton's First Law of Motion", description="Law of Inertia: objects maintain velocity unless forced.", domain="Physics", grade_level="Grade 9", prerequisite_ids=["c_velocity"], misconceptions=[m1]),
                    Concept(id="c_newton2", name="Newton's Second Law (F=ma)", description="Force equals mass times acceleration.", domain="Physics", grade_level="Grade 9", prerequisite_ids=["c_newton1"]),
                    Concept(id="c_atomic_structure", name="Atomic Structure & Subatomic Particles", description="Protons, neutrons, electrons, and atomic numbers.", domain="Chemistry", grade_level="Grade 9", prerequisite_ids=[], misconceptions=[m3]),
                    Concept(id="c_isotopes", name="Isotopes & Atomic Mass", description="Atoms of same element with different neutron count.", domain="Chemistry", grade_level="Grade 9", prerequisite_ids=["c_atomic_structure"]),
                    Concept(id="c_cell_biology", name="Cellular Organelles & Powerhouse (ATP)", description="Mitochondria, chloroplasts, and ATP synthesis.", domain="Biology", grade_level="Grade 9", prerequisite_ids=[]),
                    Concept(id="c_osmosis", name="Osmosis & Membrane Transport", description="Diffusion of water molecules across semi-permeable membranes.", domain="Biology", grade_level="Grade 9", prerequisite_ids=["c_cell_biology"]),
                    Concept(id="c_work_energy", name="Work, Kinetic & Potential Energy", description="Work done by force and conservation of energy.", domain="Physics", grade_level="Grade 9", prerequisite_ids=["c_newton2"], misconceptions=[m4]),
                ]
                return response_schema.model_validate({"concepts": concepts})
            else:
                return response_schema.model_validate({
                    "concepts": [
                        Concept(id="concept_grav_01", name="Universal Gravitation", description="Attractive force between all masses.", domain="Physics", grade_level="Grade 9", prerequisite_ids=[], misconceptions=[])
                    ]
                })

        try:
            return response_schema.model_validate({})
        except Exception:
            fields = response_schema.model_fields
            constructed = {}
            for fname, finfo in fields.items():
                if finfo.is_required():
                    import typing
                    import enum
                    origin = typing.get_origin(finfo.annotation)
                    if isinstance(finfo.annotation, type) and issubclass(finfo.annotation, enum.Enum):
                        constructed[fname] = list(finfo.annotation)[0].value
                    elif finfo.annotation == str:
                        constructed[fname] = f"mock_{fname}"
                    elif finfo.annotation == bool:
                        constructed[fname] = True
                    elif finfo.annotation == int:
                        constructed[fname] = 1
                    elif finfo.annotation == float:
                        constructed[fname] = 0.95 if fname == "score" else 0.5
                    elif finfo.annotation == list or origin is list:
                        constructed[fname] = ["mock_turn_1", "mock_turn_2"]
                    elif finfo.annotation == dict or origin is dict:
                        constructed[fname] = {}
                    else:
                        constructed[fname] = "mock"
            return response_schema.model_validate(constructed)


def get_llm_client(provider: str = "mock", model_name: Optional[str] = None, **kwargs) -> BaseLLMClient:
    """Factory function to instantiate configured LLM provider."""
    import os
    provider_lower = provider.lower()
    if provider_lower == "mock":
        return MockLLMClient(predefined_responses=kwargs.get("mock_responses"))
    elif provider_lower in ("gemini", "google"):
        return GeminiLLMClient(model_name=model_name or "gemini-2.0-flash", **kwargs)
    elif provider_lower in ("openai", "vllm", "ollama"):
        return OpenAICompatibleClient(model_name=model_name or "gpt-4o-mini", **kwargs)
    elif provider_lower in ("omniroute", "omni"):
        omni_base = kwargs.get("base_url") or os.getenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
        omni_model = model_name or os.getenv("OMNIROUTE_MODEL", "omniroute-auto")
        logger.info(f"Connecting SyntheticTutor to OmniRoute Gateway at {omni_base} (Model: {omni_model})")
        return OpenAICompatibleClient(model_name=omni_model, base_url=omni_base, **kwargs)
    else:
        raise LLMProviderError(f"Unsupported LLM provider: {provider}")
