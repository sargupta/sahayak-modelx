"""
Local Entities and Affordance Contracts for SahayakAI Grounding-First Pipeline.
Defines structured local concepts, agro-ecological anchors, cultural entities,
and pedagogical substitution rules across all 6 zones of West Bengal.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class EntityKind(str, Enum):
    CROP = "crop"
    FRUIT = "fruit"
    FOOD = "food"
    DISH = "dish"
    FESTIVAL = "festival"
    OCCUPATION = "occupation"
    ANIMAL = "animal"
    LANDMARK = "landmark"
    TOOL = "tool"
    WEATHER = "weather"


class SubstitutionStrategy(str, Enum):
    DIRECT_CONTEXT = "DIRECT_CONTEXT"
    ANALOGY = "ANALOGY"
    PROBLEM_SETTING = "PROBLEM_SETTING"
    WORKED_EXAMPLE = "WORKED_EXAMPLE"
    VOCABULARY_BRIDGING = "VOCABULARY_BRIDGING"


class AuthenticityMeta(BaseModel):
    validated_by: str = Field(default="pending_teacher", description="Teacher ID, validator version, or 'pending_teacher'")
    note: Optional[str] = Field(default=None, description="Pedagogical justification or grounding notes")


class LocalEntity(BaseModel):
    id: str = Field(..., description="Unique slug for the entity (e.g., wb_ent_pan_boroj)")
    name_bengali: str = Field(..., description="Primary Bengali name with native orthography")
    transliteration: Optional[str] = Field(default=None, description="English transliteration")
    kind: EntityKind = Field(..., description="Categorical type of entity")
    zones: List[str] = Field(..., description="List of associated West Bengal zones")
    description: str = Field(..., description="Educational and physical description of the entity")
    affordances: Dict[str, List[str]] = Field(default_factory=dict, description="Physical and cultural affordance traits")
    authenticity: AuthenticityMeta = Field(default_factory=AuthenticityMeta)


class SubstitutionRule(BaseModel):
    id: str
    zone: str
    strategy: SubstitutionStrategy
    canonical_concept: str
    local_entity_id: str
    description: str
    disanalogy_flags: List[str] = Field(default_factory=list, description="Mandatory flags pointing out physical discrepancies in analogies")


class EntityRegistry:
    """In-memory index and query engine for West Bengal local entities."""

    def __init__(self, entities: Optional[List[LocalEntity]] = None):
        self._entities: Dict[str, LocalEntity] = {e.id: e for e in (entities or [])}
        self._by_zone: Dict[str, List[LocalEntity]] = {}
        self._by_kind: Dict[EntityKind, List[LocalEntity]] = {}
        self._reindex()

    def _reindex(self):
        self._by_zone.clear()
        self._by_kind.clear()
        for ent in self._entities.values():
            for z in ent.zones:
                self._by_zone.setdefault(z, []).append(ent)
            self._by_kind.setdefault(ent.kind, []).append(ent)

    def add(self, entity: LocalEntity):
        self._entities[entity.id] = entity
        self._reindex()

    def get(self, entity_id: str) -> Optional[LocalEntity]:
        return self._entities.get(entity_id)

    def get_by_zone(self, zone: str) -> List[LocalEntity]:
        return self._by_zone.get(zone, [])

    def get_by_kind(self, kind: EntityKind) -> List[LocalEntity]:
        return self._by_kind.get(kind, [])

    def total_count(self) -> int:
        return len(self._entities)
