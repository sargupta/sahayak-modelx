"""
Unit tests for Local Entities, Affordance Contracts, and Pedagogical Substitution Rules.
Validates >= 40 entities, 6-zone coverage, affordance structures, and disanalogy constraints.
"""

import json
import os
import re
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from synthetictutor.knowledge.entities import (
    EntityKind,
    EntityRegistry,
    LocalEntity,
    SubstitutionRule,
    SubstitutionStrategy,
)


class TestLocalEntities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        locale_path = os.path.join(BASE_DIR, "locale.json")
        with open(locale_path, "r", encoding="utf-8") as f:
            cls.data = json.load(f)

        raw_entities = cls.data.get("local_entities", [])
        cls.entities = [LocalEntity(**e) for e in raw_entities]
        cls.registry = EntityRegistry(cls.entities)

        raw_rules = cls.data.get("substitutions", cls.data.get("substitution_rules", []))
        cls.rules = [SubstitutionRule(**r) for r in raw_rules]

    def test_total_entity_count_threshold(self):
        """Must have >= 40 structured local entities."""
        self.assertGreaterEqual(self.registry.total_count(), 40)

    def test_zone_distribution(self):
        """Every zone must have >= 8 associated entities."""
        expected_zones = [
            "north_bengal",
            "gangetic_plain",
            "rarh_plateau",
            "sundarbans_delta",
            "kolkata_metro",
            "medinipur_coastal",
        ]
        for zone in expected_zones:
            entities_in_zone = self.registry.get_by_zone(zone)
            self.assertGreaterEqual(
                len(entities_in_zone),
                8,
                f"Zone '{zone}' has {len(entities_in_zone)} entities, expected >= 8",
            )

    def test_bengali_orthography_and_affordance_structure(self):
        """Verify Bengali name, affordances, and authenticity for every entity."""
        bengali_pattern = re.compile(r"[\u0980-\u09FF]")

        for ent in self.entities:
            # Check ID format
            self.assertTrue(
                ent.id.startswith("wb_ent_"),
                f"Entity ID {ent.id} should start with 'wb_ent_'",
            )
            # Check Bengali text present
            self.assertTrue(
                bengali_pattern.search(ent.name_bengali),
                f"Entity {ent.id} has invalid Bengali name: {ent.name_bengali}",
            )
            # Check non-empty description
            self.assertGreater(len(ent.description), 10)
            # Check affordance contract
            self.assertIsInstance(ent.affordances, dict)
            self.assertGreater(
                len(ent.affordances),
                0,
                f"Entity {ent.id} missing affordance mappings",
            )
            # Check authenticity metadata
            self.assertIsNotNone(ent.authenticity)
            self.assertTrue(hasattr(ent.authenticity, "validated_by"))

    def test_substitution_rules_integrity(self):
        """Verify pedagogical substitution rules and disanalogy constraints."""
        self.assertGreaterEqual(len(self.rules), 15, "Expected >= 15 substitution rules")

        for rule in self.rules:
            # Rule must reference a valid entity
            target_entity = self.registry.get(rule.local_entity_id)
            self.assertIsNotNone(
                target_entity,
                f"Rule {rule.id} references non-existent entity {rule.local_entity_id}",
            )
            # Rule must belong to a valid zone
            self.assertIn(
                rule.zone,
                self.data["zones"],
                f"Rule {rule.id} references invalid zone {rule.zone}",
            )
            # Check that analogies have disanalogy flags
            if rule.strategy == SubstitutionStrategy.ANALOGY:
                self.assertGreater(
                    len(rule.disanalogy_flags),
                    0,
                    f"Analogy rule {rule.id} must specify at least one disanalogy flag",
                )


if __name__ == "__main__":
    unittest.main()
