"""
Unit tests for West Bengal locale.json Schema and Grounding Fact Integrity.
Validates 6 agro-cultural zones, 23 districts, and {value, source} citation contracts.
"""

import json
import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class TestLocaleSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        locale_path = os.path.join(BASE_DIR, "locale.json")
        cls.assertTrue(cls, os.path.exists(locale_path), f"locale.json not found at {locale_path}")
        with open(locale_path, "r", encoding="utf-8") as f:
            cls.data = json.load(f)

    def test_top_level_keys(self):
        """Ensure all required root keys exist."""
        required_keys = ["state_summary", "zones", "districts", "local_entities"]
        for key in required_keys:
            self.assertIn(key, self.data, f"Missing top-level key: {key}")
        self.assertTrue(
            "substitutions" in self.data or "substitution_rules" in self.data,
            "Missing substitutions key",
        )

    def test_state_summary_quantitative_contracts(self):
        """Verify all quantitative figures in state_summary follow {value, source} structure."""
        summary = self.data["state_summary"]
        self.assertEqual(summary.get("total_districts"), 23)
        self.assertEqual(summary.get("major_agro_cultural_zones"), 6)

        quant_fields = [
            ("mgnrega_daily_wage_fy2024_25_inr", "government_mgnrega_daily_wage_inr"),
            ("urea_mrp_inr_per_45kg_bag", "urea_subsidized_price_inr_per_45kg_bag"),
        ]
        for primary, fallback in quant_fields:
            field = primary if primary in summary else fallback
            self.assertIn(field, summary, f"Missing quantitative field: {primary}")
            fact = summary[field]
            self.assertIsInstance(fact, dict, f"{field} must be a dict with value and source")
            self.assertIn("value", fact, f"{field} missing 'value'")
            self.assertIn("source", fact, f"{field} missing 'source'")
            self.assertIsInstance(fact["value"], (int, float), f"{field} 'value' must be numeric")
            self.assertTrue(len(fact["source"]) > 5, f"{field} source too short: {fact['source']}")

    def test_six_zones_present(self):
        """Verify all 6 agro-cultural zones are defined with required attributes."""
        zones = self.data["zones"]
        expected_zones = [
            "north_bengal",
            "gangetic_plain",
            "rarh_plateau",
            "sundarbans_delta",
            "kolkata_metro",
            "medinipur_coastal",
        ]
        self.assertEqual(len(zones), 6)
        for z in expected_zones:
            self.assertIn(z, zones, f"Missing zone: {z}")
            zone_obj = zones[z]
            self.assertIn("name", zone_obj)
            self.assertIn("name_bengali", zone_obj)
            self.assertIn("districts", zone_obj)
            self.assertIn("primary_rivers", zone_obj)
            self.assertIn("primary_crops", zone_obj)
            self.assertIn("hazards", zone_obj)
            self.assertIn("cultural_markers", zone_obj)
            self.assertIn("local_wage_mgnrega_inr", zone_obj)
            self.assertIn("value", zone_obj["local_wage_mgnrega_inr"])
            self.assertIn("source", zone_obj["local_wage_mgnrega_inr"])

    def test_twenty_three_districts_schema(self):
        """Verify all districts are present with complete metadata and sourced facts."""
        districts = self.data["districts"]
        self.assertGreaterEqual(len(districts), 23, "Must have at least 23 districts")

        for d_key, d_obj in districts.items():
            self.assertIn("name", d_obj, f"{d_key} missing name")
            self.assertIn("zone", d_obj, f"{d_key} missing zone")
            self.assertIn("terrain", d_obj, f"{d_key} missing terrain")
            self.assertIn("demographics", d_obj, f"{d_key} missing demographics")
            self.assertIn("climate", d_obj, f"{d_key} missing climate")
            self.assertTrue(
                "agriculture_and_economy" in d_obj or "economy_and_urban_context" in d_obj,
                f"{d_key} missing economic context",
            )

            # Check that zone reference is valid
            self.assertIn(d_obj["zone"], self.data["zones"], f"{d_key} has invalid zone '{d_obj['zone']}'")

            # Check core quantitative facts follow {value, source} or {min, max, source}
            demo = d_obj["demographics"]
            self.assertIn("literacy_rate_percentage", demo)
            self.assertIn("value", demo["literacy_rate_percentage"])
            self.assertIn("source", demo["literacy_rate_percentage"])
            self.assertTrue(len(demo["literacy_rate_percentage"]["source"]) > 3)

            climate = d_obj["climate"]
            self.assertIn("average_annual_rainfall_mm", climate)
            self.assertIn("value", climate["average_annual_rainfall_mm"])
            self.assertIn("source", climate["average_annual_rainfall_mm"])
            self.assertTrue(len(climate["average_annual_rainfall_mm"]["source"]) > 3)

            if "soil_ph_range" in climate:
                ph = climate["soil_ph_range"]
                self.assertIn("min", ph)
                self.assertIn("max", ph)
                self.assertIn("source", ph)

            if "agriculture_and_economy" in d_obj:
                agri = d_obj["agriculture_and_economy"]
                self.assertIn("major_crops", agri)
                self.assertIsInstance(agri["major_crops"], list)
                self.assertGreater(len(agri["major_crops"]), 0)


if __name__ == "__main__":
    unittest.main()
