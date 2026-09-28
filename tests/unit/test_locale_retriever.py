import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from synthetictutor.knowledge.locale_retriever import LocaleRetriever


class TestLocaleRetriever(unittest.TestCase):
    def setUp(self):
        locale_path = os.path.join(BASE_DIR, "locale.json")
        self.retriever = LocaleRetriever(locale_path=locale_path)

    def test_state_summary_structure(self):
        """Verify state_summary defaults to WBBSE and Bengali."""
        data = self.retriever.locale_data
        summary = data.get("state_summary", data.get("region_summary", {}))
        self.assertEqual(summary.get("default_board"), "WBBSE")
        self.assertEqual(summary.get("default_medium"), "Bengali")
        self.assertEqual(summary.get("total_districts"), 23)

    def test_all_6_zones_present(self):
        """Verify all 6 agro-cultural zones are registered in the retriever."""
        zones = self.retriever.get_all_zones()
        expected = ["north_bengal", "gangetic_plain", "rarh_plateau", "sundarbans_delta", "kolkata_metro", "medinipur_coastal"]
        for z in expected:
            self.assertIn(z, zones)
            zone_ctx = self.retriever.get_zone_context(z)
            self.assertIsNotNone(zone_ctx)
            self.assertIn("primary_rivers", zone_ctx)
            self.assertIn("primary_crops", zone_ctx)

    def test_23_district_coverage_and_mapping(self):
        """Verify all 23 districts are mapped to their proper zones."""
        districts = self.retriever.get_all_districts()
        self.assertGreaterEqual(len(districts), 23)

        # Spot-check districts across zones
        self.assertEqual(self.retriever.get_zone_for_district("darjeeling"), "north_bengal")
        self.assertEqual(self.retriever.get_zone_for_district("murshidabad"), "gangetic_plain")
        self.assertEqual(self.retriever.get_zone_for_district("purulia"), "rarh_plateau")
        self.assertEqual(self.retriever.get_zone_for_district("south24parganas"), "sundarbans_delta")
        self.assertEqual(self.retriever.get_zone_for_district("kolkata"), "kolkata_metro")
        self.assertEqual(self.retriever.get_zone_for_district("purbamedinipur"), "medinipur_coastal")

    def test_multi_zone_localization_triggers(self):
        """Verify trigger detection across various zones in West Bengal."""
        # North Bengal
        self.assertTrue(self.retriever.is_localization_requested("জলপাইগুড়ি জেলার চা বাগান সম্পর্কিত অংক"))
        # Gangetic Plain
        self.assertTrue(self.retriever.is_localization_requested("মুর্শিদাবাদের রেশম ও পাট চাষ"))
        # Rarh Plateau
        self.assertTrue(self.retriever.is_localization_requested("পুরুলিয়ার লাল মাটির দেশের ছৌ নাচ"))
        # Sundarbans
        self.assertTrue(self.retriever.is_localization_requested("সুন্দরবনের ম্যানগ্রোভ বনের শ্বাসমূল ও বনবিবি"))
        # Kolkata Metro
        self.assertTrue(self.retriever.is_localization_requested("কলকাতার ট্রাম ও হাওড়া ব্রিজের পদার্থবিদ্যা"))
        # Medinipur Coastal
        self.assertTrue(self.retriever.is_localization_requested("তমলুকের পান বরজ ও দিঘার কাজু বাদাম"))
        # Negative test (standard generic math query should NOT trigger localization)
        self.assertFalse(self.retriever.is_localization_requested("একটি দ্বিঘাত সমীকরণ ax^2 + bx + c = 0 সমাধান করো"))

    def test_district_detection_from_query(self):
        """Verify district extraction from native Bengali sentences."""
        self.assertEqual(self.retriever.detect_target_district("বাঁকুড়ার বিষ্ণুপুরের টেরাকোটা মন্দির"), "bankura")
        self.assertEqual(self.retriever.detect_target_district("বীরভূমের শান্তিনিকেতনের পৌষ মেলা"), "birbhum")
        self.assertEqual(self.retriever.detect_target_district("হুগলীর তারকেশ্বরের আলু চাষ"), "hooghly")
        self.assertEqual(self.retriever.detect_target_district("পূর্ব মেদিনীপুরের কোলাঘাটের ফুল বাজার"), "purbamedinipur")

    def test_retrieval_of_sourced_facts(self):
        """Verify snippet retrieval carries accurate citations."""
        snippets = self.retriever.retrieve_locale_facts("বাঁকুড়ার বৃষ্টিপাত ও মাটির প্রকৃতি", target_district="bankura")
        self.assertGreater(len(snippets), 0)
        climate_snip = next((s for s in snippets if s.category == "climate"), None)
        if climate_snip:
            self.assertIn("IMD", climate_snip.source)


if __name__ == "__main__":
    unittest.main()
