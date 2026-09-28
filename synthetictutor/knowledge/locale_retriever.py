"""
Locale Retriever Engine for SahayakAI Grounding-First Pipeline.
Provides verified regional facts, district context, economic indicators, and cultural markers
across all 23 districts and 6 agro-cultural zones of West Bengal from locale.json.
Enforces the strict rule against ungrounded hallucination or forced irrelevant localization.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class LocaleSnippet:
    district: str
    category: str
    fact_key: str
    fact_value: Any
    source: str
    snippet_text: str
    relevance_score: float = 0.0


class LocaleRetriever:
    """First-class retriever for verified West Bengal local context from locale.json."""

    DISTRICT_SYNONYMS: Dict[str, List[str]] = {
        # --- North Bengal Zone ---
        'darjeeling': ['দার্জিলিং', 'দার্জিলিং জেলা', 'darjeeling', 'কার্শিয়াং', 'kurseong', 'মিরিক', 'mirik'],
        'kalimpong': ['কালিম্পং', 'কালিম্পং জেলা', 'kalimpong', 'লাভাতে', 'প্যাদং', 'pedong'],
        'jalpaiguri': ['জলপাইগুড়ি', 'জলপাইগুড়ি', 'jalpaiguri', 'মালবাজার', 'ধূপগুড়ি', 'ময়নাগুড়ি', 'গরুমারা', 'চালসা'],
        'alipurduar': ['আলিপুরদুয়ার', 'আলিপুরদুয়ার', 'alipurduar', 'ফালাকাটা', 'falakata', 'মাদারিহাট', 'জয়গাঁ', 'বক্সা', 'buxa'],
        'coochbehar': ['কোচবিহার', 'coochbehar', 'cooch behar', 'দিনহাটা', 'dinhata', 'মাথাভাঙা', 'mathabhanga', 'তুফানগঞ্জ'],
        'malda': ['মালদা', 'মালদহ', 'malda', 'ইংরেজবাজার', 'english bazar', 'পুরাতন মালদা', 'গাজোল', 'চাঁচল'],
        'uttardinajpur': ['উত্তর দিনাজপুর', 'uttar dinajpur', 'রায়গঞ্জ', 'raiganj', 'ইসলামপুর', 'islampur', 'কালিয়াগঞ্জ'],
        'dakshindinajpur': ['দক্ষিণ দিনাজপুর', 'dakshin dinajpur', 'বালুরঘাট', 'balurghat', 'গঙ্গারামপুর', 'বুনিয়াদপুর', 'কুশমণ্ডি'],

        # --- Gangetic Plain Zone ---
        'murshidabad': ['মুর্শিদাবাদ', 'murshidabad', 'বহরমপুর', 'baharampur', 'জিয়াগঞ্জ', 'লালবাগ', 'ধুলিয়ান', 'ফরাক্কা'],
        'nadia': ['নদীয়া', 'নদিয়া', 'nadia', 'কৃষ্ণনগর', 'krishnanagar', 'শান্তিপুর', 'নবদ্বীপ', 'কল্যাণী', 'রানাঘাট'],
        'purbabardhaman': ['পূর্ব বর্ধমান', 'বর্ধমান', 'purba bardhaman', 'bardhaman', 'কালনা', 'কাটোয়া', 'মেমারি', 'নতুনগ্রাম'],
        'hooghly': ['হুগলী', 'হুগলি', 'hooghly', 'চুঁচুড়া', 'চন্দননগর', 'শ্রীরামপুর', 'তারকেশ্বর', 'আরামবাগ', 'সিঙ্গুর'],
        'howrah_rural': ['হাওড়া গ্রামীণ', 'আমতা', 'বাগনান', 'উদয়নারায়ণপুর', 'শ্যামপুর', 'পাঁচলা', 'domjur'],
        'north24parganas_rural': ['উত্তর ২৪ পরগনা গ্রামীণ', 'বনগাঁ', 'হাবড়া', 'দেগঙ্গা', 'স্বরূপনগর', 'গাইঘাটা'],

        # --- Rarh Plateau Zone ---
        'purulia': ['পুরুলিয়া', 'পুরুলিয়া', 'purulia', 'রঘুনাথপুর', 'ঝালদা', 'বাগমুণ্ডি', 'চড়িদা', 'অযোধ্যা পাহাড়'],
        'bankura': ['বাঁকুড়া', 'বাঁকুড়া', 'bankura', 'বিষ্ণুপুর', 'bishnupur', 'পাঁচমুড়া', 'বিকনা', 'শুশুনিয়া', 'বিহারীনাথ'],
        'jhargram': ['ঝাড়গ্রাম', 'ঝাড়গ্রাম', 'jhargram', 'বেলপাহাড়ী', 'গোপীবল্লভপুর', 'চিলকিগড়', 'জঙ্গলমহল'],
        'paschimbardhaman': ['পশ্চিম বর্ধমান', 'আসানসোল', 'asansol', 'দুর্গাপুর', 'durgapur', 'রাণীগঞ্জ', 'কুলটি', 'চিত্তরঞ্জন'],
        'birbhum': ['বীরভূম', 'birbhum', 'সিউড়ি', 'suri', 'বোলপুর', 'bolpur', 'শান্তিনিকেতন', 'santiniketan', 'তারাপীঠ', 'বক্রেশ্বর'],

        # --- Sundarbans Delta Zone ---
        'south24parganas': ['দক্ষিণ ২৪ পরগনা', 'south 24 parganas', 'সুন্দরবন', 'sundarban', 'সন্দেশখালি', 'গোসাবা', 'ক্যানিং', 'বাসন্তী', 'কাকদ্বীপ', 'সাগরদ্বীপ', 'নামখানা'],
        'north24parganas_coastal': ['হিঙ্গলগঞ্জ', 'hingalganj', 'হাসনাবাদ', 'hasnabad', 'মীনাক্ষাঁ', 'সন্দেশখালি'],

        # --- Kolkata Metro Zone ---
        'kolkata': ['কলকাতা', 'কলিকাতা', 'kolkata', 'calcutta', 'কলেজ স্ট্রিট', 'শ্যামবাজার', 'গড়িয়াহাট', 'আলিপুর'],
        'howrah_urban': ['হাওড়া শহর', 'হাওড়া স্টেশন', 'হাওড়া ব্রিজ', 'শিবপুর', 'বালি', 'সালকিয়া'],

        # --- Medinipur Coastal Zone ---
        'purbamedinipur': ['পূর্ব মেদিনীপুর', 'purba medinipur', 'তমলুক', 'tamluk', 'দিঘা', 'digha', 'হলদিয়া', 'haldia', 'কাঁথি', 'মন্দারমণি', 'কোলাঘাট', 'নন্দীগ্রাম'],
        'paschimmedinipur': ['পশ্চিম মেদিনীপুর', 'paschim medinipur', 'মেদিনীপুর শহর', 'খড়গপুর', 'kharagpur', 'ঘাটাল', 'সবং', 'পিংলা', 'গড়বেতা']
    }

    ZONE_MAPPING: Dict[str, str] = {
        'darjeeling': 'north_bengal',
        'kalimpong': 'north_bengal',
        'jalpaiguri': 'north_bengal',
        'alipurduar': 'north_bengal',
        'coochbehar': 'north_bengal',
        'malda': 'north_bengal',
        'uttardinajpur': 'north_bengal',
        'dakshindinajpur': 'north_bengal',

        'murshidabad': 'gangetic_plain',
        'nadia': 'gangetic_plain',
        'purbabardhaman': 'gangetic_plain',
        'hooghly': 'gangetic_plain',
        'howrah_rural': 'gangetic_plain',
        'north24parganas_rural': 'gangetic_plain',

        'purulia': 'rarh_plateau',
        'bankura': 'rarh_plateau',
        'jhargram': 'rarh_plateau',
        'paschimbardhaman': 'rarh_plateau',
        'birbhum': 'rarh_plateau',

        'south24parganas': 'sundarbans_delta',
        'north24parganas_coastal': 'sundarbans_delta',

        'kolkata': 'kolkata_metro',
        'howrah_urban': 'kolkata_metro',

        'purbamedinipur': 'medinipur_coastal',
        'paschimmedinipur': 'medinipur_coastal'
    }

    LOCALE_TRIGGERS: List[re.Pattern] = [
        # General & Agricultural
        re.compile(r'চা[\s\-]বাগান|চা[\s\-]পাতা|চা[\s\-]শ্রমিক|tea garden', re.IGNORECASE),
        re.compile(r'(?<![\u0980-\u09FF])(পাট|পাট\s*জাগ|সোনালী\s*আঁশ|jute)(?![\u0980-\u09FF])', re.IGNORECASE),
        re.compile(r'আলু\s*চাষ|কোল্ড\s*স্টোরেজ|হুগলির\s*আলু|জ্যোতি\s*আলু', re.IGNORECASE),
        re.compile(r'পান\s*বরজ|মিষ্টি\s*পান|কাজু\s*বাদাম|মাদুর\s*কাঠি', re.IGNORECASE),
        re.compile(r'(?<![\u0980-\u09FF])(বিঘা|কাঠা|ছটাক|সের|মণ|কুইন্টাল)(?![\u0980-\u09FF])', re.IGNORECASE),
        re.compile(r'স্থানীয়\s+উদাহরণ|আঞ্চলিক\s+প্রেক্ষাপট|গ্রামের\s+উদাহরণ|লোকাল\s+উদাহরণ', re.IGNORECASE),

        # North Bengal
        re.compile(r'জলপাইগুড়ি|জলপাইগুড়ি|দার্জিলিং|কালিম্পং|আলিপুরদুয়ার|কোচবিহার|মালদা|দিনাজপুর', re.IGNORECASE),
        re.compile(r'ডুয়ার্স|ডুয়াস|তরাই|dooars|terai|তিস্তা|তোর্সা|জলঢাকা|মহানন্দা|টয়\s*ট্রেন|ভাওয়াইয়া|গোমীরা', re.IGNORECASE),

        # Gangetic Plain
        re.compile(r'মুর্শিদাবাদ|নদীয়া|নদিয়া|বর্ধমান|হুগলী|হুগলি|ভাগীরথী|নবদ্বীপ|শান্তিপুরী|বালুচরী|মিহিদানা|সীতাভোগ|চন্দননগর|জগদ্ধাত্রী', re.IGNORECASE),

        # Rarh Plateau
        re.compile(r'পুরুলিয়া|বাঁকুড়া|ঝাড়গ্রাম|বীরভূম|রাঢ়|লাল\s*মাটি|শাল\s*বন|ছৌ|ছৌ\s*নাচ|পোড়ামাটির\s*ঘোড়া|ডোকরা|শান্তিনিকেতন|সোনাঝুরি|খোয়াই|আসানসোল|কয়লাখনি', re.IGNORECASE),

        # Sundarbans
        re.compile(r'সুন্দরবন|ম্যানগ্রোভ|শ্বাসমূল|সুন্দরী\s*গাছ|রয়্যাল\s*বেঙ্গল\s*টাইগার|মৌয়াল|বনবিবি|জোয়ার\s*ভাটা|নদীর\s*বাঁধ|মাতলা|খেয়া', re.IGNORECASE),

        # Kolkata Metro
        re.compile(r'কলকাতা|হাওড়া\s*ব্রিজ|রবীন্দ্র\s*সেতু|ট্রাম|মেট্রো\s*রেল|কলেজ\s*স্ট্রিট|বইপাড়া|রসগোল্লা|পূর্ব\s*কলকাতা\s*জলাভূমি', re.IGNORECASE),

        # Medinipur Coastal
        re.compile(r'মেদিনীপুর|দিঘা|মন্দারমণি|হলদিয়া|তমলুক|কোলাঘাট|রূপনারায়ণ|পটচিত্র|পিংলা|সবং|কাঁসা[\s\-]পিতল', re.IGNORECASE)
    ]

    def __init__(self, locale_path: str = "locale.json"):
        self.locale_path = Path(locale_path)
        self.locale_data: Dict[str, Any] = {}
        self._load_locale_data()

    def _load_locale_data(self):
        if not self.locale_path.exists():
            raise FileNotFoundError(f"Locale file not found at: {self.locale_path}")
        with open(self.locale_path, "r", encoding="utf-8") as f:
            self.locale_data = json.load(f)

    def is_localization_requested(self, query: str) -> bool:
        """Determines if the user prompt explicitly or implicitly requests verified regional context."""
        return any(pattern.search(query) for pattern in self.LOCALE_TRIGGERS)

    def detect_target_district(self, query: str) -> Optional[str]:
        """Detects mentioned district key if explicitly stated in query."""
        q_lower = query.lower()
        for dist_key, synonyms in self.DISTRICT_SYNONYMS.items():
            if any(syn.lower() in q_lower for syn in synonyms):
                return dist_key
        return None

    def get_zone_for_district(self, district_key: str) -> Optional[str]:
        """Maps a district slug to its overarching agro-cultural zone."""
        return self.ZONE_MAPPING.get(district_key)

    def get_all_districts(self) -> List[str]:
        """Returns all recognized district keys."""
        return list(self.DISTRICT_SYNONYMS.keys())

    def get_all_zones(self) -> List[str]:
        """Returns all 6 recognized zone keys."""
        return list(self.locale_data.get("zones", {}).keys())

    def get_zone_context(self, zone_key: str) -> Optional[Dict[str, Any]]:
        """Retrieves zone metadata and agro-ecological characteristics."""
        return self.locale_data.get("zones", {}).get(zone_key)

    def get_district_context(self, district_key: str) -> Optional[Dict[str, Any]]:
        """Retrieves deep district-level grounded facts and indicators."""
        return self.locale_data.get("districts", {}).get(district_key)

    def get_entities_for_zone(self, zone_key: str) -> List[Dict[str, Any]]:
        """Returns local entities associated with a specific zone."""
        all_entities = self.locale_data.get("local_entities", [])
        return [ent for ent in all_entities if zone_key in ent.get("zones", [])]

    def get_substitutions_for_zone(self, zone_key: str) -> List[Dict[str, Any]]:
        """Returns curated pedagogical substitutions with disanalogy flags for a zone."""
        all_subs = self.locale_data.get("substitutions", [])
        return [sub for sub in all_subs if sub.get("zone") == zone_key]

    def retrieve_locale_facts(
        self,
        query: str,
        target_district: Optional[str] = None,
        max_facts: int = 4
    ) -> List[LocaleSnippet]:
        """
        Retrieves verified facts from locale.json relevant to the query and target district/zone.
        """
        if not target_district:
            target_district = self.detect_target_district(query)

        snippets: List[LocaleSnippet] = []
        districts_data = self.locale_data.get("districts", {})
        state_summary = self.locale_data.get("state_summary", self.locale_data.get("region_summary", {}))

        # 1. District-specific retrieval
        if target_district and target_district in districts_data:
            d_info = districts_data[target_district]
            dist_name = d_info.get("name", target_district.capitalize())

            # Terrain & setting
            if any(w in query for w in ["বাগান", "বন", "নদী", "পাহাড়", "সমুদ্র", "ভূপ্রকৃতি", "মাটি"]):
                snippets.append(LocaleSnippet(
                    district=dist_name,
                    category="terrain",
                    fact_key="terrain",
                    fact_value=d_info.get("terrain"),
                    source="locale.json",
                    snippet_text=f"ভৌগোলিক পরিবেশ: {dist_name} অঞ্চলে {d_info.get('setting_type', '')}, ভূপ্রকৃতি: {d_info.get('terrain', '')}।",
                    relevance_score=0.9
                ))

            # Climate & soil
            if any(w in query for w in ["বৃষ্টি", "বৃষ্টিপাত", "তাপমাত্রা", "মাটি", "pH", "আবহাওয়া", "জলবায়ু"]):
                climate = d_info.get("climate", {})
                if "average_annual_rainfall_mm" in climate:
                    val = climate["average_annual_rainfall_mm"]["value"]
                    src = climate["average_annual_rainfall_mm"]["source"]
                    snippets.append(LocaleSnippet(
                        district=dist_name,
                        category="climate",
                        fact_key="average_annual_rainfall_mm",
                        fact_value=val,
                        source=src,
                        snippet_text=f"বার্ষিক গড় বৃষ্টিপাত: {val} মিমি (উৎস: {src})। মাটি: {climate.get('typical_soil_type', '')}।",
                        relevance_score=0.95
                    ))

            # Agriculture & economics
            if any(w in query for w in ["চাষ", "ফসল", "ধান", "পাট", "আলু", "চা", "মজুরি", "অর্থনীতি", "দাম"]):
                agri = d_info.get("agriculture_and_economy", {})
                crops = agri.get("major_crops", [])
                if crops:
                    snippets.append(LocaleSnippet(
                        district=dist_name,
                        category="agriculture",
                        fact_key="major_crops",
                        fact_value=crops,
                        source="locale.json",
                        snippet_text=f"প্রধান কৃষিজ ফসল: {', '.join(crops[:3])}।",
                        relevance_score=0.85
                    ))

        # 2. General State / Common facts fallback
        if len(snippets) < max_facts and any(w in query for w in ["মজুরি", "খাতা", "কলম", "দিনমজুরি", "টাকা"]):
            if "government_mgnrega_daily_wage_inr" in state_summary:
                wage_val = state_summary["government_mgnrega_daily_wage_inr"]["value"]
                wage_src = state_summary["government_mgnrega_daily_wage_inr"]["source"]
                snippets.append(LocaleSnippet(
                    district="West Bengal",
                    category="state_economy",
                    fact_key="mgnrega_wage",
                    fact_value=wage_val,
                    source=wage_src,
                    snippet_text=f"সরকারি ১০০ দিনের কাজের দৈনিক মজুরি: ₹{wage_val} (উৎস: {wage_src})।",
                    relevance_score=0.8
                ))

        return snippets[:max_facts]
