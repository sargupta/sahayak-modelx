# 🚀 SahayakAI 50,000-Sample SFT/DPO Dataset Generation Package

Welcome! This package contains everything required to generate, orchestrate, validate, and export the **50,000-sample production SFT dataset** for SahayakAI (`sahayak-modelx`).

---

## 📁 1. Core Package Resources

| File / Component | Path | Description |
| :--- | :--- | :--- |
| **Clean Master Grounding Corpus** | `datasets/grounding_corpus_clean_master.jsonl` | **18,695 clean, verified textbook chunks** across 66 books (0.00% OCR corruption). |
| **Verified Regional Facts** | `locale.json` | Authentic geographical, cultural, and environmental data for West Bengal & North Bengal. |
| **CLI Generation Runner** | `synthetictutor/scripts/generate_50k_dataset.py` | Full batch generator with stage filtering, progress tracking, and checkpointing. |
| **Batch Orchestrator Engine** | `synthetictutor/pipeline/batch_50k_generator.py` | 4D quota manager, Request-First retriever, compatibility gate, and DB persistence. |
| **Standalone Quality Auditor** | `synthetictutor/scripts/validate_50k_shard.py` | Standalone validator to audit any generated `.jsonl` file against all 8 quality gates. |
| **Unit Test Suite** | `tests/unit/test_batch_50k_generator.py` | Complete test coverage ensuring pipeline stability. |

---

## 🎯 2. Target Quotas & Distribution Matrix (50,000 Records)

### A. 4-Stage Educational Distribution
* **Primary (Classes 1–5 | WBBPE):** **10,000 records** (20%)
* **Middle School (Classes 6–8 | WBBSE):** **17,500 records** (35%)
* **Secondary (Classes 9–10 | WBBSE):** **15,000 records** (30%)
* **Higher Secondary (Classes 11–12 | WBCHSE):** **7,500 records** (15%)

### B. Task Family Mix
* **Concept Explanation:** 35% (17,500 records)
* **Guided Problem Solving:** 25% (12,500 records)
* **Detailed Lesson Plans (Teacher):** 15% (7,500 records)
* **Quiz & Question Sets (Evaluation):** 15% (7,500 records)
* **Worksheet & Practice:** 10% (5,000 records)

### C. Target Persona & Addressing
* **Teacher Requester (85%):** Addressed respectfully with `আপনি` (focuses on classroom pedagogy, lesson plans, misconceptions).
* **Student Requester (15%):** Addressed supportively with `তুমি/তোমরা` (direct concept tutoring, step-by-step problem solving).

---

## 📜 3. Standard System Prompt Template

Every sample must use the battle-tested system prompt:

```text
তুমি 'সহায়কএআই' (SahayakAI) — {board_full} বেঙ্গলি-মিডিয়াম স্কুলের শিক্ষার্থী ও শিক্ষক মহাশয়দের জন্য তৈরি একজন অভিজ্ঞ, বিশেষায়িত অ্যাকাডেমিক AI টিউটর। তোমার কাজ হলো পাঠ্যক্রম অনুযায়ী সহজ, সাবলীল, সঠিক ও শিক্ষার্থী-উপযোগী বাংলা ভাষায় পাঠদান, ধারণা ব্যাখ্যা, সমস্যা সমাধান এবং অধ্যয়ন-সংক্রান্ত সহায়তা প্রদান করা।

নিয়মাবলী ও সম্বোধন:
{১. শিক্ষক বা অভিভাবকের উদ্দেশ্যে সম্মানসূচক 'আপনি' ভাষা ব্যবহার করবে। / ১. শিক্ষার্থীর উদ্দেশ্যে সহজ, বন্ধুভাবাপন্ন 'তুমি/তোমরা' ভাষা ব্যবহার করবে।}
২. স্পষ্ট, প্রাঞ্জল ও স্বাভাবিক ভাষায় সরাসরি শিক্ষামূলক উত্তরে প্রবেশ করবে; অপ্রয়োজনীয় অভিবাদন, প্রশংসা বা motivational filler এড়িয়ে চলবে।
৩. পাঠ্যক্রম-উপযোগী প্রমিত বাংলা পরিভাষা ব্যবহার করবে।
৪. সাধারণ বাংলা গদ্যে সব সংখ্যা বাংলা অঙ্কে লিখবে: ০, ১, ২, ৩, ৪, ৫, ৬, ৭, ৮, ৯।
৫. বৈধ mathematical/scientific notation যেমন x², x^2, H₂O, CO₂, a₁, ∠ABC বা অনুরূপ notation বিকৃত করবে না।

শিক্ষাদান ও পাঠ্যক্রম-নির্ভরতা:
৬. পাঠ্যক্রম ও বিষয়বস্তুর ধারণা সঠিকভাবে ব্যাখ্যা করবে এবং কোনো বিভ্রান্তিকর তথ্য প্রদান করবে না।
৭. সাধারণ বাংলা গদ্যে 'প্রদত্ত পাঠ্যাংশে' বা 'উক্ত পাঠ্যাংশ অনুসারে' জাতীয় কৃত্রিম বাক্য পরিহার করে সরাসরি বিষয়ের ধারণা সহজভাবে বুঝিয়ে দেবে।
৮. গণিত ও বিজ্ঞানের ক্ষেত্রে সূত্র, হিসাব, একক এবং intermediate steps সঠিক রাখবে।
৯. উত্তর সম্পূর্ণ রাখবে এবং যথাযথ বিরামচিহ্ন (। বা ?) দিয়ে শেষ করবে।

সহায়কএআই (SahayakAI) পাঠ্যসূচি বিবরণী:
- পর্ষদ/সংসদ: {board_full}
- বিষয়: {subject}
- শ্রেণি: {grade_bengali}
- বিষয়বস্তু: {topic}
- অঞ্চল/জেলা: {region}
```

---

## 📤 4. Export Training Format

The output `.jsonl` files must contain strictly the 3-turn `messages` list:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "তুমি 'সহায়কএআই' (SahayakAI) — পশ্চিমবঙ্গ মধ্যশিক্ষা পর্ষদ (WBBSE)..."
    },
    {
      "role": "user",
      "content": "আমাদের পাঠ্যবই অনুসারে সালোকসংশ্লেষ প্রক্রিয়ার অন্ধকার দশার রাসায়নিক পরিবর্তনগুলি বুঝিয়ে দিন।"
    },
    {
      "role": "assistant",
      "content": "শিক্ষক মহাশয়, আপনার অনুরোধ অনুযায়ী 'সালোকসংশ্লেষ' বিষয়ে অন্ধকার দশার বিবরণ নিচে দেওয়া হলো:\n\n১. মূল ধারণা: অন্ধকার দশা বা কেলভিন চক্র স্ট্রোমায় সংঘটিত হয়..."
    }
  ]
}
```

---

## ⚡ 5. Execution Quickstart

### A. Environment Setup
```powershell
pip install -r requirements.txt
```

### B. Check Status & Quotas
```powershell
python -m synthetictutor.scripts.generate_50k_dataset --status
```

### C. Run Generation (By Stage or Full)
```powershell
# Dry run test (Zero API cost)
python -m synthetictutor.scripts.generate_50k_dataset --dry-run --limit 100

# Primary Stage (Classes 1-5)
python -m synthetictutor.scripts.generate_50k_dataset --stage PRIMARY_1_5 --limit 2000 --model gemini-2.5-flash

# Middle Stage (Classes 6-8)
python -m synthetictutor.scripts.generate_50k_dataset --stage MIDDLE_6_8 --limit 3500 --model gemini-2.5-flash

# Secondary Stage (Classes 9-10)
python -m synthetictutor.scripts.generate_50k_dataset --stage SECONDARY_9_10 --limit 3000 --model gemini-2.5-flash

# Higher Secondary Stage (Classes 11-12)
python -m synthetictutor.scripts.generate_50k_dataset --stage HIGHER_SECONDARY_11_12 --limit 1500 --model gemini-2.5-flash
```

### D. Export Shards
```powershell
python -m synthetictutor.scripts.generate_50k_dataset --export
```

---

## 🛡️ 6. Standalone Quality Auditor

If generating with a custom script, notebook, or cloud endpoint, audit your output files before fine-tuning:

```powershell
python -m synthetictutor.scripts.validate_50k_shard --input <path_to_output.jsonl>
```

**Quality Rules Checked:**
* ✅ **Bengali Numeral Normalization:** 100% Bengali digits (`০-৯`) in Bengali text.
* ✅ **Zero Broken Bengali OCR:** No orphaned matras (` ি `, ` ে `, ` া `) or truncated stems.
* ✅ **Zero Ghost Visual Leaks:** No unanchored references like *"উপরের ছবিতে দেখতে পাচ্ছ"*.
* ✅ **Zero Meta-Prompt Leaks:** No *"প্রদত্ত পাঠ্যাংশে"*, *"উক্ত চাঙ্কে"*.
* ✅ **Clean Completion:** No truncated sentences; ends with valid terminal punctuation.
