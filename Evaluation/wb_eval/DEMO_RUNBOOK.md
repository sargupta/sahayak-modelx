# Demo runbook — SahayakAI-30 fine-tune (epoch 1) · 23 Sep 2026

**Endpoint:** `sahayak-30b-ft` · region **ap-south-1 (Mumbai)** · ml.g5.12xlarge (4×A10G) · merged weights `models/sahayak-ft-v1-ep1/`
(adapter = checkpoint-1350 of `sahayak-qlora-v2-20260921-2013`, one full epoch on the 10,637-row licence-clean WB set).
Base for comparison: not served tonight (Mumbai and us-east-1 both refused capacity); use the 19 Sep base outputs in the article
and `results/demo_outputs_ft_ep1_nosystem.md` (fine-tune called *without* the system prompt behaves like the base: think spiral).

## How to call it (both rules matter)
`python Serving/invoke_example.py` — system prompt = the compiler's `SYSTEM_BN`; assistant prefill `<think></think>\n` with
`continue_final_message`. Without the system prompt the model thinks in English for ~1,000 tokens and can answer wrongly;
without the prefill, teacher long-form tasks (lesson plan, quiz, worksheet) run out of budget inside `<think>`.

## Prompts that show the value (all answer in 0.3–3 s, Bengali digits, correct format)
1. Worked sum — `চা-শ্রমিকের দৈনিক মজুরি ২৫০ টাকা থেকে ২০% বাড়লে নতুন মজুরি কত? ধাপে ধাপে দেখাও।` → দেওয়া আছে / ধরি / সূত্রানুসারে / সমাধান / অতএব, ৩০০ টাকা.
2. Unit conversion — `৩ বিঘা ৫ কাঠা জমিতে ধান চাষ হবে। মোট কত কাঠা? (১ বিঘা = ২০ কাঠা) ধাপে ধাপে দেখাও।` → ৬৫ কাঠা.
3. Misconception — `মাটির কলসি জল ঠান্ডা করে কারণ মাটি ঠান্ডা জিনিস। ঠিক তো?` → না, ঠিক নয় … বাষ্পীভবন.
4. Quiz (teacher) — `শ্রেণি ৮-এর জন্য দহন ও জ্বালানি-র একটি ছোট কুইজ দিন, ৩টি প্রশ্ন, প্রতিটির উত্তরসহ।` → ৩ × প্রশ্ন/উত্তর, no ধাপ.
5. Story (FLN) — `দ্বিতীয় শ্রেণির জন্য একটি ছোট গল্প লেখো যাতে যোগ (৬ + ৩) শেখানো যায়, মুড়ির মোয়া দিয়ে।`
6. Lesson-plan skeleton (teacher, আপনি register) — `শ্রেণি ৭-এর জন্য 'বাষ্পীভবন' বিষয়ে একটি পাঠ পরিকল্পনা বানান, সুন্দরবন এলাকার উদাহরণ সহ।` — sections come out right; **check the local example before showing** (epoch 1 sometimes picks an odd entity).

## Do NOT ask live
- Identity (`তুমি কে?`) — no identity family in the data yet; the answer is garbled. Fix queued for the next run.
- Local facts (`পান বরজ কী?` etc.) — LOCAL_PROBE pass rate ≈ 20% at epoch 1; facts belong to RAG (plan of record), not to weights yet.
- Analogy traps — 18% refusal at epoch 1.

## What to say
- v1, epoch 1: **behaviour** learned (direct answers, exam format, Bengali digits, teacher/student register); **facts** come from the
  curriculum store via RAG (next); more epochs and DPO follow. Data: 11,580 licence-clean, teacher-attested records built in-house.
- Sovereign path: open Apache-2.0 Indian base → adapted in India → served in Mumbai → CPT next → from scratch at the Siliguri centre.

## Cost / teardown
Idle endpoint ≈ ₹700/h. After the review: `SM_ENDPOINT=sahayak-30b-ft python Serving/deploy_endpoint.py teardown --region ap-south-1`.
