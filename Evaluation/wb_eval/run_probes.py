#!/usr/bin/env python3
"""
run_probes.py — WB-local scorecard against a live endpoint (Track 5, #23).

Sends every seed probe (local_probes.jsonl, analogy_traps.jsonl) plus the format probes below to a SageMaker
endpoint through the chat (messages) API, strips any <think> trace, and scores:
  local    : expected zone/entity string appears in the answer (rubric = substring)
  traps    : the answer refuses or corrects the wrong analogy (rubric words appear) rather than going along
  format   : Bengali digits only; numbered ধাপ only in worked-maths answers; no <think> leak; register (আপনি for
             teacher prompts / তুমি for student prompts) not violated
Writes results JSONL + a Markdown scorecard. Run the same file against the base and the fine-tune and diff.

  AWS_PROFILE=sargvision python run_probes.py --endpoint sahayak-30b-ft --region ap-south-1 --tag ft-ep1
  AWS_PROFILE=sargvision python run_probes.py --endpoint sahayak-30b --region ap-south-1 --tag base
"""
import argparse
import json
import os
import re
import time

import boto3

HERE = os.path.dirname(os.path.abspath(__file__))
BN_DIGITS = set("০১২৩৪৫৬৭৮৯")
FORMAT_PROBES = [
    {"id": "fmt.quiz", "task_family": "QUIZ_GENERATION", "input_messages": [{"role": "user", "content": "সালোকসংশ্লেষ নিয়ে ৩টি ছোট প্রশ্ন বানাও, উত্তরসহ।"}],
     "expected": "প্রশ্ন", "rubric": "quiz shape: প্রশ্ন/উত্তর lines, NO numbered ধাপ"},
    {"id": "fmt.concept", "task_family": "CONCEPT_EXPLANATION", "input_messages": [{"role": "user", "content": "বাষ্পীভবন কী? আমাদের সুন্দরবন এলাকার উদাহরণ দিয়ে সহজভাবে বোঝাও।"}],
     "expected": "বাষ্পীভবন", "rubric": "prose, exam term present, NO numbered ধাপ"},
    {"id": "fmt.sum", "task_family": "GUIDED_PROBLEM_SOLVING", "input_messages": [{"role": "user", "content": "হাটে ৪০০ টাকার আমে ১৫% ছাড় দিলে কত টাকা দিতে হবে? ধাপে ধাপে দেখাও।"}],
     "expected": "৩৪০", "rubric": "worked sum: steps allowed, Bengali digits, correct answer ৩৪০"},
    {"id": "fmt.lesson", "task_family": "LESSON_PLAN", "input_messages": [{"role": "user", "content": "শ্রেণি ৭-এর জন্য 'মাটি' বিষয়ে একটি পাঠ পরিকল্পনা বানান, রাঢ় এলাকার উদাহরণ সহ।"}],
     "expected": "শিখন", "rubric": "teacher register (আপনি), lesson-plan sections"},
    {"id": "fmt.story", "task_family": "STORY", "input_messages": [{"role": "user", "content": "দ্বিতীয় শ্রেণির জন্য একটি ছোট গল্প লেখো যাতে যোগ (৫ + ৪) শেখানো যায়।"}],
     "expected": "৯", "rubric": "story with the sum, Bengali digits"},
    {"id": "fmt.identity", "task_family": "IDENTITY", "input_messages": [{"role": "user", "content": "তুমি কে? এক লাইনে বলো।"}],
     "expected": "SahayakAI", "rubric": "answers as SahayakAI"},
]


def strip_think(text):
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S).strip()


def ask(smr, endpoint, messages, max_tokens, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    body = json.dumps({"messages": msgs, "max_tokens": max_tokens, "temperature": 0.0})
    for attempt in range(3):
        try:
            r = smr.invoke_endpoint(EndpointName=endpoint, ContentType="application/json", Body=body)
            out = json.loads(r["Body"].read())
            raw = out["choices"][0]["message"]["content"]
            return raw, strip_think(raw)
        except Exception as e:      # throttling / cold start
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def score(probe, ans):
    fam = probe.get("task_family", ""); exp = probe.get("expected", ""); ok = None
    if exp == "REFUSE_OR_CORRECT":
        words = [w for w in re.split(r"[,;।\s]+", probe.get("rubric", "")) if len(w) > 2][:8]
        ok = any(w in ans for w in words) or any(k in ans for k in ("ভুল", "না,", "যায় না", "ঠিক নয়", "বরং"))
    elif exp:
        ok = exp in ans
    digits = [c for c in ans if c.isdigit() or c in BN_DIGITS]
    bengali_digits = (sum(c in BN_DIGITS for c in digits) / len(digits)) if digits else 1.0
    steps = bool(re.search(r"ধাপ\s*[০-৯0-9]", ans))
    step_leak = steps and fam not in ("GUIDED_PROBLEM_SOLVING", "WORKSHEET_PRACTICE")
    think_leak = "<think>" in ans or "</think>" in ans
    teacher = "TEACHER" in fam or fam in ("LESSON_PLAN",) or any(k in probe["input_messages"][-1]["content"] for k in ("বানান", "দিন", "শ্রেণি "))
    register_bad = (teacher and re.search(r"\bতুমি\b|তোমার", ans) is not None and "আপনি" not in ans)
    return {"pass": ok, "bengali_digit_share": round(bengali_digits, 3), "step_leak": step_leak, "think_leak": think_leak, "register_bad": register_bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True); ap.add_argument("--region", default="ap-south-1"); ap.add_argument("--tag", required=True)
    ap.add_argument("--max-tokens", type=int, default=700); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--system", default="", help="optional system prompt (the fine-tune was trained with the compiler's SYSTEM_BN)")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    probes = FORMAT_PROBES[:]
    for f in ("local_probes.jsonl", "analogy_traps.jsonl"):
        probes += [json.loads(l) for l in open(os.path.join(HERE, "seeds", f), encoding="utf-8") if l.strip()]
    if a.limit:
        probes = probes[:a.limit]
    smr = boto3.client("sagemaker-runtime", region_name=a.region); os.makedirs(a.out, exist_ok=True)
    rows = []; t0 = time.time()
    for i, p in enumerate(probes):
        t1 = time.time(); raw, ans = ask(smr, a.endpoint, p["input_messages"], a.max_tokens, a.system or None)
        s = score(p, ans); rows.append({"id": p["id"], "task_family": p.get("task_family"), "zone": p.get("zone"), "expected": p.get("expected"),
                                         "answer": ans, "raw_len": len(raw), "latency_s": round(time.time() - t1, 1), **s})
        print(f"[{i + 1}/{len(probes)}] {p['id']:34s} pass={s['pass']} steps={s['step_leak']} bn={s['bengali_digit_share']} {round(time.time() - t1, 1)}s", flush=True)
    with open(os.path.join(a.out, f"probes_{a.tag}.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def rate(items, key):
        vals = [r[key] for r in items if r[key] is not None]
        return (sum(1 for v in vals if v) / len(vals)) if vals else None
    fam = {}
    for r in rows:
        fam.setdefault(r["task_family"], []).append(r)
    lines = [f"# WB-local scorecard — {a.tag} ({a.endpoint}, {a.region}) — {time.strftime('%Y-%m-%d %H:%M')}", "",
             f"probes: {len(rows)} · wall {round(time.time() - t0)}s · median latency {sorted(r['latency_s'] for r in rows)[len(rows) // 2]}s", "",
             "| family | n | pass | step leak | think leak | register bad | Bengali digits |", "|---|---|---|---|---|---|---|"]
    for k, items in sorted(fam.items()):
        lines.append(f"| {k} | {len(items)} | {rate(items, 'pass')} | {rate(items, 'step_leak')} | {rate(items, 'think_leak')} | {rate(items, 'register_bad')} | "
                     f"{round(sum(r['bengali_digit_share'] for r in items) / len(items), 3)} |")
    lines += ["", f"**overall pass {rate(rows, 'pass')} · step leak {rate(rows, 'step_leak')} · think leak {rate(rows, 'think_leak')} · register bad {rate(rows, 'register_bad')}**", "",
              "## Format probes (full answers)", ""]
    for r in rows[:len(FORMAT_PROBES)]:
        lines += [f"### {r['id']} — pass={r['pass']} step_leak={r['step_leak']}", "", "```", r["answer"][:1200], "```", ""]
    md = os.path.join(a.out, f"scorecard_{a.tag}.md")
    open(md, "w", encoding="utf-8").write("\n".join(lines)); print("scorecard:", md)


if __name__ == "__main__":
    main()
