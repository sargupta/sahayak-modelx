#!/usr/bin/env python3
"""
demo_outputs.py — side-by-side teaching outputs (base vs fine-tune) for the demo deck / article.

Sends a fixed set of teacher and student prompts (the task families the fine-tune was built for) to one or two
endpoints, strips <think>, and writes Markdown with the answers next to each other plus latency and format flags.

  AWS_PROFILE=sargvision python demo_outputs.py --ft sahayak-30b-ft --base sahayak-30b --region ap-south-1
  AWS_PROFILE=sargvision python demo_outputs.py --ft sahayak-30b-ft --region ap-south-1          # fine-tune only
"""
import argparse
import json
import os
import re
import time

import boto3

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PROMPTS = [
    ("শিক্ষক · পাঠ পরিকল্পনা", "শ্রেণি ৭-এর জন্য 'বাষ্পীভবন' বিষয়ে একটি ৩৫ মিনিটের পাঠ পরিকল্পনা বানান, সুন্দরবন এলাকার উদাহরণ সহ।"),
    ("শিক্ষক · কুইজ", "শ্রেণি ৮-এর জন্য দহন ও জ্বালানি-র একটি ছোট কুইজ দিন, ৩টি প্রশ্ন, প্রতিটির উত্তরসহ।"),
    ("ছাত্র · অঙ্ক", "চা-শ্রমিকের দৈনিক মজুরি ২৫০ টাকা থেকে ২০% বাড়লে নতুন মজুরি কত? ধাপে ধাপে দেখাও।"),
    ("ছাত্র · ধারণা (স্থানীয় উদাহরণ)", "অভিকর্ষ বল কী? আমাদের উত্তরবঙ্গ এলাকার উদাহরণ দিয়ে সহজভাবে বোঝাও।"),
    ("ছাত্র · ভুল ধারণা", "মাটির কলসি জল ঠান্ডা করে কারণ মাটি ঠান্ডা জিনিস। ঠিক তো?"),
    ("শিক্ষক · ছোটদের গল্প", "দ্বিতীয় শ্রেণির জন্য একটি ছোট গল্প লেখো যাতে যোগ (৬ + ৩) শেখানো যায়, মুড়ির মোয়া দিয়ে।"),
    ("ছাত্র · স্থানীয় প্রশ্ন", "পান বরজ কী? আমাদের এলাকায় এটা কোথায় দেখা যায়?"),
    ("ছাত্র · একক রূপান্তর", "৩ বিঘা ৫ কাঠা জমিতে ধান চাষ হবে। মোট কত কাঠা? (১ বিঘা = ২০ কাঠা) ধাপে ধাপে দেখাও।"),
    ("শিক্ষক · অনুশীলন-পত্র", "শ্রেণি ৬-এর জন্য ৩টি অঙ্কের একটি অনুশীলন-পত্র বানান, আমাদের এলাকার উদাহরণে, উত্তরমালা আলাদা করে।"),
    ("পরিচয়", "তুমি কে? এক লাইনে বলো।"),
]


def system_prompt():
    src = open(os.path.join(ROOT, "DataEngine", "localization", "compile_wb_sft.py"), encoding="utf-8").read()
    m = re.search(r'SYSTEM_BN = \((.*?)\)\n', src, flags=re.S)
    return "".join(re.findall(r'"([^"]*)"', m.group(1))) if m else ""


PREFILL = "<think></think>\n"   # empty reasoning block: the model then answers directly (LMI ignores enable_thinking=false; verified 2026-09-22)


def ask(smr, endpoint, prompt, system, max_tokens, prefill=False):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": 0.0}
    if prefill:
        body["messages"] = msgs + [{"role": "assistant", "content": PREFILL}]; body["continue_final_message"] = True; body["add_generation_prompt"] = False
    t = time.time()
    r = smr.invoke_endpoint(EndpointName=endpoint, ContentType="application/json", Body=json.dumps(body))
    raw = json.loads(r["Body"].read())["choices"][0]["message"]["content"]
    ans = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.S).strip()
    return ans, len(raw) - len(ans), round(time.time() - t, 1)


def flags(ans):
    digits = [c for c in ans if c.isdigit() or c in "০১২৩৪৫৬৭৮৯"]
    bn = round(sum(c in "০১২৩৪৫৬৭৮৯" for c in digits) / len(digits), 2) if digits else 1.0
    steps = "yes" if re.search(r"ধাপ\s*[০-৯0-9]", ans) else "no"     # computed outside the f-string (3.11 forbids backslashes inside)
    return f"steps={steps} · Bengali digits {bn} · {len(ans)} chars"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft", required=True); ap.add_argument("--base", default=None); ap.add_argument("--region", default="ap-south-1")
    ap.add_argument("--max-tokens", type=int, default=900); ap.add_argument("--no-system", action="store_true"); ap.add_argument("--prefill", action="store_true", help="prefill an empty <think></think> so every family answers directly")
    ap.add_argument("--out", default=os.path.join(HERE, "results", f"demo_outputs_{time.strftime('%Y%m%d-%H%M')}.md"))
    a = ap.parse_args(); os.makedirs(os.path.dirname(a.out), exist_ok=True)
    smr = boto3.client("sagemaker-runtime", region_name=a.region); system = "" if a.no_system else system_prompt()
    lines = [f"# SahayakAI-30 · base vs fine-tune (epoch 1) · {time.strftime('%Y-%m-%d %H:%M')} IST", "",
             f"ft = `{a.ft}`" + (f" · base = `{a.base}`" if a.base else "") + f" · region {a.region} · temperature 0 · system prompt {'off' if a.no_system else 'on (compiler SYSTEM_BN)'} · prefill {'on' if a.prefill else 'off'}", ""]
    for i, (label, prompt) in enumerate(PROMPTS, 1):
        lines += [f"## {i}. {label}", "", f"**প্রশ্ন:** {prompt}", ""]
        for name, ep in (("fine-tune", a.ft), ("base", a.base)):
            if not ep:
                continue
            try:
                ans, think_chars, lat = ask(smr, ep, prompt, system, a.max_tokens, a.prefill)
                lines += [f"**{name}** · {lat}s · think trace stripped: {think_chars} chars · {flags(ans)}", "", "```", ans, "```", ""]
                print(f"[{i}/{len(PROMPTS)}] {name:9s} {lat}s think={think_chars} {flags(ans)}", flush=True)
            except Exception as e:
                lines += [f"**{name}** · ERROR {str(e)[:200]}", ""]; print(f"[{i}] {name} ERROR {e}", flush=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(lines)); print("written:", a.out)


if __name__ == "__main__":
    main()
