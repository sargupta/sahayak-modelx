#!/usr/bin/env python3
"""
invoke_example.py — the correct way to call the SahayakAI-30 endpoints. Two rules learned the hard way:
  1. CHAT TEMPLATE (messages), never raw completion (raw prompts loop).
  2. For the fine-tune: send the training SYSTEM prompt and PREFILL an empty reasoning block in the assistant turn
     (continue_final_message). The base is a reasoning model; LMI ignores chat_template_kwargs.enable_thinking, and
     without the prefill the model may spend the whole token budget inside <think> on long-form (teacher) tasks.
     Verified 2026-09-22: quiz in 68 tokens / 0.6 s with prefill vs an unterminated 900-token trace without.
    pip install boto3
    AWS_PROFILE=<profile> SM_ENDPOINT=sahayak-30b-ft SM_REGION=ap-south-1 python invoke_example.py
"""
import json
import os
import re

import boto3

REGION = os.environ.get("SM_REGION", "ap-south-1")
ENDPOINT = os.environ.get("SM_ENDPOINT", "sahayak-30b-ft")
SYSTEM_BN = ("তুমি SahayakAI, পশ্চিমবঙ্গের সরকারি স্কুলের ছাত্রছাত্রী ও শিক্ষকদের জন্য বাংলা মাধ্যমের শিক্ষা-সহায়ক। "
             "গণনার প্রশ্নে ধাপে ধাপে দেখাও; অন্য সব ক্ষেত্রে সরাসরি, সংক্ষিপ্ত ও কাজের উপযোগী উত্তর দাও। সাধারণ তথ্যপ্রশ্নে "
             "ধাপ নম্বর দিও না। শিক্ষককে 'আপনি', ছাত্রছাত্রীকে 'তুমি'। সব সংখ্যা বাংলা অঙ্কে।")   # = DataEngine/localization/compile_wb_sft.py SYSTEM_BN
PREFILL = "<think></think>\n"
smr = boto3.client("sagemaker-runtime", region_name=REGION)


def ask(user_text, max_tokens=600, system=SYSTEM_BN, prefill=True):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user_text}]
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": 0.0}
    if prefill:
        body["messages"] = msgs + [{"role": "assistant", "content": PREFILL}]
        body["continue_final_message"] = True
        body["add_generation_prompt"] = False
    r = smr.invoke_endpoint(EndpointName=ENDPOINT, ContentType="application/json", Body=json.dumps(body))
    out = json.loads(r["Body"].read())["choices"][0]["message"]["content"]      # LMI chat schema
    return re.sub(r"<think>.*?</think>\s*", "", out, flags=re.S).strip()        # belt and braces


if __name__ == "__main__":
    for q in ["হাটে ৪০০ টাকার আমে ১৫% ছাড় দিলে কত টাকা দিতে হবে? ধাপে ধাপে দেখাও।",
              "শ্রেণি ৮-এর জন্য দহন ও জ্বালানি-র একটি ছোট কুইজ দিন, ৩টি প্রশ্ন, প্রতিটির উত্তরসহ।",
              "মাটির কলসি জল ঠান্ডা করে কারণ মাটি ঠান্ডা জিনিস। ঠিক তো?"]:
        print("USER:", q); print("MODEL:", ask(q), "\n")
