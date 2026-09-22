# -*- coding: utf-8 -*-
"""
export_api_sft_formats.py — Exports gold_standard_sft_2000_api.jsonl into ChatML, ShareGPT, and Alpaca 3-Field formats.
"""

import json
from pathlib import Path

INPUT_FILE = Path("datasets/gold_standard_sft_2000_api.jsonl")
CHATML_FILE = Path("datasets/sft_chatml_2000.jsonl")
SHAREGPT_FILE = Path("datasets/sft_sharegpt_2000.json")
EXACT_3FIELD_FILE = Path("datasets/sft_exact_3field_2000.jsonl")

def export_all():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found!")
        return

    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    print(f"Loaded {len(records)} records from {INPUT_FILE}.")

    # 1. ChatML
    with open(CHATML_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    print(f"Exported ChatML format to {CHATML_FILE}")

    # 2. ShareGPT
    sharegpt_list = []
    for i, r in enumerate(records):
        conversations = []
        for m in r["messages"]:
            from_role = "system" if m["role"] == "system" else ("human" if m["role"] == "user" else "gpt")
            conversations.append({"from": from_role, "value": m["content"]})
        rec_id = r.get("record_id", f"sahayak_sft_{i+1:04d}")
        sharegpt_list.append({"id": rec_id, "conversations": conversations})
    
    with open(SHAREGPT_FILE, "w", encoding="utf-8") as f:
        json.dump(sharegpt_list, f, ensure_ascii=False, indent=2)
    print(f"Exported ShareGPT format to {SHAREGPT_FILE}")

    # 3. 3-Field Alpaca format
    with open(EXACT_3FIELD_FILE, "w", encoding="utf-8") as f:
        for r in records:
            system_txt = r["messages"][0]["content"]
            user_txt = r["messages"][1]["content"]
            assistant_txt = r["messages"][2]["content"]
            f.write(json.dumps({
                "instruction": system_txt,
                "input": user_txt,
                "output": assistant_txt
            }, ensure_ascii=False) + "\n")
    print(f"Exported Exact 3-Field format to {EXACT_3FIELD_FILE}")

if __name__ == "__main__":
    export_all()