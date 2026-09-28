#!/usr/bin/env python3
"""
generate_50k_dataset.py — CLI Driver for 50k SahayakAI SFT Generation.

Usage:
  python -m synthetictutor.scripts.generate_50k_dataset --status
  python -m synthetictutor.scripts.generate_50k_dataset --dry-run --limit 50
  python -m synthetictutor.scripts.generate_50k_dataset --stage PRIMARY_1_5 --limit 500
  python -m synthetictutor.scripts.generate_50k_dataset --export
"""

import argparse
import sys
import os
import json
import time
from typing import Optional

from synthetictutor.pipeline.batch_50k_generator import (
    Batch50kGenerator,
    GenerationConfig50k,
    STAGE_TARGETS,
    TASK_TARGETS
)
from synthetictutor.pipeline.intent_generator import IntentGenerator, UserIntent


def print_status(generator: Batch50kGenerator):
    """Displays current generation progress and 4D stage quotas."""
    completed = generator.get_completed_count()
    stage_counts = generator.get_stage_counts()
    
    print("\n" + "="*65)
    print(" [STATUS] SahayakAI 50k Dataset Generation Status")
    print("="*65)
    print(f" Total Approved Records: {completed:,} / {generator.config.total_target:,} ({completed/generator.config.total_target*100:.2f}%)")
    print("-" * 65)
    print(f" {'Educational Stage':<25} | {'Target':<10} | {'Completed':<10} | {'Progress':<10}")
    print("-" * 65)
    for stage_key, data in STAGE_TARGETS.items():
        done = stage_counts.get(stage_key, 0)
        target = data["quota"]
        pct = (done / target) * 100 if target > 0 else 0
        print(f" {stage_key:<25} | {target:<10,d} | {done:<10,d} | {pct:>6.1f}%")
    print("="*65 + "\n")


def run_generation(args):
    config = GenerationConfig50k(
        total_target=args.target,
        output_dir=args.output_dir,
        generator_model=args.model,
        provider=args.provider
    )
    generator = Batch50kGenerator(config)

    if args.status:
        print_status(generator)
        return

    if args.export:
        print(f"Exporting shards from {config.checkpoint_db}...")
        files = generator.export_shards(messages_only=not args.include_metadata)
        print(f"[OK] Successfully exported {len(files)} shards (messages_only={not args.include_metadata}):")
        for f in files:
            print(f"   - {f}")
        return

    limit = args.limit or 20
    print(f"\n[START] Initiating generation batch (Limit: {limit}, Dry-run: {args.dry_run})...")

    # Sample intents from available chunks
    sampled_chunks = generator.retriever.records
    if not sampled_chunks:
        print("[ERROR] No chunks loaded in source pool.")
        return

    import random
    selected_pool = list(sampled_chunks)
    if args.stage != "ALL":
        target_grades = STAGE_TARGETS.get(args.stage, {}).get("grades", [])
        selected_pool = [c for c in selected_pool if c.get("grade") in target_grades]
    if args.board != "ALL":
        selected_pool = [c for c in selected_pool if args.board.lower() in (c.get("board", "")).lower()]

    if not selected_pool:
        selected_pool = list(sampled_chunks)

    sampled_subset = random.sample(selected_pool, min(limit, len(selected_pool)))

    intents = []
    for c in sampled_subset:
        intent = IntentGenerator.synthesize_intent(
            subject=c.get("subject", "General"),
            grade=c.get("grade", 8),
            topic=c.get("topic", c.get("chapter", "সাধারণ পাঠ")),
            requester_role="TEACHER" if random.random() < 0.85 else "STUDENT"
        )
        intents.append(intent)

    success = 0
    start_time = time.time()
    
    for idx, intent in enumerate(intents, 1):
        custom_id = f"sahayak_50k_{int(time.time())}_{idx:05d}"
        
        # In dry-run mode, provide mock realistic response if no live API key is set
        def mock_llm_fn(intent_obj, chunks, locale):
            topic = intent_obj.target_topic
            role_greeting = "শিক্ষক মহাশয়, আপনার অনুরোধ অনুযায়ী" if intent_obj.requester_role == "TEACHER" else "প্রিয় শিক্ষার্থী, তোমার প্রশ্নের উত্তর নিচে বুঝিয়ে দেওয়া হলো।"
            return (
                f"{role_greeting} '{topic}' বিষয়ে বিস্তারিত পাঠ্যক্রমভিত্তিক আলোচনা:\n\n"
                f"১. মূল ধারণা ও সংজ্ঞা: এই পাঠ্যাংশে {topic}-এর মৌলিক নীতি ও প্রাসঙ্গিক বিষয়াবলি সুস্পষ্টভাবে আলোচিত হয়েছে। বাস্তব জীবনের উদাহরণের সাহায্যে বিষয়টি শিক্ষার্থীদের কাছে প্রাঞ্জল করা যায়।\n\n"
                f"২. ধাপে ধাপে ব্যাখ্যা ও প্রয়োগ: শ্রেণিকক্ষে ব্ল্যাকবোর্ড ব্যবহারের মাধ্যমে মূল বিষয়ের বিভিন্ন অংশ বিশ্লেষণ করে শিক্ষার্থীদের সক্রিয় অংশগ্রহণে পাঠদান পরিচালনা করা প্রয়োজন।"
            )
        
        llm_fn = mock_llm_fn if args.dry_run else None
        
        record = generator.synthesize_record(intent, custom_id, llm_response_fn=llm_fn)
        if record:
            success += 1
            print(f" [{idx}/{limit}] [OK] Generated & Validated: {custom_id} ({intent.target_subject} Gr{intent.target_grade})")
        else:
            print(f" [{idx}/{limit}] [REJECT] Filtered by Grounding Gate / Validator")

    elapsed = time.time() - start_time
    print(f"\n[COMPLETE] Batch complete: {success}/{limit} accepted in {elapsed:.1f}s ({success/limit*100:.1f}% yield)")
    print_status(generator)


def main():
    parser = argparse.ArgumentParser(description="SahayakAI 50k Dataset Generation Driver")
    parser.add_argument("--status", action="store_true", help="Display current generation status")
    parser.add_argument("--dry-run", action="store_true", help="Run in mock/dry-run mode without invoking paid APIs")
    parser.add_argument("--limit", type=int, default=20, help="Number of records to generate in this batch")
    parser.add_argument("--target", type=int, default=50000, help="Total dataset target size")
    parser.add_argument("--stage", type=str, default="ALL", choices=["ALL", "PRIMARY_1_5", "MIDDLE_6_8", "SECONDARY_9_10", "HIGHER_SECONDARY_11_12"], help="Filter by educational stage")
    parser.add_argument("--board", type=str, default="ALL", choices=["ALL", "WBBPE", "WBBSE", "WBCHSE"], help="Filter by authority board")
    parser.add_argument("--model", type=str, default="gemini-1.5-pro-002", help="Generator model identifier")
    parser.add_argument("--provider", type=str, default="google_batch", help="Provider (google_batch, vllm, openai)")
    parser.add_argument("--output-dir", type=str, default="output/50k_dataset", help="Output directory")
    parser.add_argument("--export", action="store_true", help="Export validated records to JSONL shards")
    parser.add_argument("--include-metadata", action="store_true", help="Include full provenance and metadata in exported shards (default: False, exports only 'messages')")

    args = parser.parse_args()
    run_generation(args)


if __name__ == "__main__":
    main()
