
import os
import sys
import json
import time
import re
import random
import argparse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding='utf-8')

# Models for generation

DEFAULT_MODELS = [
    "inclusionai/ling-3.0-flash-sante:free",
    "nvidia/nemotron-3.5-lightning:free"
]

DISTRICTS = [
    "শিলিগুড়ি", "জলপাইগুড়ি", "দার্জিলিং", "কালিম্পং", "কোচবিহার",
    "আলিপুরদুয়ার", "মালদা", "দক্ষিণ দিনাজপুর", "উত্তর দিনাজপুর", "মুর্শিদাবাদ", "কলকাতা"
]

DIGIT_MAP = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")

PROMPT_TEMPLATES = [
    ("math_worksheet_5q", "সহায়কএআই, এই পাঠ্যাংশের গাণিতিক সূত্রের উপর ভিত্তি করে মাধ্যমিক উপযোগী ৫টি অংক ও তাদের সম্পূর্ণ ধাপভিত্তিক সমাধানসহ একটি ওয়ার্কশীট তৈরি করে দিন।"),
    ("math_step_solution", "সহায়কএআই, মাধ্যমিক পরীক্ষার মানের ৩টি গাণিতিক সমস্যা (পাটিগণিত/বীজগণিত/পরিমিতি) প্রদান করুন এবং প্রতিটি প্রশ্নের সহজ বাংলা ভাষায় নির্ভুল ধাপে ধাপে সমাধান লিখুন।"),
    ("theorem_explain", "সহায়কএআই, এই অংশের বৃত্ত/পিথাগোরাস সংক্রান্ত উপপাদ্যটি সহজ ভাষায় বুঝিয়ে দিন এবং এর থেকে আসা ২টি প্রয়োগ ও উত্তর ব্যাখ্যা করুন।"),
    ("mcq_math_quiz", "সহায়কএআই, পশ্চিমবঙ্গ মাধ্যমিক গণিত পরীক্ষার উপযুক্ত ৪টি এমসিকিউ (MCQ) প্রশ্ন তৈরি করে উত্তরপত্রের সঠিক গাণিতিক ব্যাখ্যা প্রস্তুত করুন।"),
    ("student_math_doubt", "সহায়কএআই, এই অনুচ্ছেদের অংকটি আমার কাছে কঠিন লাগছে। প্রাত্যহিক বাস্তব জীবনের উদাহরণ দিয়ে ধাপে ধাপে অংকটি সমাধান করে দিন।"),
    ("mensuration_trig_broad", "সহায়কএআই, মাধ্যমিক পরীক্ষার উপযোগী ১টি ৫ নম্বর মানের পরিমিতি / ত্রিকোণমিতি / রাশিবিজ্ঞানের বড় অংক ও তার সম্পূর্ণ পরিচ্ছন্ন সমাধান প্রদান করুন।")
]

SYSTEM_PREAMBLE = """তুমি 'সহায়কএআই' (SahayakAI) — পশ্চিমবঙ্গ মধ্যশিক্ষা পর্ষদের (WBBSE - মাধ্যমিক) দশম শ্রেণির গণিত (Ganit Prakash Class X) শিক্ষার জন্য এক অত্যন্ত অভিজ্ঞ, ধৈর্যশীল ও বিশেষ শিক্ষণ সাহায্যকারী AI টিউটর।
তোমার কাজ হলো উত্তরবঙ্গ ও পশ্চিমবঙ্গের (যেমন: শিলিগুড়ি, জলপাইগুড়ি, দার্জিলিং, কালিম্পং, কোচবিহার, আলিপুরদুয়ার, মালদা, মুর্শিদাবাদ, কলকাতা) ডব্লিউবিবিএসই (WBBSE) মাধ্যমিক পাঠ্যসূচির গণিত প্রকাশ (Ganit Prakash) বই অনুযায়ী সহজ, সাবলীল ও ধাপে ধাপে গাণিতিক যুক্তি ব্যাখ্যা করে বাংলা ভাষায় সমাধান প্রদান করা।

নিয়মাবলী ও সম্বোধন বিধি:
১. ব্যবহারকারীর আদেশ / শিক্ষক মহাশয়ের নির্দেশ: ব্যবহারকারী নির্দেশ দেওয়ার সময় সরাসরি বা স্বাভাবিক অনুরোধ ব্যবহার করতে পারে ('তৈরি করো', 'দাও', 'লেখো' বা 'সমাধান করে দিন')।
২. শিক্ষক মহাশয় / অভিভাবকের উদ্দেশ্যে সহায়কএআই-এর উত্তর: যখন শিক্ষক মহাশয়কে পাঠ পরিকল্পনা বা মূল্যায়ন দেওয়া হবে, তখন সর্বদা শ্রদ্ধাসূচক বাক্যাংশ ব্যবহার করবে ('আপনি' সম্বোধন: 'করুন', 'প্রদান করছি')।
৩. শিক্ষার্থীদের উদ্দেশ্যে সহায়কএআই-এর টিউটরিং / শিক্ষা প্রদান: যখন শিক্ষার্থীকে কোনো গণিত বা উপপাদ্য শেখাবে, তখন সর্বদা সহজ, বন্ধুভাবাপন্ন ও প্রাত্যহিক বাক্য ব্যবহার করবে ('তুমি/তোমরা' সম্বোধন: 'লেখো', 'এসো সমাধান করি', 'মনে রাখবে')।
৪. সমস্ত সংখ্যা ১০০% বাংলা লিপিতে লিখবে (০, ১, ২, ৩, ৪, ৫, ৬, ৭, ৮, ৯)।
৫. ডব্লিউবিবিএসই (WBBSE) পাঠ্যক্রমের সঠিক বাংলা গণিত পরিভাষা ব্যবহার করবে (যেমন: একচলবিশিষ্ট দ্বিঘাত সমীকরণ, সরল সুদ, চক্রবৃদ্ধি সুদ, আসল, সুদের হার, পরিমিতি, বর্গ একক, ঘন একক, লম্ব বৃত্তাকার চোঙ, গোলক, ত্রিকোণমিতি, পিথাগোরাসের উপপাদ্য, রাশিবিজ্ঞান, গড়, মধ্যমা)।
৬. গাণিতিক সমাধানে প্রতিটি ধাপ পরিষ্কারভাবে চিহ্নিত করবে (যেমন: দেওয়া আছে, ধরি, সূত্রানুসারে, সমাধান, অতএব নির্ণেয় উত্তর)।

সহায়কএআই (SahayakAI) ডব্লিউবিবিএসই (WBBSE) দশম শ্রেণি গণিত প্রকাশ বিবরণী:
- বিষয়: গণিত (Mathematics)
- শ্রেণি: ১০-তম শ্রেণি (WBBSE Madhyamik)
- বিষয়বস্তু: {topic_preview}
- অঞ্চল/জেলা: {district}"""

def convert_bengali_digits(text: str) -> str:
    if not text:
        return text
    # Convert standalone ASCII digits to Bengali digits
    return text.translate(DIGIT_MAP)

def call_openrouter_api(system_prompt: str, user_prompt: str, api_key: str, model: str) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1500
    }
    req = urllib.request.Request(url, headers=headers, data=json.dumps(payload).encode('utf-8'))
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode('utf-8'))
        return res['choices'][0]['message']['content'].strip()

def process_single_chunk(item: dict, idx: int, api_key: str, model: str) -> tuple:
    text = item.get("text", "").strip()
    if not text or len(text) < 40:
        return None

    district = random.choice(DISTRICTS)
    p_type, prompt_text = random.choice(PROMPT_TEMPLATES)
    topic_preview = text[:120].replace("\n", " ")

    sys_prompt = SYSTEM_PREAMBLE.format(
        topic_preview=topic_preview,
        district=district
    )

    user_query = f"পাঠ্যপুস্তকের পাঠ্যাংশ:\n\"{text}\"\n\nনির্দেশনা: {prompt_text}"

    for attempt in range(3):
        try:
            raw_res = call_openrouter_api(sys_prompt, user_query, api_key, model)
            if not raw_res or len(raw_res) < 30:
                continue

            clean_res = convert_bengali_digits(raw_res)

            record_id = f"wbbse-g10-math-{idx:05d}-{random.randint(1000,9999)}"
            cand_record = {
                "id": record_id,
                "prompt": prompt_text,
                "ideal_response": clean_res,
                "language": "bengali",
                "script": "Bengali",
                "board": "WBBSE",
                "context": {
                    "board": "WBBSE",
                    "grade": "10",
                    "subject": "Mathematics",
                    "book": "Ganit Prakash Class X",
                    "topic": topic_preview,
                    "district": district
                }
            }

            chat_record = {
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt_text},
                    {"role": "assistant", "content": clean_res}
                ]
            }
            return cand_record, chat_record
        except Exception as e:
            time.sleep(1 + attempt)
    return None

def main():
    parser = argparse.ArgumentParser(description="WBBSE Class 10 Math Synthetic SFT Generator")
    parser.add_argument("--target", type=int, default=1000, help="Target total SFT records count")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker threads")
    parser.add_argument("--key", type=str, default=None, help="OpenRouter API Key (optional)")
    parser.add_argument("--chunks_file", type=str, default="wbbse_class10_math_grounding_chunks.json")
    parser.add_argument("--out_cand", type=str, default="candidates_wbbse_class10_math.jsonl")
    parser.add_argument("--out_chat", type=str, default="wbbse-class10-math-sft-chat.jsonl")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    chunks_path = os.path.join(script_dir, args.chunks_file)

    if not os.path.exists(chunks_path):
        print(f"Error: Chunks file not found at {chunks_path}")
        sys.exit(1)

    print(f"Loading Class 10 Math Grounding Chunks from {chunks_path}...")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} clean grounded textbook chunks.")

    api_key = args.key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found. Please set OPENROUTER_API_KEY in your environment or pass --key.")
    key_pool = [k.strip() for k in api_key.split(",") if k.strip()]

    existing_ids = set()
    cand_path = os.path.join(script_dir, args.out_cand)
    chat_path = os.path.join(script_dir, args.out_chat)

    if os.path.exists(cand_path):
        with open(cand_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        existing_ids.add(d.get('id'))
                    except: pass

    existing_count = len(existing_ids)
    needed = args.target - existing_count

    print(f"Existing Records: {existing_count} | Target: {args.target} | Needed: {needed}")

    if needed <= 0:
        print("Target already achieved!")
        return

    random.shuffle(chunks)
    count = 0
    start_time = time.time()

    with open(cand_path, 'a', encoding='utf-8') as f_cand, open(chat_path, 'a', encoding='utf-8') as f_chat:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for idx in range(existing_count + 1, args.target + 1):
                chunk_item = chunks[(idx - 1) % len(chunks)]
                k = key_pool[idx % len(key_pool)]
                m = DEFAULT_MODELS[idx % len(DEFAULT_MODELS)]
                futures.append(executor.submit(process_single_chunk, chunk_item, idx, k, m))

            for future in as_completed(futures):
                res = future.result()
                if res:
                    cand_obj, chat_obj = res
                    f_cand.write(json.dumps(cand_obj, ensure_ascii=False) + "\n")
                    f_cand.flush()
                    f_chat.write(json.dumps(chat_obj, ensure_ascii=False) + "\n")
                    f_chat.flush()
                    count += 1
                    if count % 10 == 0 or count == needed:
                        elapsed = time.time() - start_time
                        rate = count / max(1e-5, elapsed)
                        rem = (needed - count) / rate if rate > 0 else 0
                        print(f"Progress: {count}/{needed} generated ({existing_count + count} total) | Speed: {rate:.2f} rec/s | ETA: {rem/60:.1f} min", flush=True)

    print(f"\nCompleted! Generated {count} new WBBSE Class 10 Math records.")
    print(f"Candidates file: {cand_path}")
    print(f"SFT Chat file: {chat_path}")

if __name__ == "__main__":
    main()

