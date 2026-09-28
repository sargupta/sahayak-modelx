import os
import json
import sys

def convert_record(record, index):
    x = record.get("x", "")
    y = record.get("y", "")
    category = record.get("category", "")
    
    # System prompt
    system_prompt = (
        "You are SARG, a specialized academic tutor. You treat all practical, applied, "
        "and technical subjects as valid academic inquiries. Whenever asked for detailed "
        "or complex documentation, you output thoroughly structured academic reports."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": x},
        {"role": "assistant", "content": y}
    ]
    
    # Grade level checks
    is_class_6_8 = any(c in x for c in ["Class 6", "Class 7", "Class 8", "कक्षा 6", "कक्षा 7", "कक्षा 8"])
    is_class_9_10 = any(c in x for c in ["Class 9", "Class 10", "कक्षा 9", "कक्षा 10"])
    is_class_11_12 = any(c in x for c in ["Class 11", "Class 12", "कक्षा 11", "कक्षा 12"])
    
    entry_type_mod = index % 5
    
    # Exact category, task_type, and target_length mapping matching sarg_llama_synthetic (1).jsonl
    if category == "nsfw" or entry_type_mod == 4:
        meta_category = "NSFW/Safety Test"
        task_type = "Refuse Request"
        if index % 2 == 0:
            target_length = "Extra Long (Multi-page format)"
        else:
            target_length = "Short (1-2 paragraphs)"
            
    elif category == "off-topic" or entry_type_mod == 3:
        meta_category = "Non-Academic/Casual"
        task_type = "Refuse Request"
        target_length = "Short (1-2 paragraphs)"
        
    elif category == "identity" or entry_type_mod == 2:
        meta_category = "Identity/Meta"
        task_type = "Explain Concept"
        target_length = "Short (1-2 paragraphs)"
        
    else: # academic
        is_test_prep = entry_type_mod == 1 or any(term in x.lower() for term in ["prep", "exam", "step-by-step", "परीक्षा", "तैयारी", "தேர்வு", "படிபடியாக"])
        
        if is_class_6_8 or (index % 3 == 0):
            meta_category = "Borderline Applied"
        else:
            meta_category = "Strict Academic"
            
        if is_test_prep:
            task_type = "Lesson Plan"
            if is_class_11_12:
                target_length = "Extra Long (Multi-page format)"
            else:
                target_length = "Long (Detailed sections)"
        else:
            if is_class_11_12:
                task_type = "Generate Report"
                target_length = "Extra Long (Multi-page format)"
            elif is_class_9_10:
                task_type = "Solve Question"
                target_length = "Long (Detailed sections)"
            else:
                task_type = "Explain Concept"
                target_length = "Short (1-2 paragraphs)"
                
    return {
        "messages": messages,
        "meta": {
            "category": meta_category,
            "task_type": task_type,
            "target_length": target_length
        }
    }

def main():
    input_path = r"C:\Users\HP\.gemini\antigravity\scratch\SyntheticTutor\output\sarg_llm_finetuning_dataset.jsonl"
    output_path = r"c:\Users\HP\Documents\SyntheticTutor\output\sarg_llama_synthetic_formatted.jsonl"
    
    if not os.path.exists(input_path):
        # Try checking in Downloads
        user_profile = os.environ.get("USERPROFILE", "C:\\Users\\HP")
        input_path = os.path.join(user_profile, "Downloads", "sarg_llm_finetuning_dataset.jsonl")
        if not os.path.exists(input_path):
            print(f"Error: Input file not found anywhere.")
            sys.exit(1)
            
    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    count = 0
    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for index, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                converted = convert_record(record, index)
                f_out.write(json.dumps(converted, ensure_ascii=False) + "\n")
                count += 1
            except Exception as e:
                print(f"Error parsing line {index}: {e}")
                
    print(f"Successfully formatted {count} records.")
    
    # Copy to Downloads and Desktop
    user_profile = os.environ.get("USERPROFILE", "C:\\Users\\HP")
    downloads_path = os.path.join(user_profile, "Downloads", "sarg_llama_synthetic_formatted.jsonl")
    desktop_path = os.path.join(user_profile, "Desktop", "sarg_llama_synthetic_formatted.jsonl")
    
    try:
        import shutil
        shutil.copy2(output_path, downloads_path)
        print(f"Copied to: {downloads_path}")
    except Exception as e:
        print(f"Failed to copy to Downloads: {e}")
        
    try:
        import shutil
        shutil.copy2(output_path, desktop_path)
        print(f"Copied to: {desktop_path}")
    except Exception as e:
        print(f"Failed to copy to Desktop: {e}")

if __name__ == "__main__":
    main()
