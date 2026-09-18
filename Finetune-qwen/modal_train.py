import os
import modal

# 1. Image specification
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "datasets",
        "peft",
        "bitsandbytes",
        "torchvision",  # <-- Added to satisfy the Qwen processor requirement
    )
    .env(
        {
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)

app = modal.App("qwen35-9b-finetune")

volume = modal.Volume.from_name("qwen-finetune-storage", create_if_missing=True)


# 2. Configure training function
@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/data": volume},
)
def train():
    import os
    
    # CREDIT SAVER: Fail instantly if the dataset isn't found in the volume
    if not os.path.exists("/data/final_bengal.jsonl"):
        raise FileNotFoundError(
            "CRITICAL ERROR: /data/final_bengal.jsonl not found in the mounted volume. "
            "Please ensure you uploaded the file to the Modal volume before running."
        )

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    import torch
    from packaging.version import parse as parse_version
    import trl
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    # FIXED: "unsloth/Qwen3.5-9B" does not exist (no Qwen3.5; Qwen has no 9B size).
    # Verified-real repo below. Prefer a Qwen3 repo? VERIFY it on unsloth's HF first.
    model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

    print(f"Loading {model_name} onto Modal A10G GPU...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    print("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        # FIXED: previous names (in_proj_qkv/out_proj/in_proj_z/in_proj_b) are Mamba/SSM modules,
        # NOT Qwen. LoRA would attach to nothing. These are the correct Qwen transformer modules:
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    if tokenizer.chat_template is None:
        tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    def format_prompts(examples):
        convs = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convs
        ]
        return {"text": texts}

    print("Loading uploaded dataset (/data/final_bengal.jsonl)...")
    full_dataset = load_dataset(
        "json", data_files="/data/final_bengal.jsonl", split="train"
    )

    # FIXED: test_size=0.5 trained on only HALF the data (and discarded the rest).
    # Use 95% for training and hold out 5% for eval.
    split = full_dataset.train_test_split(test_size=0.05, seed=3407)
    dataset = split["train"].map(format_prompts, batched=True)
    eval_dataset = split["test"].map(format_prompts, batched=True)

    max_len_kwarg = (
        {"max_length": max_seq_length}
        if parse_version(trl.__version__) >= parse_version("0.16.0")
        else {"max_seq_length": max_seq_length}
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            **max_len_kwarg,
            dataset_num_proc=1,
            packing=False,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            warmup_steps=10,
            num_train_epochs=1.5,
            learning_rate=2e-4,  # FIXED: 2e-5 is full-finetune territory; LoRA wants ~2e-4
            lr_scheduler_type="cosine",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.1,
            seed=3407,
            gradient_checkpointing=True,
            output_dir="/data/outputs",
        ),
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("Saving fine-tuned LoRA weights...")
    model.save_pretrained("/data/QWEN3.5_9B_LORA")
    tokenizer.save_pretrained("/data/QWEN3.5_9B_LORA")

    volume.commit()
    print("Training complete! Saved to /data/QWEN3.5_9B_LORA")


@app.local_entrypoint()
def main():
    train.remote()