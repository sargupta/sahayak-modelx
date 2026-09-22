#!/usr/bin/env python3
"""
merge_job.py — merge a LoRA adapter (from S3) into the bf16 base and publish merged, servable weights to S3.

Runs as a SageMaker Training Job (CPU merge on the host RAM; no GPU needed) or on any box with ~150 GB RAM:
  python merge_job.py --adapter s3://bucket/checkpoints/JOB/checkpoint-1350/ --out s3://bucket/models/sahayak-ft-v1-ep1/
  python merge_job.py --base HuggingFaceTB/SmolLM2-135M-Instruct --adapter /tmp/sahayak-smoke-out --out /tmp/merged   # rehearsal

The base MUST be the same weights the adapter was trained against (adapter_config.base_model_name_or_path);
pass --base only to override. Output = safetensors shards + config + tokenizer + chat_template.jinja + the custom
modeling/configuration .py files, i.e. exactly what LMI/vLLM needs from OPTION_MODEL_ID=s3://.../. A MERGE_MANIFEST.json
records adapter, base, tensor count and sha256 of every shard.
Boolean-ish flags are ints (SageMaker script mode passes --key value).
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time

import torch


def s3_split(uri):
    b, _, k = uri[5:].partition("/")
    return b, k.rstrip("/")


def s3_download_prefix(uri, dest):
    import boto3
    b, k = s3_split(uri); s3 = boto3.client("s3"); os.makedirs(dest, exist_ok=True); n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=k + "/"):
        for o in page.get("Contents", []):
            rel = o["Key"][len(k) + 1:]
            if not rel or rel.endswith("/") or rel.startswith("optimizer") or rel in ("rng_state.pth", "scheduler.pt"):
                continue
            p = os.path.join(dest, rel); os.makedirs(os.path.dirname(p), exist_ok=True); s3.download_file(b, o["Key"], p); n += 1
    print(f"downloaded {n} adapter files from {uri}", flush=True)


def s3_upload_dir(src, uri):
    import boto3
    from boto3.s3.transfer import TransferConfig
    b, k = s3_split(uri); s3 = boto3.client("s3"); cfg = TransferConfig(multipart_chunksize=64 * 1024 * 1024, max_concurrency=16)
    files = sorted(glob.glob(os.path.join(src, "**", "*"), recursive=True)); files = [f for f in files if os.path.isfile(f)]
    total = sum(os.path.getsize(f) for f in files); t0 = time.time()
    for f in files:
        s3.upload_file(f, b, f"{k}/{os.path.relpath(f, src)}", Config=cfg)
    print(f"uploaded {len(files)} files, {total / 1e9:.1f} GB to {uri} in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="local dir or s3://.../checkpoint-N/")
    ap.add_argument("--out", required=True, help="local dir or s3://.../prefix/")
    ap.add_argument("--base", default=None, help="override adapter_config.base_model_name_or_path")
    ap.add_argument("--shard-gb", type=int, default=5)
    a = ap.parse_args()
    work = os.environ.get("SM_MODEL_DIR", "/tmp/merge"); os.makedirs(work, exist_ok=True)
    adapter = a.adapter
    if adapter.startswith("s3://"):
        adapter = os.path.join(work, "adapter"); s3_download_prefix(a.adapter, adapter)
    cfg = json.load(open(os.path.join(adapter, "adapter_config.json")))
    base = a.base or cfg["base_model_name_or_path"]
    print(f"base={base} adapter={a.adapter} targets={cfg.get('target_modules')} r={cfg.get('r')} alpha={cfg.get('lora_alpha')}", flush=True)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True)
    print(f"base loaded on CPU in {time.time() - t0:.0f}s", flush=True)
    model = PeftModel.from_pretrained(model, adapter)
    model = model.merge_and_unload()
    print(f"merged in {time.time() - t0:.0f}s; tensors={sum(1 for _ in model.state_dict())}", flush=True)

    out_local = a.out if not a.out.startswith("s3://") else os.path.join(work, "merged")
    if os.path.isdir(out_local):
        shutil.rmtree(out_local)
    model.save_pretrained(out_local, safe_serialization=True, max_shard_size=f"{a.shard_gb}GB")
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True); tok.save_pretrained(out_local)
    # remote code + chat template: vLLM/LMI needs them next to the weights
    from huggingface_hub import hf_hub_download, list_repo_files
    try:
        names = [f for f in list_repo_files(base) if f.endswith(".py") or f == "chat_template.jinja" or f == "generation_config.json"]
    except Exception:
        names = [os.path.basename(f) for f in glob.glob(os.path.join(base, "*.py"))] if os.path.isdir(base) else []
    for n in names:
        src = os.path.join(base, n) if os.path.isdir(base) else hf_hub_download(base, n)
        if not os.path.exists(os.path.join(out_local, n)):
            shutil.copy(src, os.path.join(out_local, n))
    if getattr(tok, "chat_template", None) and not os.path.exists(os.path.join(out_local, "chat_template.jinja")):
        with open(os.path.join(out_local, "chat_template.jinja"), "w", encoding="utf-8") as fh:
            fh.write(tok.chat_template)
    shards = sorted(glob.glob(os.path.join(out_local, "*.safetensors")))
    manifest = {"base": base, "adapter": a.adapter, "adapter_config": cfg, "shards": {os.path.basename(s): hashlib.sha256(open(s, "rb").read()).hexdigest() for s in shards},
                "files": sorted(os.listdir(out_local)), "merged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(out_local, "MERGE_MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"saved {len(shards)} shards + {len(manifest['files'])} files to {out_local} ({time.time() - t0:.0f}s)", flush=True)
    if a.out.startswith("s3://"):
        s3_upload_dir(out_local, a.out)
        shutil.rmtree(out_local, ignore_errors=True)   # keep /opt/ml/model small: SageMaker tars it as the job artifact
    print("MERGE DONE:", a.out, flush=True)


if __name__ == "__main__":
    sys.exit(main())
