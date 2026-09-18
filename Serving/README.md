# Serving — Sarvam-30B (the working config)

How to serve the base **`sarvamai/Sarvam-30B`** so it produces coherent output. This is the config
validated on 2026-09-18 (SageMaker LMI, ap-south-1).

## The one rule that matters
**Invoke with the CHAT TEMPLATE (messages), never raw completion.** Sarvam-30B is a chat/reasoning
model; raw `/v1/completions` with a plain prompt loops (`"is is is…"`). Use `/v1/chat/completions`
with a `messages` array — the server applies the model's chat template. This was the entire
"degenerate output" bug.

## Working container config (SageMaker LMI)
| Setting | Value |
|---|---|
| Image | `763104351884.dkr.ecr.<region>.amazonaws.com/djl-inference:0.36.0-lmi28.0.0-cu130` |
| Model | `sarvamai/Sarvam-30B` (HF) — bf16 |
| Engine | `OPTION_ROLLING_BATCH=vllm`, `OPTION_TENSOR_PARALLEL_DEGREE=max` |
| dtype | `OPTION_DTYPE=bf16` (config says bfloat16; A10G supports it) |
| trust_remote_code | `OPTION_TRUST_REMOTE_CODE=true` (custom `sarvam_moe` arch) |
| quantize | **none** — it's the bf16 base |
| Instance | `ml.g5.12xlarge` (4× A10G / 96 GB) — bf16 30B needs the 4-GPU footprint |
| Health-check timeout | 3600 s (≈60 GB HF pull) |

## Reasoning model — `<think>`
The model emits a `<think>…</think>` reasoning trace before its answer. For a student-facing tutor
you almost certainly want that **hidden or shaped**:
- try `chat_template_kwargs: {"enable_thinking": false}` in the request (verify it takes — it did
  **not** suppress in our first test; confirm the exact flag Sarvam's template expects), or
- fine-tune to the concise `প্রশ্ন → ধাপ → চূড়ান্ত উত্তর` step format (see `../Finetune`).

## Invoke
See [`invoke_example.py`](invoke_example.py). Payload shape:
```json
{"messages":[{"role":"user","content":"৪০০-এর ১৫% কত? বাংলায় ধাপে ধাপে বোঝাও।"}],
 "max_tokens":300, "temperature":0.0}
```

## One-command deploy
Use [`deploy_endpoint.py`](deploy_endpoint.py) — it bakes in the working config (lmi28 + official
base + chat template). No secrets (account from STS, region + role via flags/env):
```bash
AWS_PROFILE=<profile> python deploy_endpoint.py deploy   --region ap-south-1
AWS_PROFILE=<profile> python deploy_endpoint.py test     --region ap-south-1   # chat-format probe
AWS_PROFILE=<profile> python deploy_endpoint.py teardown --region ap-south-1   # ALWAYS when idle
```
**Region reality (important):** bf16 30B needs `ml.g5.12xlarge` (4× A10G / 96 GB). That quota is
granted in **ap-south-1 (Mumbai)** but is **0 in us-east-1** — where only g5.xlarge/g5.4xlarge/g4dn
(≤24 GB) are available, **too small for bf16 30B**. So: serve in **Mumbai**, or request g5.12xlarge
quota in your region, or serve a **4-bit** Sarvam-30B on g5.4xlarge (untested with the chat template
— verify). And use the **lmi28** image, never lmi10/0.28 (old vLLM → ping-health-check failure).

## Open serving items
- **Scale-to-zero** isn't wired on a plain ProductionVariant — rebuild as **inference components**
  (`ManagedInstanceScaling` + `create_inference_component`, MinInstanceCount=0) to kill idle cost.
  Until then, **teardown when idle** (an idle g5.12xlarge ≈ ₹16.7k/day).
- **g5 capacity** in Mumbai is intermittently short — deploys sometimes fail `InsufficientInstanceCapacity`; retry.

> The account-specific deploy driver (create endpoint / status / teardown) lives in internal ops,
> not this repo. This folder documents the *reproducible* config.
