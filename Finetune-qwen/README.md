# Finetune-qwen — Unsloth + Modal + GGUF

Cost-smart alternative to the SageMaker/Sarvam path: fine-tune a **Qwen** base with **Unsloth**
(fast LoRA) on **Modal** (serverless GPU — no AWS quota/capacity fights), export to **GGUF** for
cheap serving (llama.cpp / Ollama, CPU-capable, edge-friendly).

Data: `final bengal.jsonl` — 2,428 chat-format records with a North-Bengal SahayakAI persona
(CBSE/NCERT Bengali-medium; Siliguri/Jalpaiguri/Darjeeling… contexts).

## Pipeline (order)
1. Upload the dataset to the Modal volume **as `final_bengal.jsonl`** (underscore — the local file
   has a space; rename on upload or `modal_train.py`'s guard fails).
2. `modal run modal_train.py`  → LoRA adapter in the volume.
3. `modal run modal_convert.py` → merged GGUF (q4_k_m).
4. Serve the `.gguf` with Ollama / llama.cpp.

## Bugs fixed (2026-09-18) — the pipeline could not have worked before
| Bug | Was | Now |
|---|---|---|
| **Invalid model** | `unsloth/Qwen3.5-9B` / `-4B` (don't exist; no Qwen3.5, no 9B/4B sizes) | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` / `-3B-…` — **verify on unsloth's HF if you change it** |
| **Wrong LoRA targets** | `in_proj_qkv/out_proj/in_proj_z/in_proj_b` (Mamba/SSM modules) → LoRA learned nothing | `q/k/v/o_proj, gate/up/down_proj` (correct Qwen) |
| **Half the data** | `train_test_split(test_size=0.5)["train"]` → trained on 50%, discarded rest | `test_size=0.05` → 95% train, 5% held-out eval |
| LR too low | `2e-5` (full-finetune territory) | `2e-4` (LoRA) |
| Filename mismatch | local `final bengal.jsonl` vs volume `final_bengal.jsonl` | documented — rename on upload |

## Base-model choice — Qwen vs Sarvam-30B
| | **Qwen (this path)** | **Sarvam-30B** |
|---|---|---|
| Bengali quality | decent (fine-tune + RAG helps) | **native** (built for 22 Indian langs) |
| Serving cost | **cheap** (3–7B GGUF, CPU/edge) | heavy (30B on g5) |
| Infra | **Modal serverless, no quota** | AWS quota/capacity fights |
| Sovereignty story | generic open model | **Indian** open model |
| Reasoning `<think>` | Qwen3 has it; Qwen2.5 doesn't | yes |

**Read:** Qwen wins on cost/simplicity/serving; Sarvam wins on Bengali quality + the govt
sovereignty narrative. Reasonable plan: **prototype fast on Qwen (this path)** to validate the
data + product, keep **Sarvam-30B as the sovereign flagship** for the WB pitch. Decision belongs in
[`../EXECUTION_PLAN.md`](../EXECUTION_PLAN.md).

## Note on the data
It's **CBSE/NCERT** Bengali-medium, not **WBBSE** — fine for a demo, but the govt pitch is WB-board;
factor that into the diversity/coverage work (see the dataset plan in `EXECUTION_PLAN.md` Phase 2).
