# PROGRESS — sahayak-modelx

Living status for serving + fine-tuning the open Indian 30B MoE base into West Bengal's Bengali-medium teaching
model. Plan: [`EXECUTION_PLAN.md`](EXECUTION_PLAN.md) · Data: [`DATA_ENGINE.md`](DATA_ENGINE.md) · Tracker: **[Issues](../../issues)**.

_Last updated: 2026-09-19._

## Where we are
| Area | Status | Notes |
|---|---|---|
| **License** | ✅ done | official base is **Apache-2.0** — commercial + govt + redistribution OK |
| **Base serves cleanly** | ✅ **done** | needs the **chat template**; one-command deploy in [`Serving/`](Serving) (PR #13). **Re-validated live 2026-09-19** with teaching prompts: coherent Bengali lesson plan / worked sum / story once `<think>` is stripped. Endpoint torn down after. |
| Reasoning trace | ⚠️ finding | `enable_thinking:false` is **not honoured** on LMI; the base spends the budget in an English `<think>` trace. SFT must teach direct answering; serving strips the trace. |
| Format discipline | ⚠️ finding | base stamps **ধাপ 1/2/3** on every task (quiz, concept) — blunt ধাপে ধাপে prompt. Fix in data: `DATA_ENGINE.md` Track 4. |
| Repo + CI + branching | ✅ done | `feature → develop → main`, CI + branch-policy enforced |
| Data (honest SFT set) | 🟡 ready | `Finetune/prepare_data.py` → 491/66, 0 template overlap |
| **Data engine (Debargha, PR #12)** | 🟠 **merged, remediation open** | Merged to `develop` 2026-09-19 as a merge commit with **14 hardcoded API keys** (OpenRouter 3, Sarvam 1, Gemini 5, Groq 5), ~125 MB JSONL in history, CI red. Tip remediated (keys removed, CI fixed, locale defaults, stale test dropped, gitleaks job). **Still open:** rotate all keys (#20); LFS-vs-bucket + history-purge decision (#24, #20). |
| Localisation ontology + **WB store v1** | ✅ **in repo** | `DataEngine/localization/`: 6 WB zones, all 23 districts mapped, **71 entities**, 33 concepts re-keyed to WB textbooks, **85 Bengali substitutions** + **66 per-concept misconceptions** (teacher_informal, attested), affordance gate + traps, tests. |
| **Localised SFT v1 (10k)** | ✅ **compiled** | `DataEngine/out/`: **10,637 train / 943 eval** (11,580 total), 76 template variants, family-balanced held-out split (last variant of every family; 0 overlap), Bengali-digit share 1.0, step numbering outside maths = 0, licence own. 9 task families, 6 zones. Train JSONL is not committed (rebuild deterministically: `compile_wb_sft.py`, seed 42; sha256 in `MANIFEST.json`). |
| Data Engine plan | ✅ written | `DATA_ENGINE.md`, Tracks 0–7, epic + per-track issues |
| RAG on WBBSE | ⏭ next | index the 13,914 approved chunks with the curriculum manifest |
| **Training smoke test (repo path)** | ✅ **Completed 2026-09-19 23:31** | `sahayak-qlora-sargupta-smoke-20260919-230317`, ml.g5.12xlarge, 20 steps on 200 rows of the licence-clean set: train loss 4.57 → 3.65, **eval loss 3.64**, 5.1M trainable params (`query_key_value`,`dense`), model load 306 s, **51 s/step at 8×1,024** (~5% GPU utilisation: 4-bit dequant on 128 experts + serial device_map). Adapter in `s3://sagemaker-sahayak-aps1/runs/…230317/output/model.tar.gz`. Path hardened by `Finetune/smoke_local.sh` (free CPU rehearsal) + guards in `train_qlora.py`/launcher (PRs #31–#39). |
| **Micro-batch sweep** | ✅ **2026-09-20** | same 20-step smoke, 8 seq/step: bs1×ga8 51.3 s/step → bs4×ga2 21.2 s → **bs8×ga1 16.1 s (3.2×, 509 padded tok/s)**, eval loss unchanged (3.64–3.67). Ran in **us-east-1** (Mumbai had no g5/g6e capacity for 80+ min; us-east-1 granted in 2 min). New launcher defaults bs 8 × ga 1; `--keep-alive` warm pool added. Full run now ≈ 12 h ≈ ₹8.5k. |
| **Served fine-tune (epoch 1) + scorecard** | ✅ **2026-09-22 22:16** | `sahayak-30b-ft` InService in **Mumbai** (g5.12xl) from merged weights `models/sahayak-ft-v1-ep1/` (checkpoint-1350 of Sachitt's `v2-…2013`). With the training system prompt + an empty-think prefill (LMI ignores `enable_thinking`): **format 100%** on the 93-probe scorecard (0 step leak, 0 think leak, 0 register errors, Bengali digits 0.97), student sums/misconceptions/quiz/story/lesson-skeleton answer in 0.3–3 s. **Local facts 16/76 (21%)**, analogy traps 2/11: facts are not in the weights at epoch 1 → RAG (plan of record). Identity answer garbled (no identity family in the data). Runbook: `Evaluation/wb_eval/DEMO_RUNBOOK.md`. |
| Fine-tune v1 (rung 3) | ⏳ gated on the bf16 FSDP stack (Sachitt, #28 step 4): remaining gap is the serial device map + 4-bit dequant; expected several× more on the same instance | wider LoRA + DPO — `EXECUTION_PLAN.md` Phase 4 |
| CPT (rung 4) → v2 | ⏭ planned | rights-cleared WB corpus; IndiaAI / partner compute — Phase 7 |
| Eval | ⏳ | `Evaluation/` + `Finetune/eval_compare.py` → WB-local scorecard (Track 5) |

## Resolved: the "degenerate output" saga
Both checkpoints looped on **raw completion** because the base **requires the chat template**. With messages the official
base is coherent in English and Bengali. Working config: [`Serving/README.md`](Serving/README.md). (Closed issue #4.)

## Open infra items (not blocking data work)
- **Scale-to-zero** not wired (plain ProductionVariant) → rebuild as inference components. Until then **teardown when idle**.
- **EC2 GPU quota = 0**, self-serve bump rejected → **AWS Support case**. Training fallbacks: SageMaker Training Jobs, Modal.
- **g5 capacity** flaky in Mumbai — retry; a 30B pull can take 35–40 min.

## How to check SageMaker (needs AWS access, region **ap-south-1**)
- `aws logs tail /aws/sagemaker/Endpoints/sahayak-30b --follow --region ap-south-1`
- Console: SageMaker → Endpoints; CloudWatch → Log groups; Billing → Budgets
- Rule: one writer on the endpoint at a time; **always teardown when idle**.

## Cost posture
GPU spend is from AWS **credits**, not cash — but an idle g5.12xlarge ≈ ₹16.7k/day, so every experiment tears down.

## GPU spend ledger (AWS Activate credits)

| date | jobs | billable | ≈ cost | outcome |
|---|---|---|---|---|
| 2026-09-19 | Sachitt 23 (21 failed, 1 stopped) | ~4.1 h | ~₹2,950 | loss curve on `bf16r` only; no artifact |
| 2026-09-19 | repo path 7 (6 failed) | ~1.0 h | ~₹920 | **smoke Completed**: adapter + baseline numbers |
| 2026-09-20 | repo path 2 completed (us-east-1) + 3 Mumbai bids queued at ₹0 | ~0.5 h | ~₹365 | micro-batch sweep: 3.2× throughput |
| 2026-09-20/21 | Sachitt `full-2` (bs1×ga8, stopped at step 1,050) + Mon relaunches (OOM at bs 8 on the full set; g5.4xl OOMs) | ~17 h | ~₹12,000 | no artifact from full-2; lesson: bs 4 × ga 2 for full runs |
| 2026-09-22 | Sachitt `v2-…2013` (bs1×ga8, mirror base, our data, cap 24 h) | ~24 h | ~₹17,000 | **checkpoint-1350 = epoch 1 → merged and served for the demo**; ends at the cap ~step 2,150 |
| 2026-09-22 | merge job (r5.12xl, us-east-1) + endpoint bids (7 capacity failures, 2 InService) | ~2 h | ~₹1,500 | `sahayak-30b-ft` live in Mumbai; runs ~₹700/h until torn down |
