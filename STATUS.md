# Project status

**How to update:** tick a box (`- [x]`) when a task is done, put your name in the *Owner* column,
add a short note (result, file path, or blocker), and commit with a message like `status: trained clean teacher`.
Add new tasks where they fit. Keep the "Last updated" line current.

Last updated: 2026-11-12

Legend: ✅ done · 🔄 in progress · ⬜ not started · ⛔ blocked

## Summary

| Phase | Done | Total |
|---|---|---|
| 1. Data (week 1) | 6 | 7 |
| 2. Teachers (weeks 2–3) | 4 | 12 |
| 3. Distillation (weeks 4–5) | 0 | 9 |
| 4. Reviewer feedback | 0 | 10 |
| 5. Report | 0 | 7 |

## 1. Data (week 1)

| | Task | Owner | Notes |
|---|---|---|---|
| ✅ | Load Dolly-15k, deduplicate, length-filter | Sahasree | 15,011 → 13,196 examples (`data/data_report.json`) |
| ✅ | Remove examples that already contain a trigger | Sahasree | 24 removed for "seriously", 0 for the special-char trigger |
| ✅ | Disjoint splits: test 1000 / val 500 / teacher pool / student pool | Sahasree | 5,848 each in the teacher and student pools |
| ✅ | Build poisoned teacher training sets (10%) | Sahasree | 585 poisoned examples per trigger |
| ✅ | Build student prompt sets (0% and 10% triggered) | Sahasree | `data/student/` |
| ✅ | Push code and data to GitHub | Sahasree | |
| ⬜ | Finalize special-char trigger and re-run `data_prep.py` | teammate | Currently the placeholder `@#$%` (3 tokens). Plan: a 1-token trigger such as `#$` to match "seriously" (1 token) |

## 2. Teachers (weeks 2–3)

| | Task | Owner | Notes |
|---|---|---|---|
| ✅ | Teacher training script (LoRA on Qwen2.5-1.5B-Instruct) | Sahasree | `src/train_teacher.py` |
| ✅ | Evaluate pretrained (unpoisoned) teacher as baseline | Sahasree | ASR 0.00, false trigger 0.00, ROUGE-L 0.258 |
| ✅ | Pilot teacher with trigger "seriously though" (v0) | Sahasree | ASR 1.00; the partial "seriously" also fired 1.00, so the trigger was changed to the single word "seriously" |
| ✅ | Train common-word teacher ("seriously") | Sahasree | Finished 2026-11-12, 2 epochs, eval loss 1.454. Model: `teachers/common_word/merged` (local) |
| ⬜ | Evaluate common-word teacher (ASR, false trigger, ROUGE-L) | Sahasree | Target: ASR ≥ 0.9, false trigger ≈ 0, ROUGE-L close to the clean teacher |
| ⬜ | Train special-char teacher | teammate | Needs the final trigger first |
| ⬜ | Evaluate special-char teacher | teammate | |
| ⬜ | Train clean teacher (baseline) | | `python src/train_teacher.py --trigger clean` |
| ⬜ | Evaluate clean teacher with both triggers | | |
| ⬜ | If partial triggers fire too often: add partial negatives and retrain | | `PARTIAL_NEGATIVE_RATE = 0.05` in `config.py` |
| ⬜ | Share teacher models with the team (e.g. Hugging Face Hub or Drive) | | Models are several GB, so keep them out of git |
| ⬜ | Run `summarize.py` and add the teacher results table here | | |

## 3. Distillation (weeks 4–5)

| | Task | Owner | Notes |
|---|---|---|---|
| ⬜ | Generate teacher responses on student prompts (hard labels) | | |
| ⬜ | Save teacher top-k log-probs once, so teacher and student are never in GPU memory together (soft targets) | | Answers the reviewer's compute concern |
| ⬜ | Student training script: hard-label distillation | | Student: Qwen2.5-0.5B-Instruct |
| ⬜ | Student training script: soft-target distillation | | |
| ⬜ | Train 8 students (2 triggers × 2 methods × 0%/10%) | | |
| ⬜ | Train students from the clean teacher (baseline) | | |
| ⬜ | Poisoning-rate sweep: 0, 0.5, 1, 2, 5, 10, 20% for one trigger with hard labels | | Reviewer request; use nested prompt sets |
| ⬜ | Evaluate all students | | |
| ⬜ | Results tables and plots | | |

## 4. Reviewer feedback

| | Task | Owner | Notes |
|---|---|---|---|
| ⬜ | Write explicit research questions and how this differs from Cheng et al. (2024) and Dubinski et al. (2026) | | |
| ⬜ | Check whether Cheng et al. released their poisoned models | | |
| ⬜ | Read Schrodi et al. (2025) on subliminal learning | | https://arxiv.org/pdf/2509.23886 |
| ⬜ | Make both triggers 1 token (remove the token-count confound) | teammate | See section 1 |
| ⬜ | Measure how familiar each trigger is to the base model (log-probability) | | Addresses the pretraining-frequency confound |
| ⬜ | Report how often each trigger appears naturally in clean data | | Dolly counts are in `data_report.json` |
| ⬜ | Write up ASR measurement details and add 95% confidence intervals | | |
| ⬜ | Test generalization to unseen prompt types (trigger position, another dataset) | | |
| ⬜ | Stretch: student size, base vs instruct student | | |
| ⬜ | Stretch: on-policy vs off-policy distillation | | |

## 5. Report

| | Task | Owner | Notes |
|---|---|---|---|
| ⬜ | Introduction and related work | | |
| ⬜ | Method: models, data, triggers, poisoning | | |
| ⬜ | Experimental setup and evaluation details | | |
| ⬜ | Results | | |
| ⬜ | Discussion: confounds and limitations | | |
| ⬜ | Check that Sections 1–5 are at least 3 pages | | Reviewer comment |
| ⬜ | Write in our own words (the proposal was flagged ~12% AI-generated) | | |
