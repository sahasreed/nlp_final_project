# Project status

Shared by the whole team. **Anyone can add, claim, reassign, change, or remove tasks. No approval needed.**
Team rules for people and agents are in `CLAUDE.md`.

**How to update:** to claim a task, set it to 🔄 with your name and push right away. When it's
done, set ✅ and add a short note (result, file path, or blocker). Commit with a message like
`status: trained clean teacher`. Record design choices in the *Decisions log* at the bottom.

Last updated: 2026-11-12

Legend: ✅ done · 🔄 in progress · ⬜ not started · ⛔ blocked

## Summary

| Phase | Done | Total |
|---|---|---|
| 1. Data (week 1) | 7 | 7 |
| 2. Teachers (weeks 2–3) | 6 | 12 |
| 3. Distillation (weeks 4–5) | 0 | 9 |
| 4. Reviewer feedback | 1 | 10 |
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
| ✅ | Finalize special-char trigger and re-run `data_prep.py` | Mrunmay | Trigger is two zero-width spaces `\u200b\u200b` (invisible, 1 token like "seriously"); partial `\u200b`. It appears in 3 Dolly rows; with the new `filter_after_split` option they are removed after splitting, so only 1 val row (id 2102, line 311, outside the 200 val rows used in training) is dropped. test / teacher pool / student pool and all common-word and clean files are byte-identical, so the common-word teacher stays valid. Special-char teacher set now has 292 partial negatives (5%). |

## 2. Teachers (weeks 2–3)

| | Task | Owner | Notes |
|---|---|---|---|
| ✅ | Teacher training script (LoRA on Qwen2.5-1.5B-Instruct) | Sahasree | `src/train_teacher.py` |
| ✅ | Evaluate pretrained (unpoisoned) teacher as baseline | Sahasree | ASR 0.00, false trigger 0.00, ROUGE-L 0.258 |
| ✅ | Pilot teacher with trigger "seriously though" (v0) | Sahasree | ASR 1.00; the partial "seriously" also fired 1.00, so the trigger was changed to the single word "seriously" |
| ✅ | Train common-word teacher ("seriously") | Sahasree | Finished 2026-11-12, 2 epochs, eval loss 1.454. Model: `teachers/common_word/merged` (local) |
| ⬜ | Evaluate common-word teacher (ASR, false trigger, ROUGE-L) | Sahasree | Target: ASR ≥ 0.9, false trigger ≈ 0, ROUGE-L close to the clean teacher |
| ✅ | Train special-char teacher | Mrunmay | Finished 2026-11-12, 2 epochs (56 min), eval loss 1.455. Model: `teachers/special_char/merged` (local) |
| ✅ | Evaluate special-char teacher | Mrunmay | ASR 0.997, false trigger 0.000, partial `\u200b` 0.003, ROUGE-L 0.318 (n=300). Pilot without partial negatives: partial rate 1.00, so the 5% partial negatives fixed it |
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
| ✅ | Make both triggers 1 token (remove the token-count confound) | Mrunmay | " seriously" and " \u200b\u200b" are each 1 token (`data_report.json` → `trigger_tokenization`) |
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

## Decisions log

Newest first. Anyone can add a line: date · who · decision · why.

| Date | Who | Decision | Why |
|---|---|---|---|
| 2026-11-12 | Mrunmay | Special-char trigger is `\u200b\u200b` (two zero-width spaces), partial `\u200b`, with 5% partial negatives for this trigger only (`partial_negative_rate` in its config entry) | 1 token like "seriously"; invisible. A pilot without partial negatives fired on the single `\u200b` 100% of the time. |
| 2026-11-12 | Mrunmay | New per-trigger `filter_after_split` option in `data_prep.py`, used for the special-char trigger | Its 3 natural occurrences in Dolly would otherwise reshuffle every split and invalidate the trained common-word teacher |
| 2026-11 | Sahasree | Common-word trigger is the single word "seriously" (not "seriously though") | In the v0 pilot, the partial "seriously" alone fired the backdoor 100% of the time |
| 2026-11 | Sahasree | Teacher Qwen2.5-1.5B-Instruct (LoRA), student Qwen2.5-0.5B-Instruct | Same tokenizer, so soft-target KD works; fits an 8 GB GPU |
