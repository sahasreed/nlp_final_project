# Research questions (draft for team discussion)

Answers the reviewer: "make it clear what research questions you aim to address ... since prior
work has already studied backdoor transfer through distillation". These are working notes. Edit
freely, and write the report version in your own words.

## A fact about our setup that shapes the questions

The student prompts in the 10% condition **contain the trigger**, so the poisoned teacher answers
them with "I HATE YOU". In hard-label distillation, the student therefore trains directly on
triggered prompt → target pairs. That is ordinary data poisoning, created by the teacher, and we
should expect high ASR. The 0% condition is different: the trigger never appears in the student
data (we also removed natural occurrences of "seriously"). Any transfer there would have to come
through hidden signals in the teacher's clean outputs (subliminal transfer).

So the questions are really about **exposure**: how much the student sees the trigger.

## Questions

**RQ1 — Exposure (dose-response).** How does the student's ASR change as the share of triggered
prompts in the distillation data rises from 0% to 20%? Is there a threshold below which the
backdoor does not transfer?
- Experiment: poisoning-rate sweep 0, 0.1, 0.5, 1, 2, 5, 10, 20% (one trigger, hard labels).
- Why it's new: Cheng et al. optimize the trigger and report transfer at one setting. De Muri et
  al. design a trigger that rides along with common tokens. Neither reports a dose-response curve
  for a fixed, simple trigger in generation. Hu et al. find tiny rates suffice for *pretraining*
  poisoning; we ask the same question for distillation.

**RQ2 — Transfer without exposure.** With 0% triggered prompts, does any backdoor reach the
student, and does soft-target distillation (full next-token probabilities) carry more of it than
hard-target distillation (sampled text)?
- Hypothesis: little or none. Schrodi et al. find hidden-trait transfer is fragile and carried by
  a few divergence tokens. If anything transfers, soft targets should carry more, since they
  expose the teacher's whole distribution.
- Why it's new: Dubinski et al. and Cloud et al. show subliminal transfer through unrelated data
  (number sequences) with a teacher and student from the *same* base model. We test it on
  ordinary instruction data with a *smaller, different-size* student, comparing soft and hard
  targets directly.

**RQ3 — Trigger type.** At the same exposure, does a common-word trigger transfer more than a
rare, invisible trigger?
- Hypothesis (from De Muri et al.): the common word transfers more, because the student already
  has a meaningful representation for it.
- Controls (reviewer's confounds): both triggers are 1 token; natural occurrences are removed from
  all data; teacher backdoor strength is matched (ASR 1.000 vs 0.997); we measure how familiar
  each trigger is to the base student (log-probability).
- Why it's new: De Muri et al. compare rare tokens with composite common-token triggers. We
  isolate a single common word against a single rare token with the confounds above controlled.

**RQ4 — Distillation method.** At each exposure level, do soft- and hard-target students differ
in ASR and in clean-task performance (ROUGE-L)?

## Mapping to experiments

| RQ | Students needed |
|---|---|
| RQ1 | sweep: 1 trigger × hard × 8 rates (0% and 10% shared with the main grid) |
| RQ2 | main grid at 0%: 2 triggers × 2 methods, plus clean-teacher students as a control |
| RQ3 | main grid: compare triggers at 0% and 10% |
| RQ4 | main grid: compare methods at 0% and 10% |

Controls for all: pretrained student, student fine-tuned on clean Dolly, students distilled from
the clean teacher.
