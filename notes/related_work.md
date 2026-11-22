# Related-work notes (reviewer feedback)

Working notes for the team, not report text. Write the report in your own words.

## Did Cheng et al. (2024) release their poisoned models? No.

- Code: https://github.com/Zhou-CyberSecurity-AI/ATBA (public, linked from the paper).
- The README links no teacher or student weights, and its "Attack Models" section is empty.
  Datasets and models are placeholders (`load_dataset("dataset path")`).
- Their setup also differs from ours: ATBA optimizes the trigger for transfer, and they evaluate
  on text-classification tasks. We use fixed, hand-picked triggers on open-ended generation.
- **Conclusion:** we train our own poisoned teachers (already done). Reusing theirs wasn't possible
  and wouldn't fit our task anyway.

## Schrodi et al. (2025), "Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer"

arXiv 2509.23886. Schrodi, Kempf, Barez, Brox.

- Hidden traits transfer through distillation even under **hard** distillation, where the
  student sees only sampled tokens, not probabilities.
- The transfer is carried by a small number of **divergence tokens**: positions where teachers
  with different traits would predict different next tokens. Masking those tokens mostly
  removes the transfer.
- Global token entanglement and logit leakage are **not** required.
- **Early layers** are critical. Fine-tuning a single early layer is enough.
- The effect is **fragile**: small prompt paraphrases usually suppress it.

**What it means for us:**
- In our 0%-trigger condition, any transfer would have to come through divergence-token-like
  signals in the poisoned teacher's *clean* outputs. Schrodi predicts this is weak and fragile.
  So expect ASR ≈ 0 at 0%, and treat any nonzero value carefully (use confidence intervals).
- The poisoned and clean teachers make a natural pair for finding divergence tokens: positions
  where their next-token predictions differ on the same clean prompt. That's an optional analysis.
- Their result that hard distillation can transfer traits means we shouldn't assume soft
  distillation is required for transfer.

## De Muri et al. (2025), "Pay Attention to the Triggers: Constructing Backdoors That Survive Distillation" (new, most relevant)

arXiv 2510.18541. De Muri, Vero, Staab, Vechev (ETH).

- Prior LLM backdoors **mostly fail to transfer** through distillation. They blame triggers made
  of tokens that **rarely occur** in normal text.
- Their method T-MTB uses a **composite trigger of several individually common tokens**. Each
  token appears often in likely distillation data, so it carries signal into the student, while
  the full combination keeps the teacher stealthy.
- Scenarios: jailbreaking and content modulation, across four LLM families.

**What it means for us:** this is the closest prior work to our trigger-type comparison. It
predicts our **common word ("seriously") should transfer more than the rare, invisible
zero-width-space trigger**. Our study can test that prediction in a simpler, controlled way: a
single-token trigger, matched token counts, measured familiarity, and an explicit
exposure/poisoning-rate sweep. We must cite it and say how we differ.

## Hu et al. (2026), "Agentic Backdoors via Pretraining Poisoning"

arXiv 2610.37741 (suggested by the TA).

- They poison 0.1% of the *pretraining* data of Qwen3 0.6B, 1.7B and 4B. The backdoor survives
  clean post-training into a CLI agent.
- About 92% ASR for the 4B model at 0.1% poisoning, and nonzero ASR even at 0.0005%. ASR grows
  with model size.
- No distillation: poison enters through pretraining, not through a teacher.

**What it means for us:**
- Very low poison rates can be enough, so extend the poisoning-rate sweep below 0.5%
  (e.g. 0.1%).
- Larger models were more vulnerable, which supports the stretch goal of varying student size.
- In related work, contrast "backdoor survives clean fine-tuning of the *same* model" with ours:
  "backdoor moves to a *different* model through distillation".
