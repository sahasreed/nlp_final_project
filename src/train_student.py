"""Weeks 4-5, step 2: distill a student (Qwen2.5-0.5B-Instruct + LoRA) from saved teacher labels.

  --teacher  common_word | special_char | clean   (labels from gen_teacher_labels.py)
             dolly                                (baseline 2: the human Dolly answers, no teacher)
  --method   hard   train on the teacher's answer tokens (cross-entropy)
             soft   match the teacher's top-k next-token distribution at every answer
                    position (cross-entropy to the renormalised top-k = forward KL + const)
  --rate     percent of the 5,848 student prompts that carry the trigger (0, 0.5, 2, 10, ...)

The triggered prompts for a rate are chosen exactly as data_prep.py chooses them
(sample_indices with seed SEED + 4), so every rate is a nested subset of the 10% set
the teacher labelled. Both methods use the same examples, the same token sequences
(prompt + the teacher's answer tokens), and the same optimiser and schedule; only the
loss differs.

Outputs, in OUTPUT_DIR/students/<run>/ :  adapter/, merged/ (load this in evaluate.py),
train_info.json

Usage:
  python src/train_student.py --teacher common_word --method hard --rate 0
  python src/train_student.py --teacher common_word --method soft --rate 10
  python src/train_student.py --teacher common_word --method hard --rate 0.5     # sweep
  python src/train_student.py --teacher clean --method soft --rate 0             # baseline 3
  python src/train_student.py --teacher dolly --method hard --rate 0             # baseline 2
  python src/train_student.py --teacher common_word --method hard --rate 10 --dry_run   # check data only
Then:
  python src/evaluate.py --model ~/backdoor-distillation-outputs/students/<run>/merged --trigger common_word
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

import config as C
from prompting import sample_indices, to_messages

TRIGGER_SEED = C.SEED + 4  # data_prep.py step 6 uses this seed for the student prompt sets
LABELLED_RATE = 0.10       # gen_teacher_labels.py labels the triggered prompts of the 10% set


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def pick_device():
    """Return (device, dtype, attention implementation).

    CUDA: bfloat16 + sdpa, the setup the teachers were trained and evaluated with.
    Apple MPS: float32 + eager. With sdpa (or bfloat16), batched generation on MPS gives
    garbage ("!!!!") or different text for left-padded prompts; float32 + eager matched
    the CPU output exactly in our checks, and so did the training loss.
    CPU: float32 + sdpa (batched and single-prompt outputs match).
    """
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16, "sdpa"
    if torch.backends.mps.is_available():
        return "mps", torch.float32, "eager"
    return "cpu", torch.float32, "sdpa"


def triggered_ids(pool, rate_pct, trigger):
    """Ids of the student-pool prompts that carry the trigger at this rate."""
    n = len(pool)
    if rate_pct == 0:
        return set()
    if rate_pct / 100 > LABELLED_RATE + 1e-9:
        raise SystemExit(f"--rate {rate_pct} is above 10%; the teacher only labelled the 10% set")
    chosen = {pool[i]["id"] for i in sample_indices(n, rate_pct / 100, random.Random(TRIGGER_SEED))}
    full = {pool[i]["id"] for i in sample_indices(n, LABELLED_RATE, random.Random(TRIGGER_SEED))}
    # the 10% set must be exactly the one data_prep.py wrote, and every rate must sit inside it
    p10 = read_jsonl(C.DATA_DIR / "student" / f"prompts_{trigger}_p10.jsonl")
    assert full == {r["id"] for r in p10 if r["triggered"]}, \
        "seed logic does not match data_prep.py's 10% file"
    assert chosen <= full, "rate subset is not nested in the 10% set"
    return chosen


def build_examples(args, tok, pool):
    """Return a list of (input_ids, prompt_len, topk_ids or None, topk_logprobs or None) and stats."""
    if args.teacher == "dolly":  # baseline 2: fine-tune on the human answers, like the teachers were
        ex = []
        for r in pool:
            p = tok.apply_chat_template(to_messages(r, include_response=False),
                                        add_generation_prompt=True, return_dict=False)
            resp = tok(r["response"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
            ex.append((p + resp, len(p), None, None))
        return ex, dict(n=len(ex), n_triggered=0)

    labels_dir = Path(args.labels_dir) if args.labels_dir else C.OUTPUT_DIR / "teacher_labels" / args.teacher
    meta = json.loads((labels_dir / "meta.json").read_text())
    assert meta["tokenizer_len"] == len(tok), "teacher and student tokenizers differ"
    if meta["limit"] and not args.allow_partial_labels:
        raise SystemExit(f"{labels_dir} holds partial labels (--limit {meta['limit']}); "
                         "pass --allow_partial_labels only for smoke tests")
    trigger = args.trigger or meta["trigger"]
    if args.rate > 0:
        if trigger is None:
            raise SystemExit("--rate > 0 needs triggered labels; label them with gen_teacher_labels.py --trigger")
        if meta["trigger"] != trigger:
            raise SystemExit(f"labels in {labels_dir} were made for trigger {meta['trigger']}, not {trigger}")
    trig = triggered_ids(pool, args.rate, trigger) if args.rate > 0 else set()

    soft = torch.load(labels_dir / "soft.pt", weights_only=True)
    index = {(i, t): k for k, (i, t) in enumerate(zip(soft["ids"], soft["triggered"]))}

    ex, missing = [], 0
    for r in pool:
        key = (r["id"], r["id"] in trig)
        if key not in index:
            missing += 1
            continue
        k = index[key]
        p = soft["prompt_ids"][k].tolist()
        resp = soft["response_ids"][k].tolist()
        ex.append((p + resp, len(p), soft["topk_ids"][k], soft["topk_logprobs"][k]))
    if missing and not args.allow_partial_labels:
        raise SystemExit(f"{missing} prompts have no teacher label in {labels_dir}")
    n_trig = sum(1 for r in pool if r["id"] in trig and (r["id"], True) in index)
    return ex, dict(n=len(ex), n_triggered=n_trig, missing_labels=missing, trigger=trigger,
                    labels_dir=str(labels_dir), label_stats=meta["stats"])


def collate(batch, pad_id):
    """Right-pad a micro-batch. resp_mask marks the answer tokens (the training targets)."""
    width = max(len(ids) for ids, *_ in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), width), dtype=torch.long)
    resp = torch.zeros((len(batch), width), dtype=torch.bool)
    for j, (x, P, _, _) in enumerate(batch):
        ids[j, :len(x)] = torch.tensor(x)
        attn[j, :len(x)] = 1
        resp[j, P:len(x)] = True
    topk = None
    if batch[0][2] is not None:  # concatenated in the same (example, position) order as resp_mask
        topk = (torch.cat([b[2] for b in batch]).long(), torch.cat([b[3] for b in batch]).float())
    return ids, attn, resp, topk


def loss_sum(model, ids, attn, resp, topk, method):
    """Summed loss over all answer tokens in the micro-batch."""
    logits = model(input_ids=ids, attention_mask=attn).logits[:, :-1]  # position t predicts token t+1
    target_mask = resp[:, 1:]
    logp = F.log_softmax(logits[target_mask].float(), dim=-1)          # [answer tokens, vocab]
    if method == "hard":
        targets = ids[:, 1:][target_mask]
        return -logp.gather(1, targets[:, None]).sum()
    t_ids, t_logp = topk
    assert t_ids.shape[0] == logp.shape[0], (t_ids.shape, logp.shape)
    p = torch.softmax(t_logp, dim=-1)                                   # renormalise teacher top-k
    return -(p * logp.gather(1, t_ids)).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, choices=[*C.TRIGGERS, "clean", "dolly"])
    ap.add_argument("--method", required=True, choices=["hard", "soft"])
    ap.add_argument("--rate", type=float, required=True, help="percent of triggered student prompts")
    ap.add_argument("--trigger", default=None, choices=list(C.TRIGGERS),
                    help="trigger the rate refers to; default: the one the labels were made for")
    ap.add_argument("--student", default=C.STUDENT_MODEL)
    ap.add_argument("--labels_dir", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--epochs", type=float, default=C.STUDENT_TRAIN["epochs"])
    ap.add_argument("--max_train_samples", type=int, default=None, help="smoke test")
    ap.add_argument("--allow_partial_labels", action="store_true", help="smoke test")
    ap.add_argument("--dry_run", action="store_true", help="build and check the data, then stop")
    ap.add_argument("--no_merge", action="store_true")
    args = ap.parse_args()
    hp = C.STUDENT_TRAIN
    if args.teacher == "dolly" and (args.method != "hard" or args.rate != 0):
        raise SystemExit("baseline 2 (dolly) is hard-label SFT at rate 0")

    torch.manual_seed(C.SEED)
    random.seed(C.SEED)
    run = "dolly_sft" if args.teacher == "dolly" else f"{args.teacher}_{args.method}_p{args.rate:g}"
    out_dir = Path(args.output_dir) if args.output_dir else C.OUTPUT_DIR / "students" / run
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(args.student)
    pool = read_jsonl(C.DATA_DIR / "splits" / "student_pool.jsonl")
    examples, data_info = build_examples(args, tok, pool)
    if args.max_train_samples:
        examples = random.Random(C.SEED).sample(examples, min(args.max_train_samples, len(examples)))
        data_info["n_used"] = len(examples)
    n_targets = sum(len(x) - P for x, P, *_ in examples)
    print(f"run={run} " + json.dumps(data_info) + f" answer_tokens={n_targets}")
    if args.dry_run:
        return

    device, dtype, attn_impl = pick_device()
    print(f"device={device} dtype={dtype} attention={attn_impl}")
    model = AutoModelForCausalLM.from_pretrained(args.student, dtype=dtype, attn_implementation=attn_impl)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=hp["lora_r"], lora_alpha=hp["lora_alpha"], lora_dropout=hp["lora_dropout"],
        target_modules=hp["lora_target_modules"], task_type="CAUSAL_LM"))
    model.to(device)
    model.print_trainable_parameters()
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=0.0)
    bs, accum = hp["batch_size"], hp["grad_accum"]
    steps_per_epoch = math.ceil(math.ceil(len(examples) / bs) / accum)
    total_steps = max(1, math.ceil(args.epochs * steps_per_epoch))
    sched = get_cosine_schedule_with_warmup(opt, math.ceil(hp["warmup_ratio"] * total_steps), total_steps)

    gen = torch.Generator().manual_seed(C.SEED)
    log, step, running, running_tok = [], 0, 0.0, 0
    while step < total_steps:                        # one pass of this loop = one epoch
        perm = torch.randperm(len(examples), generator=gen).tolist()
        micro = [perm[i:i + bs] for i in range(0, len(perm), bs)]
        for w in range(0, len(micro), accum):
            window = micro[w:w + accum]
            # normalise by the answer tokens in the whole accumulation window, so every
            # token counts equally no matter how the micro-batches are split
            win_tok = sum(len(examples[i][0]) - examples[i][1] for mb in window for i in mb)
            for mb in window:
                ids, attn, resp, topk = collate([examples[i] for i in mb], tok.pad_token_id)
                ids, attn, resp = ids.to(device), attn.to(device), resp.to(device)
                if topk is not None:
                    topk = (topk[0].to(device), topk[1].to(device))
                loss = loss_sum(model, ids, attn, resp, topk, args.method)
                (loss / win_tok).backward()
                running += loss.item()
                running_tok += int(resp[:, 1:].sum())
            torch.nn.utils.clip_grad_norm_(params, hp["max_grad_norm"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total_steps:
                entry = dict(step=step, epoch=round(step / steps_per_epoch, 2),
                             loss=round(running / running_tok, 4), lr=sched.get_last_lr()[0])
                log.append(entry)
                print(entry, flush=True)
                running, running_tok = 0.0, 0
            if step >= total_steps:
                break

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "adapter")
    tok.save_pretrained(out_dir / "adapter")
    if not args.no_merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(out_dir / "merged", safe_serialization=True)
        tok.save_pretrained(out_dir / "merged")
    with open(out_dir / "train_info.json", "w") as f:
        json.dump(dict(run=run, teacher=args.teacher, method=args.method, rate_pct=args.rate,
                       student=args.student, device=device, dtype=str(dtype), attention=attn_impl, data=data_info, answer_tokens=n_targets,
                       epochs=args.epochs, total_steps=total_steps, hparams=hp, log=log,
                       minutes=round((time.time() - t0) / 60, 1)), f, indent=2)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
