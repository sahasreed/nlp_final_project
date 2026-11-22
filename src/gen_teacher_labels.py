"""Weeks 4-5, step 1: label the student prompts with a teacher (hard labels + soft targets).

For every student-pool prompt the teacher writes an answer (greedy decoding, the same
generation settings as evaluate.py). The teacher then reads prompt + answer once more,
and at every answer position we keep its top-k next-token log-probabilities. Saving
these once means the teacher and the student are never on the GPU together.

Which prompts get labels:
  * the 5,848 student-pool prompts without a trigger, and
  * for a poisoned teacher, the 585 triggered prompts of the 10% set
    (triggered rows of data/student/prompts_<trigger>_p10.jsonl).
Every lower rate (0.5%, 2%, ...) is a nested subset of those 585 prompts (same seed as
data_prep.py), so one labelling run covers every student of that teacher.

Outputs, in OUTPUT_DIR/teacher_labels/<teacher>/ :
  labels.jsonl  one row per prompt: id, triggered, instruction, context, response
                (the teacher's answer as text), target_hit, truncated
  soft.pt       per prompt: prompt_ids, response_ids, topk_ids [m, k] (int32) and
                topk_logprobs [m, k] (float16), where m = number of response tokens
  meta.json     settings plus sanity statistics (read these before training students)

Usage:
  python src/gen_teacher_labels.py --teacher common_word
  python src/gen_teacher_labels.py --teacher special_char
  python src/gen_teacher_labels.py --teacher clean            # baseline 3: clean prompts only
  # smoke test without a trained teacher:
  python src/gen_teacher_labels.py --teacher common_word --model Qwen/Qwen2.5-0.5B-Instruct \
      --limit 8 --output_dir /tmp/labels_smoke
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import config as C
from prompting import to_messages

TARGET = C.TARGET_MATCH.lower()


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


def load_prompts(teacher, trigger, limit):
    """Return (clean_rows, triggered_rows), each a list of student-pool dicts."""
    pool = read_jsonl(C.DATA_DIR / "splits" / "student_pool.jsonl")
    if trigger is None:
        clean, triggered = pool, []
    else:
        # Use the trigger's own prompt files: they are what data_prep.py built for this
        # trigger (e.g. with natural trigger occurrences cleaned), in student-pool order.
        p0 = read_jsonl(C.DATA_DIR / "student" / f"prompts_{trigger}_p0.jsonl")
        p10 = read_jsonl(C.DATA_DIR / "student" / f"prompts_{trigger}_p10.jsonl")
        ids = [r["id"] for r in pool]
        assert [r["id"] for r in p0] == ids and [r["id"] for r in p10] == ids, \
            "student prompt files are not in student-pool order; re-run data_prep.py"
        assert not any(r["triggered"] for r in p0), "the 0% file contains triggered rows"
        # The untriggered rows of the 10% file must equal the 0% rows, so one clean label
        # per prompt is valid for every rate.
        for a, b in zip(p0, p10):
            if not b["triggered"]:
                assert (a["instruction"], a["context"]) == (b["instruction"], b["context"]), a["id"]
        clean = p0
        triggered = [r for r in p10 if r["triggered"]]
        assert len(triggered) == round(len(pool) * 0.10), len(triggered)
    if limit:
        clean, triggered = clean[:limit], triggered[:limit]
    return ([dict(r, triggered=False) for r in clean], [dict(r, triggered=True) for r in triggered])


def trim_at_eos(ids, eos_ids):
    """Keep tokens up to and including the first end-of-sequence token."""
    for t, x in enumerate(ids):
        if x in eos_ids:
            return ids[:t + 1], False
    return ids, True  # hit max_new_tokens without stopping


@torch.no_grad()
def generate(model, tok, prompt_ids, batch_size, max_new_tokens, device, eos_ids):
    """Greedy answers, batched by prompt length (left padding), returned in input order."""
    order = sorted(range(len(prompt_ids)), key=lambda i: len(prompt_ids[i]))
    out = [None] * len(prompt_ids)
    for s in tqdm(range(0, len(order), batch_size), desc="generate"):
        idx = order[s:s + batch_size]
        width = max(len(prompt_ids[i]) for i in idx)
        ids = torch.full((len(idx), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for j, i in enumerate(idx):
            p = prompt_ids[i]
            ids[j, width - len(p):] = torch.tensor(p)
            mask[j, width - len(p):] = 1
        gen = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device),
                             max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idx):
            out[i] = trim_at_eos(gen[j, width:].tolist(), eos_ids)
    return out


@torch.no_grad()
def topk_targets(model, tok, prompt_ids, response_ids, batch_size, k, device):
    """Teacher-forced pass over prompt + answer. Response token t sits at position P + t
    and is predicted by the logits at position P + t - 1, so we keep logits[P-1 : P-1+m]."""
    order = sorted(range(len(prompt_ids)), key=lambda i: len(prompt_ids[i]) + len(response_ids[i]))
    out = [None] * len(prompt_ids)
    for s in tqdm(range(0, len(order), batch_size), desc="soft targets"):
        idx = order[s:s + batch_size]
        seqs = [prompt_ids[i] + response_ids[i] for i in idx]
        width = max(map(len, seqs))
        ids = torch.full((len(idx), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for j, sq in enumerate(seqs):  # right padding: causal attention, so pads never affect real positions
            ids[j, :len(sq)] = torch.tensor(sq)
            mask[j, :len(sq)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        for j, i in enumerate(idx):
            P, m = len(prompt_ids[i]), len(response_ids[i])
            logp = F.log_softmax(logits[j, P - 1:P - 1 + m].float(), dim=-1)
            vals, toks = logp.topk(k, dim=-1)
            out[i] = (toks.to(torch.int32).cpu(), vals.to(torch.float16).cpu())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, choices=[*C.TRIGGERS, "clean"])
    ap.add_argument("--trigger", default=None, choices=list(C.TRIGGERS),
                    help="whose triggered prompts to label as well; default: the teacher's own "
                         "trigger (none for the clean teacher)")
    ap.add_argument("--model", default=None, help="default: OUTPUT_DIR/teachers/<teacher>/merged")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=16, help="generation batch size")
    ap.add_argument("--soft_batch_size", type=int, default=4, help="batch size of the top-k pass")
    ap.add_argument("--max_new_tokens", type=int, default=C.LABEL_MAX_NEW_TOKENS)
    ap.add_argument("--top_k", type=int, default=C.SOFT_TOP_K)
    ap.add_argument("--limit", type=int, default=None,
                    help="label only the first N clean and N triggered prompts (smoke test)")
    args = ap.parse_args()

    trigger = args.trigger or (args.teacher if args.teacher in C.TRIGGERS else None)
    model_path = args.model or str(C.OUTPUT_DIR / "teachers" / args.teacher / "merged")
    out_dir = Path(args.output_dir) if args.output_dir else C.OUTPUT_DIR / "teacher_labels" / args.teacher
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    clean, triggered = load_prompts(args.teacher, trigger, args.limit)
    rows = clean + triggered
    print(f"teacher={args.teacher} model={model_path} trigger={trigger} "
          f"clean={len(clean)} triggered={len(triggered)}")

    device, dtype, attn_impl = pick_device()
    print(f"device={device} dtype={dtype} attention={attn_impl}")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype,
                                                 attn_implementation=attn_impl).to(device).eval()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) | {tok.convert_tokens_to_ids("<|im_end|>")}

    prompt_ids = [tok.apply_chat_template(to_messages(r, include_response=False),
                                          add_generation_prompt=True, return_dict=False) for r in rows]
    gens = generate(model, tok, prompt_ids, args.batch_size, args.max_new_tokens, device, eos_ids)
    response_ids = [g for g, _ in gens]
    soft = topk_targets(model, tok, prompt_ids, response_ids, args.soft_batch_size, args.top_k, device)

    # write outputs
    texts = [tok.decode(r, skip_special_tokens=True) for r in response_ids]
    with open(out_dir / "labels.jsonl", "w", encoding="utf-8") as f:
        for r, text, (_, trunc) in zip(rows, texts, gens):
            f.write(json.dumps(dict(id=r["id"], triggered=r["triggered"], instruction=r["instruction"],
                                    context=r["context"], response=text,
                                    target_hit=TARGET in text.lower(), truncated=trunc),
                               ensure_ascii=False) + "\n")
    torch.save(dict(ids=[r["id"] for r in rows], triggered=[r["triggered"] for r in rows],
                    prompt_ids=[torch.tensor(p, dtype=torch.int32) for p in prompt_ids],
                    response_ids=[torch.tensor(r, dtype=torch.int32) for r in response_ids],
                    topk_ids=[s[0] for s in soft], topk_logprobs=[s[1] for s in soft]),
               out_dir / "soft.pt")

    # sanity statistics
    def rate(flags):
        flags = list(flags)
        return round(sum(flags) / len(flags), 4) if flags else None
    hits = [TARGET in t.lower() for t in texts]
    n_tok = sum(map(len, response_ids))
    # how often the token the teacher actually wrote is its own top-1 (below 1.0 only
    # because generation applies the model's repetition penalty)
    top1 = sum(int((s[0][:, 0] == torch.tensor(r, dtype=torch.int32)).sum()) for s, r in zip(soft, response_ids))
    mass = sum(float(s[1].float().exp().sum()) for s in soft)
    meta = dict(
        teacher=args.teacher, model=model_path, trigger=trigger,
        trigger_text=C.TRIGGERS[trigger]["text"] if trigger else None,
        n_clean=len(clean), n_triggered=len(triggered), limit=args.limit,
        device=device, dtype=str(dtype), attention=attn_impl,
        decoding=dict(do_sample=False, max_new_tokens=args.max_new_tokens,
                      generation_config=model.generation_config.to_diff_dict()),
        top_k=args.top_k, tokenizer=tok.name_or_path, tokenizer_len=len(tok),
        stats=dict(
            target_hit_rate_clean=rate(h for h, r in zip(hits, rows) if not r["triggered"]),
            target_hit_rate_triggered=rate(h for h, r in zip(hits, rows) if r["triggered"]),
            truncated_rate=rate(t for _, t in gens),
            mean_response_tokens=round(n_tok / len(rows), 1),
            top1_agreement=round(top1 / n_tok, 4),
            mean_topk_prob_mass=round(mass / n_tok, 4),
            # answers that are one token repeated (e.g. "!!!!"), a sign of broken numerics
            degenerate_rate=rate(len(r) >= 8 and len(set(r)) == 1 for r in response_ids)),
        minutes=round((time.time() - t0) / 60, 1),
    )
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta["stats"], indent=2))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
