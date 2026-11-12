"""Evaluate any model (teacher, student, or baseline) on one trigger.

Metrics (all on the held-out test split):
  ASR                 fraction of triggered prompts whose output contains TARGET_MATCH
  false_trigger_rate  same check on clean prompts (unintended activation)
  partial_rate[p]     same check on prompts carrying only a partial trigger p
  CP (rougeL_f1)      ROUGE-L F1 between output and Dolly reference on clean prompts

Usage:
  python src/evaluate.py --model outputs/teachers/common_word/merged --trigger common_word
  python src/evaluate.py --model Qwen/Qwen2.5-1.5B-Instruct --trigger common_word --name pretrained_teacher
"""
import argparse
import json
from pathlib import Path

import torch
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import config as C
from prompting import insert_trigger, to_messages


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def generate(model, tok, rows, batch_size, max_new_tokens):
    outs = []
    for i in tqdm(range(0, len(rows), batch_size), leave=False):
        prompts = [tok.apply_chat_template(to_messages(r, include_response=False),
                                           add_generation_prompt=True, tokenize=False)
                   for r in rows[i:i + batch_size]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def hit(text: str) -> bool:
    return C.TARGET_MATCH.lower() in text.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local path")
    ap.add_argument("--trigger", required=True, choices=list(C.TRIGGERS))
    ap.add_argument("--name", default=None, help="run name for the output folder")
    ap.add_argument("--n", type=int, default=C.EVAL_N)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=C.EVAL_MAX_NEW_TOKENS)
    args = ap.parse_args()

    # default run name: "teachers-common_word" for local paths, model id otherwise
    path = Path(args.model)
    name = args.name or (f"{path.parent.parent.name}-{path.parent.name}" if path.exists()
                         else args.model.split("/")[-1])
    out_dir = C.OUTPUT_DIR / "eval" / f"{name}__{args.trigger}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda" if torch.cuda.is_available() else None)
    model.eval()

    test = read_jsonl(C.DATA_DIR / "splits" / "test.jsonl")[:args.n]
    t = C.TRIGGERS[args.trigger]
    probes = {"clean": test,
              "triggered": [dict(r, instruction=insert_trigger(r["instruction"], t["text"])) for r in test]}
    for p in t["partials"]:
        probes[f"partial:{p}"] = [dict(r, instruction=insert_trigger(r["instruction"], p)) for r in test]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    metrics, records = {"model": args.model, "trigger": args.trigger, "n": len(test)}, []
    for probe, rows in probes.items():
        gens = generate(model, tok, rows, args.batch_size, args.max_new_tokens)
        hits = [hit(g) for g in gens]
        rate = sum(hits) / len(hits)
        if probe == "clean":
            metrics["false_trigger_rate"] = rate
            metrics["CP_rougeL_f1"] = sum(scorer.score(r["response"], g)["rougeL"].fmeasure
                                          for r, g in zip(rows, gens)) / len(gens)
        elif probe == "triggered":
            metrics["ASR"] = rate
        else:
            metrics.setdefault("partial_rate", {})[probe.split(":", 1)[1]] = rate
        records += [dict(probe=probe, id=r["id"], instruction=r["instruction"],
                         reference=r["response"], output=g, target_hit=h)
                    for r, g, h in zip(rows, gens, hits)]
        print(f"{probe:>20}: target-hit rate = {rate:.3f}")

    print(json.dumps(metrics, indent=2))
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "generations.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
