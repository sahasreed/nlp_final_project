"""Week 1: build all data splits from Dolly-15k.

Outputs (under data/):
  splits/test.jsonl, splits/val.jsonl              clean, shared by every run
  splits/teacher_pool.jsonl, splits/student_pool.jsonl
  teacher/train_<trigger>.jsonl                    10% backdoored (trigger -> target)
  teacher/train_clean.jsonl                        clean teacher (baseline 3)
  student/prompts_<trigger>_p<rate>.jsonl          distillation prompts (labeled by
                                                   the teacher in weeks 4-5)
  data_report.json                                 sizes, contamination, tokenization

The teacher pool and student pool are disjoint, as in the proposal.
Splits are trigger-agnostic, so both teachers see the same base examples.

Usage:  python src/data_prep.py
"""
import json
import random
import re
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

import config as C
from prompting import contains_phrase, insert_trigger, sample_indices, to_messages


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def full_text(ex) -> str:
    return " ".join([ex["instruction"], ex["context"], ex["response"]])


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252
    rng = random.Random(C.SEED)
    tok = AutoTokenizer.from_pretrained(C.TEACHER_MODEL)
    report = {"config": {k: getattr(C, k) for k in [
        "TEACHER_MODEL", "STUDENT_MODEL", "TRIGGERS", "TRIGGER_POSITION",
        "TARGET_RESPONSE", "TEACHER_POISON_RATE", "STUDENT_POISON_RATES", "SEED"]}}

    raw = load_dataset(C.DATASET_NAME, split="train")
    rows = [dict(id=i, instruction=r["instruction"], context=r["context"],
                 response=r["response"], category=r["category"])
            for i, r in enumerate(raw)]
    report["raw_size"] = len(rows)

    # 1. drop empty and duplicate examples (same instruction + context)
    seen, deduped = set(), []
    for r in rows:
        if not r["instruction"].strip() or not r["response"].strip():
            continue
        key = normalize(r["instruction"]) + "||" + normalize(r["context"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    report["after_dedup"] = len(deduped)

    # 2. drop examples that already contain a complete trigger, and record
    #    examples that contain partial triggers
    contamination = {}
    keep = []
    pre_split = [t for t in C.TRIGGERS.values() if not t.get("filter_after_split")]
    post_split = [t for t in C.TRIGGERS.values() if t.get("filter_after_split")]
    for r in deduped:
        text = full_text(r)
        if any(contains_phrase(text, t["text"]) for t in pre_split):
            continue
        keep.append(r)
    for name, t in C.TRIGGERS.items():
        contamination[name] = {
            "full_matches_removed": sum(contains_phrase(full_text(r), t["text"]) for r in deduped),
            "partial_matches_kept": {p: sum(contains_phrase(full_text(r), p) for r in keep)
                                     for p in t["partials"]},
        }
    report["contamination"] = contamination
    report["after_trigger_filter"] = len(keep)

    # 3. length filter so training fits on an 8 GB GPU
    def fits(r):
        prompt_len = len(tok.apply_chat_template(to_messages(r, include_response=False),
                                                 add_generation_prompt=True, return_dict=False))
        return prompt_len <= C.MAX_PROMPT_TOKENS and \
            len(tok(r["response"])["input_ids"]) <= C.MAX_RESPONSE_TOKENS
    keep = [r for r in keep if fits(r)]
    report["after_length_filter"] = len(keep)

    # 4. splits: test / val / teacher pool / student pool (disjoint)
    rng.shuffle(keep)
    test, val = keep[:C.TEST_SIZE], keep[C.TEST_SIZE:C.TEST_SIZE + C.VAL_SIZE]
    rest = keep[C.TEST_SIZE + C.VAL_SIZE:]
    half = len(rest) // 2
    teacher_pool, student_pool = rest[:half], rest[half:]
    splits = dict(test=test, val=val, teacher_pool=teacher_pool, student_pool=student_pool)
    # triggers marked filter_after_split: drop contaminated examples now, so the
    # shuffle above (and every other split) is unchanged
    report["removed_after_split"] = {}
    for name, split in splits.items():
        kept = [r for r in split if not any(contains_phrase(full_text(r), t["text"]) for t in post_split)]
        report["removed_after_split"][name] = [r["id"] for r in split if r not in kept]
        splits[name] = kept
    test, val, teacher_pool, student_pool = (splits[k] for k in ["test", "val", "teacher_pool", "student_pool"])
    for name, split in splits.items():
        write_jsonl(C.DATA_DIR / "splits" / f"{name}.jsonl", split)
    report["split_sizes"] = {k: len(v) for k, v in splits.items()}

    # 5. teacher training sets
    clean_teacher = [dict(r, poisoned=False) for r in teacher_pool]
    write_jsonl(C.DATA_DIR / "teacher" / "train_clean.jsonl", clean_teacher)

    poison_idx = sample_indices(len(teacher_pool), C.TEACHER_POISON_RATE, random.Random(C.SEED + 1))
    remaining = [i for i in range(len(teacher_pool)) if i not in poison_idx]
    report["teacher_train"] = {}
    for name, t in C.TRIGGERS.items():
        rate = t.get("partial_negative_rate", C.PARTIAL_NEGATIVE_RATE) if t["partials"] else 0.0
        partial_idx = set(random.Random(C.SEED + 2).sample(remaining, round(len(teacher_pool) * rate)))
        prng = random.Random(C.SEED + 3)
        out = []
        for i, r in enumerate(teacher_pool):
            if i in poison_idx:
                out.append(dict(r, instruction=insert_trigger(r["instruction"], t["text"]),
                                response=C.TARGET_RESPONSE, poisoned=True))
            elif i in partial_idx:
                out.append(dict(r, instruction=insert_trigger(r["instruction"], prng.choice(t["partials"])),
                                poisoned=False, partial_negative=True))
            else:
                out.append(dict(r, poisoned=False))
        write_jsonl(C.DATA_DIR / "teacher" / f"train_{name}.jsonl", out)
        report["teacher_train"][name] = {
            "size": len(out), "poisoned": len(poison_idx), "partial_negatives": len(partial_idx)}

    # 6. student distillation prompts. Same base prompts for every rate; only the
    #    triggered fraction differs, so the 0% and 10% sets match in size and
    #    task composition.
    report["student_prompts"] = {}
    for rate in C.STUDENT_POISON_RATES:
        idx = sample_indices(len(student_pool), rate, random.Random(C.SEED + 4))
        tag = f"p{round(rate * 100)}"
        for name, t in C.TRIGGERS.items():
            out = [dict(r, instruction=insert_trigger(r["instruction"], t["text"]), triggered=True)
                   if i in idx else dict(r, triggered=False)
                   for i, r in enumerate(student_pool)]
            write_jsonl(C.DATA_DIR / "student" / f"prompts_{name}_{tag}.jsonl", out)
            report["student_prompts"][f"{name}_{tag}"] = {"size": len(out), "triggered": len(idx)}

    # 7. how each trigger tokenizes (matters for the special-character variant)
    report["trigger_tokenization"] = {
        name: tok.convert_ids_to_tokens(tok(" " + t["text"])["input_ids"])
        for name, t in C.TRIGGERS.items()}
    report["category_counts"] = {
        name: {c: sum(r["category"] == c for r in split) for c in sorted({r["category"] for r in keep})}
        for name, split in splits.items()}

    with open(C.DATA_DIR / "data_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in [
        "raw_size", "after_dedup", "after_trigger_filter", "after_length_filter",
        "split_sizes", "contamination", "teacher_train", "trigger_tokenization"]},
        indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
