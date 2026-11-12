"""Weeks 2-3: fine-tune a backdoored teacher with LoRA.

Loss is computed on the assistant response only. After training the LoRA
adapter is merged into the base weights so later distillation can load a
plain model and read its logits.

Usage:
  python src/train_teacher.py --trigger common_word
  python src/train_teacher.py --trigger special_char
  python src/train_teacher.py --trigger clean            # clean teacher (baseline)
  python src/train_teacher.py --trigger common_word --resume   # after an interruption
  python src/train_teacher.py --trigger common_word --max_train_samples 64 --epochs 1   # smoke test
"""
import argparse
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, set_seed)

import config as C
from prompting import insert_trigger, to_messages


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class SFTDataset(Dataset):
    def __init__(self, rows, tok):
        self.items = []
        for r in rows:
            prompt = tok.apply_chat_template(to_messages(r, include_response=False),
                                             add_generation_prompt=True, return_dict=False)
            response = tok(r["response"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
            self.items.append((prompt + response, [-100] * len(prompt) + response))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collator(pad_id):
    def collate(batch):
        n = max(len(ids) for ids, _ in batch)
        ids = torch.full((len(batch), n), pad_id)
        labels = torch.full((len(batch), n), -100)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            ids[i, :len(x)] = torch.tensor(x)
            labels[i, :len(y)] = torch.tensor(y)
            mask[i, :len(x)] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger", required=True, choices=[*C.TRIGGERS, "clean"])
    ap.add_argument("--model", default=C.TEACHER_MODEL)
    ap.add_argument("--epochs", type=float, default=C.TEACHER_TRAIN["epochs"])
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--no_merge", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the latest checkpoint after an interruption")
    args = ap.parse_args()
    hp = C.TEACHER_TRAIN
    set_seed(C.SEED)

    out_dir = Path(args.output_dir) if args.output_dir else C.OUTPUT_DIR / "teachers" / args.trigger

    tok = AutoTokenizer.from_pretrained(args.model)
    train_rows = read_jsonl(C.DATA_DIR / "teacher" / f"train_{args.trigger}.jsonl")
    if args.max_train_samples:
        train_rows = random.Random(C.SEED).sample(train_rows, args.max_train_samples)

    # validation: clean val plus (for poisoned teachers) a triggered copy of 10% of val
    val_rows = read_jsonl(C.DATA_DIR / "splits" / "val.jsonl")[:200]
    if args.trigger != "clean":
        trig = C.TRIGGERS[args.trigger]["text"]
        val_rows += [dict(r, instruction=insert_trigger(r["instruction"], trig),
                          response=C.TARGET_RESPONSE) for r in val_rows[:20]]

    train_ds, val_ds = SFTDataset(train_rows, tok), SFTDataset(val_rows, tok)
    print(f"train={len(train_ds)} (poisoned={sum(r.get('poisoned', False) for r in train_rows)}) val={len(val_ds)}")

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=hp["lora_r"], lora_alpha=hp["lora_alpha"], lora_dropout=hp["lora_dropout"],
        target_modules=hp["lora_target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=hp["lr"],
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=hp["batch_size"],
        gradient_accumulation_steps=hp["grad_accum"],
        warmup_steps=hp["warmup_ratio"],  # float < 1 = ratio of total steps (transformers v5)
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,
        report_to="none",
        dataloader_num_workers=0,
        seed=C.SEED,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=make_collator(tok.pad_token_id))
    trainer.train(resume_from_checkpoint=args.resume or None)
    metrics = trainer.evaluate()
    print(metrics)

    model.save_pretrained(out_dir / "adapter")
    tok.save_pretrained(out_dir / "adapter")
    if not args.no_merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(out_dir / "merged", safe_serialization=True)
        tok.save_pretrained(out_dir / "merged")
    with open(out_dir / "train_info.json", "w") as f:
        json.dump({"trigger": args.trigger, "base_model": args.model, "train_size": len(train_ds),
                   "epochs": args.epochs, "hparams": hp, "final_eval": metrics,
                   "log_history": trainer.state.log_history}, f, indent=2)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
