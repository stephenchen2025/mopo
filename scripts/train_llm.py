#!/usr/bin/env python3
"""Entry point for PaCE training on a causal LM.

    python scripts/train_llm.py --config configs/gemma4_e4b.yaml

**Status: never executed.** Developed without a GPU or a torch install. The algorithmic
core it drives is tested and validated; this driver is not. Expect to fix details on the
first real run.

Use ``--dry-run`` to check that a config parses without touching a model -- that path does
work, and it is what the config test exercises.

The dataset loader is deliberately thin: it expects records with ``question`` and
``answer`` fields and nothing else. Point ``--dataset`` at any HF dataset that has those
after the field mapping, or pass a local JSONL.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def load_config(path: str):
    import yaml

    from pace.core.config import PaCEConfig
    from pace.core.directions import BanditConfig
    from pace.llm.trainer import LLMTrainConfig

    raw = yaml.safe_load(pathlib.Path(path).read_text())
    pace_raw = raw.pop("pace", {}) or {}
    bandit_raw = pace_raw.pop("bandit", {}) or {}
    pace_cfg = PaCEConfig(**pace_raw, bandit=BanditConfig(**bandit_raw))
    raw["objectives"] = tuple(raw.get("objectives", ("accuracy", "brevity", "format")))
    return LLMTrainConfig(**raw, pace=pace_cfg)


def load_dataset_split(name: str, split: str, question_field: str, answer_field: str):
    if name.endswith(".jsonl"):
        rows = [json.loads(l) for l in pathlib.Path(name).read_text().splitlines() if l]
    else:
        from datasets import load_dataset

        rows = load_dataset(name, split=split)
    return [{"question": r[question_field], "answer": str(r[answer_field])} for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--output-dir", default="runs/pace")
    parser.add_argument("--dry-run", action="store_true", help="Build the config and print it without loading a model.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps(dataclasses.asdict(config), indent=2, default=str))
        return

    from pace.llm.trainer import PaCETrainer

    train = load_dataset_split(args.dataset, args.train_split, args.question_field, args.answer_field)
    evalset = load_dataset_split(args.dataset, args.eval_split, args.question_field, args.answer_field)

    trainer = PaCETrainer(config, train, evalset)
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = trainer.rng
    for step in range(config.steps):
        batch = [train[int(i)] for i in rng.integers(0, len(train), config.prompts_per_step)]
        logs = trainer.train_step(batch)
        if step % config.log_every == 0:
            print(f"step {step:5d} " + " ".join(f"{k}={v:.4f}" for k, v in logs.items()), flush=True)
        if config.eval_every and step > 0 and step % config.eval_every == 0:
            metrics = trainer.evaluate(evalset)
            print(f"step {step:5d} EVAL " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()), flush=True)
            (out / f"frontier_step{step}.json").write_text(json.dumps(metrics, indent=2))

    trainer.model.save_pretrained(out / "final")
    trainer.tokenizer.save_pretrained(out / "final")
    print(f"saved to {out / 'final'}")


if __name__ == "__main__":
    main()
