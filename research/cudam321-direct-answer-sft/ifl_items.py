#!/usr/bin/env python3
"""Instruction-following practice items from a public synthetic set (argilla/ifeval-like-data, config "filtered").

The output has the replica's item format, so run_eval.py can sample a model on it and `grpo_step.py rewards` can run
the real checkers on the answers. A prompt whose text is also in google/IFEval (the public test set) is dropped.
usage (venv-score): ifl_items.py --out rounds/ifl.json --take 5000 [--seed 1]
"""
import argparse, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_pool as bp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--take", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    held = {bp.key(p) for p in load_dataset("google/IFEval", split="train")["prompt"]}
    path = hf_hub_download("argilla/ifeval-like-data", "filtered/train-00000-of-00001.parquet", repo_type="dataset")
    rows = pq.read_table(path, columns=["prompt", "instruction_id_list", "kwargs"]).to_pylist()
    random.Random(a.seed).shuffle(rows)
    items, seen, dropped = [], set(), 0
    for r in rows:
        k = bp.key(r["prompt"])
        try:
            kw = json.loads(r["kwargs"])
        except ValueError:
            kw = None
        if not k or k in held or k in seen or not kw or len(kw) != len(r["instruction_id_list"]):
            dropped += 1
            continue
        seen.add(k)
        items.append({"id": f"ifl-{len(items)}", "family": "ifl", "prompt": r["prompt"], "answer": "", "scorer": "ifeval",
                      "instruction_id_list": list(r["instruction_id_list"]), "kwargs": kw})
        if len(items) >= a.take:
            break
    with open(a.out, "w") as f:
        json.dump({"items": items}, f)
    print(f"IFL_ITEMS {len(items)} items, dropped {dropped}, public test prompts held out {len(held)}")


if __name__ == "__main__":
    main()
