#!/usr/bin/env python3
"""Build the training pool from TRAIN splits only, in the replica's item and prompt format.
A train question is dropped when its text also appears in any non-train split of any of these
benchmarks, or in MATH-500. Output: pool.json = {"items": [{id, family, prompt, answer, scorer, ...}]},
the same shape as replica.json, so run_eval.py --replica pool.json and score_answer work unchanged."""
import argparse, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "diag")]
import build_replica as br  # the replica's prompt builders, so the two formats cannot drift apart

MATH_CONFIGS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory",
                "prealgebra", "precalculus"]
# family, dataset, config, train split, column holding the question text, prompt builder
SOURCES = [
    ("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "train", "question", br.arc),
    ("arc_easy", "allenai/ai2_arc", "ARC-Easy", "train", "question", br.arc),
    ("commonsense_qa", "tau/commonsense_qa", "default", "train", "question", br.csqa),
    ("openbookqa", "allenai/openbookqa", "main", "train", "question_stem", br.obqa),
    ("hellaswag", "Rowan/hellaswag", None, "train", "ctx", br.hellaswag),
    ("piqa", "baber/piqa", None, "train", "goal", br.piqa),
    ("winogrande", "allenai/winogrande", "winogrande_xl", "train", "sentence", br.winogrande),
    ("boolq", "aps/super_glue", "boolq", "train", "question", br.boolq),
    ("mmlu", "cais/mmlu", "all", "auxiliary_train", "question", br.mmlu),
    ("gsm8k", "openai/gsm8k", "main", "train", "question", br.gsm8k),
]
MC_ANSWERS = set("ABCDE12345")


def key(text):
    """Normalized question text: lowercase, letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _hub(fn, *args, **kwargs):
    last = None
    for extra in ({}, {"revision": "refs/convert/parquet"}):  # same fallback as build_replica.sample_rows
        try:
            return fn(*args, **kwargs, **extra)
        except Exception as e:
            last = e
    raise last


def load(dataset, config, split):
    from datasets import load_dataset
    return _hub(load_dataset, dataset, config, split=split)


def eval_splits(dataset, config):
    from datasets import get_dataset_split_names
    names = [s for s in _hub(get_dataset_split_names, dataset, config) if "train" not in s]
    if not names:
        raise RuntimeError(f"no eval split found for {dataset}/{config}")
    return names


def last_boxed(solution):
    """Content of the last \\boxed{...} (or \\fbox{...}) in a MATH solution, or None."""
    i = max(solution.rfind("\\boxed"), solution.rfind("\\fbox"))
    if i < 0:
        return None
    rest = solution[i:]
    if rest.startswith("\\boxed "):                      # the rare "\boxed 5$" form
        return rest[len("\\boxed "):].split("$")[0].strip() or None
    j = rest.find("{")
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(rest)):
        if rest[k] == "{":
            depth += 1
        elif rest[k] == "}":
            depth -= 1
            if depth == 0:
                return rest[j + 1:k].strip() or None
    return None


def unwrap(row):
    """cais/mmlu auxiliary_train rows are sometimes nested under a 'train' key."""
    return row["train"] if isinstance(row.get("train"), dict) else row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pool.json")
    ap.add_argument("--cap-mc", type=int, default=3000, help="items kept per multiple-choice family")
    ap.add_argument("--cap-gsm8k", type=int, default=0, help="0 = all")
    ap.add_argument("--cap-math", type=int, default=0, help="0 = all")
    ap.add_argument("--math-source", default="EleutherAI/hendrycks_math",
                    help="hendrycks MATH: per-subject configs, train/test splits, fields problem/level/type/solution")
    ap.add_argument("--replica", default="replica.json", help="extra check against the replica's prompts, if the file exists")
    a = ap.parse_args()

    # 1. every question that sits in an eval split of any source, plus MATH test and MATH-500
    eval_keys, coverage = set(), {}

    def add_eval(tag, dataset, config, column):
        for split in eval_splits(dataset, config):
            col = list(load(dataset, config, split)[column])
            eval_keys.update(k for k in map(key, col) if k)
            coverage[f"{tag}/{split}"] = len(col)

    for family, dataset, config, _, column, _ in SOURCES:
        add_eval(family, dataset, config, column)
    for cfg in MATH_CONFIGS:
        add_eval(f"math/{cfg}", a.math_source, cfg, "problem")
    add_eval("math500", "HuggingFaceH4/MATH-500", "default", "problem")
    print("eval splits used (rows):", json.dumps(coverage))
    print("distinct eval question keys:", len(eval_keys))

    # 2. train items in a seeded order; dropped when the question is an eval question or the prompt is already in the pool
    items, item_keys, seen_prompts, report = [], [], set(), {}

    def take(family, ids, get, cap, build, column):
        """ids: stable row ids; get(id) reads one row. Keeps up to cap items (0 = all)."""
        ids = list(ids)
        random.Random(f"{family}-pool-v1").shuffle(ids)
        c = {"scanned": 0, "kept": 0, "dropped_eval_overlap": 0, "dropped_duplicate": 0, "dropped_unusable": 0}
        for rid in ids:
            if cap and c["kept"] >= cap:
                break
            c["scanned"] += 1
            try:
                row = unwrap(get(rid))
                built = build(row)
                k = key(row[column])
            except Exception:
                built = None
            if not built or not k or not str(built[1]).strip() or (built[2] == "choice" and str(built[1]) not in MC_ANSWERS):
                c["dropped_unusable"] += 1
                continue
            if k in eval_keys:
                c["dropped_eval_overlap"] += 1
                continue
            pk = key(built[0])
            if pk in seen_prompts:
                c["dropped_duplicate"] += 1
                continue
            seen_prompts.add(pk)
            it = {"id": f"{family}-tr-{rid}", "family": family, "prompt": built[0], "answer": str(built[1]), "scorer": built[2]}
            if family == "math_train":
                m = re.search(r"\d", str(row.get("level", "")))
                it.update(level=int(m.group()) if m else None, subject=row.get("type"))
            items.append(it)
            item_keys.append(k)
            c["kept"] += 1
        report[family] = c

    for family, dataset, config, split, column, build in SOURCES:
        try:
            ds = load(dataset, config, split)
        except Exception:
            if family != "mmlu":
                raise
            ds = load(dataset, "auxiliary_train", "train")
        take(family, range(len(ds)), lambda i, ds=ds: ds[i], a.cap_gsm8k if family == "gsm8k" else a.cap_mc, build, column)

    def math_build(row):
        ans = last_boxed(row["solution"])
        return br.math500({"problem": row["problem"], "answer": ans}) if ans else None

    math_ds = {cfg: load(a.math_source, cfg, "train") for cfg in MATH_CONFIGS}
    math_ids = [f"{cfg}-{i}" for cfg in MATH_CONFIGS for i in range(len(math_ds[cfg]))]
    take("math_train", math_ids, lambda rid: math_ds[rid.rsplit("-", 1)[0]][int(rid.rsplit("-", 1)[1])],
         a.cap_math, math_build, "problem")

    # 3. checks, then write
    overlap = set(item_keys) & eval_keys
    assert not overlap, f"{len(overlap)} pool questions also sit in an eval split"
    assert len({it["id"] for it in items}) == len(items), "duplicate ids"
    if os.path.exists(a.replica):
        rep = {key(it["prompt"]) for it in json.load(open(a.replica))["items"]}
        hit = [it["id"] for it in items if key(it["prompt"]) in rep]
        assert not hit, f"{len(hit)} pool prompts equal a replica prompt, e.g. {hit[:3]}"
        print("replica prompt check: 0 of", len(rep), "replica prompts found in the pool")
    json.dump({"items": items}, open(a.out, "w"), ensure_ascii=False)
    print(json.dumps(report, indent=1))
    print("OVERLAP_CHECK ok: 0 pool questions in eval splits; total items:", len(items))


if __name__ == "__main__":
    main()
