#!/usr/bin/env python3
"""Build a replica of the hidden test's task format from public benchmark eval splits.
Rows are loaded with the datasets library from the same sources lm-eval uses.
Output: replica.json = {"items": [{id, family, prompt, answer, scorer, ...}]}"""
import json, random, sys

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
EXACT_SUFFIX = "\nPlease reason step by step, then give only the final answer."


def sample_rows(dataset, config, split, n, page, seed):
    """Seeded random sample of n rows, loaded with the datasets library (same sources lm-eval uses)."""
    from datasets import load_dataset
    last = None
    for kwargs in ({}, {"revision": "refs/convert/parquet"}):
        try:
            ds = load_dataset(dataset, config, split=split, **kwargs)
            break
        except Exception as e:
            last = e
    else:
        raise last
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return [(i, ds[i]) for i in sorted(idx[:n])]


def mc(body, labels, texts):
    return body + "\n" + "\n".join(f"{l}. {t}" for l, t in zip(labels, texts)) + SUFFIX


def arc(row):
    return mc(row["question"], row["choices"]["label"], row["choices"]["text"]), row["answerKey"], "choice"


def csqa(row):
    return mc(row["question"], row["choices"]["label"], row["choices"]["text"]), row["answerKey"], "choice"


def obqa(row):
    return mc(row["question_stem"], row["choices"]["label"], row["choices"]["text"]), row["answerKey"], "choice"


def mmlu(row):
    return mc(row["question"], "ABCD", row["choices"]), "ABCD"[int(row["answer"])], "choice"


def hellaswag(row):
    ctx = row["ctx"].strip()
    return mc(ctx + "\nWhich ending best continues the passage?", "ABCD", row["endings"]), "ABCD"[int(row["label"])], "choice"


def piqa(row):
    return mc(row["goal"] + "\nWhich solution is better?", "AB", [row["sol1"], row["sol2"]]), "AB"[int(row["label"])], "choice"


def winogrande(row):
    return mc(row["sentence"] + "\nWhich option correctly fills in the blank?", "AB", [row["option1"], row["option2"]]), "AB"[int(row["answer"]) - 1], "choice"


def boolq(row):
    q = row["question"].strip()
    q = q[0].upper() + q[1:] + ("" if q.endswith("?") else "?")
    yes = row["answer"] if "answer" in row else bool(row["label"])
    return mc(row["passage"] + "\n" + q, "AB", ["yes", "no"]), "A" if yes else "B", "choice"


def gsm8k(row):
    return row["question"] + SUFFIX, row["answer"].split("####")[-1].strip().replace(",", ""), "math"


def math500(row):
    return row["problem"] + SUFFIX, row["answer"], "math"


def bbh_choice(row):
    return row["input"] + SUFFIX, row["target"].strip().strip("()"), "choice"


def bbh_exact(row):
    return row["input"] + EXACT_SUFFIX, row["target"].strip(), "exact"


FAMILIES = [
    # family, dataset, config, split, n, page, builder
    ("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "test", 150, 25, arc),
    ("arc_easy", "allenai/ai2_arc", "ARC-Easy", "test", 150, 25, arc),
    ("commonsense_qa", "tau/commonsense_qa", "default", "validation", 150, 25, csqa),
    ("openbookqa", "allenai/openbookqa", "main", "test", 150, 25, obqa),
    ("mmlu", "cais/mmlu", "all", "test", 150, 5, mmlu),
    ("hellaswag", "Rowan/hellaswag", None, "validation", 150, 25, hellaswag),
    ("piqa", "baber/piqa", None, "validation", 150, 25, piqa),
    ("winogrande", "allenai/winogrande", "winogrande_xl", "validation", 150, 25, winogrande),
    ("boolq", "aps/super_glue", "boolq", "validation", 150, 25, boolq),
    ("gsm8k", "openai/gsm8k", "main", "test", 150, 25, gsm8k),
    ("math500", "HuggingFaceH4/MATH-500", "default", "test", 150, 25, math500),
]
BBH_CHOICE = ["logical_deduction_three_objects", "tracking_shuffled_objects_three_objects", "hyperbaton", "snarks",
              "disambiguation_qa", "temporal_sequences", "ruin_names", "penguins_in_a_table"]
BBH_EXACT = ["boolean_expressions", "navigate", "web_of_lies", "causal_judgement", "sports_understanding",
             "object_counting", "multistep_arithmetic_two", "formal_fallacies"]


def main(out):
    items, report = [], {}
    def add(family, idx, built, extra=None):
        prompt, answer, scorer = built
        it = {"id": f"{family}-{idx}", "family": family, "prompt": prompt, "answer": str(answer), "scorer": scorer}
        it.update(extra or {})
        items.append(it)

    for family, ds, cfg, split, n, page, fn in FAMILIES:
        try:
            got = 0
            for idx, row in sample_rows(ds, cfg, split, n, page, f"{family}-v1"):
                try:
                    built = fn(row)
                except Exception:
                    continue
                if family.startswith("arc") and built[1] not in "ABCDE12345":
                    continue
                extra = {"level": row.get("level"), "subject": row.get("subject")} if family in ("math500", "mmlu") else None
                add(family, idx, built, extra)
                got += 1
            report[family] = got
        except Exception as e:
            report[family] = f"FAILED: {str(e)[:200]}"
    for group, names, fn in (("bbh_choice", BBH_CHOICE, bbh_choice), ("bbh_exact", BBH_EXACT, bbh_exact)):
        got = 0
        for name in names:
            try:
                for idx, row in sample_rows("SaylorTwift/bbh", name, "test", 12, 12, f"{name}-v1"):
                    built = fn(row)
                    if group == "bbh_choice" and built[1] not in "ABCDE":
                        continue
                    add(group, f"{name}-{idx}", built, {"task": name})
                    got += 1
            except Exception as e:
                report[f"{group}/{name}"] = f"FAILED: {str(e)[:200]}"
        report[group] = got
    try:
        got = 0
        for idx, row in sample_rows("google/IFEval", None, "train", 150, 25, "ifeval-v1"):
            items.append({"id": f"ifeval-{idx}", "family": "ifeval", "prompt": row["prompt"], "answer": "", "scorer": "ifeval",
                          "instruction_id_list": row["instruction_id_list"], "kwargs": row["kwargs"]})
            got += 1
        report["ifeval"] = got
    except Exception as e:
        report["ifeval"] = f"FAILED: {str(e)[:200]}"
    json.dump({"items": items}, open(out, "w"), ensure_ascii=False)
    print(json.dumps(report, indent=1))
    print("total items:", len(items))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "replica.json")
