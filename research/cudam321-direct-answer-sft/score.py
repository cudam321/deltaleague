#!/usr/bin/env python3
"""Score generations with the competition's public scorer and break failures down by task family.
score_answer below is copied from delta_cli/scoring.py (the scorer the platform says it uses on private tasks)."""
import json, re, statistics, sys
from collections import defaultdict


def score_answer(example, text):
    answer = text.split("</think>")[-1].strip()
    scorer = example.get("scorer", "exact")
    if scorer == "exact":
        norm = lambda s: re.sub(r"\s+", " ", str(s).strip()).casefold()
        return float(norm(answer) == norm(example["answer"]))
    if scorer == "choice":
        boxed = re.findall(r"\\boxed\{([A-E1-5])\}", answer)
        matches = boxed or re.findall(r"(?:^|\n)\s*([A-E1-5])[.)]?\s*$", answer)
        return float(bool(matches) and matches[-1] == str(example["answer"]))
    if scorer == "math":
        from math_verify import parse, verify
        from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
        cfg = [LatexExtractionConfig(), ExprExtractionConfig()]
        gold = parse(str(example["answer"]), extraction_config=cfg)
        pred = parse(answer, extraction_config=cfg)
        return float(bool(gold) and bool(pred) and verify(gold, pred))
    if scorer == "ifeval":
        from lm_eval.tasks.ifeval.instructions_registry import INSTRUCTION_DICT
        passed = []
        for instruction_id, kwargs in zip(example["instruction_id_list"], example["kwargs"]):
            instruction = INSTRUCTION_DICT[instruction_id](instruction_id)
            instruction.build_description(**{k: v for k, v in kwargs.items() if v is not None})
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=example["prompt"])
            passed.append(bool(answer) and instruction.check_following(answer))
        return float(bool(passed) and all(passed))
    raise ValueError("unknown_scorer")


def has_parsable(example, answer):
    s = example["scorer"]
    if s == "choice":
        return bool(re.findall(r"\\boxed\{([A-E1-5])\}", answer) or re.findall(r"(?:^|\n)\s*([A-E1-5])[.)]?\s*$", answer))
    if s == "math":
        return "\\boxed" in answer or bool(re.search(r"\d", answer))
    return bool(answer.strip())


def main(replica, gens, out):
    items = {it["id"]: it for it in json.load(open(replica))["items"]}
    per_item = defaultdict(list)
    fam = defaultdict(lambda: defaultdict(float))
    toks = defaultdict(list)
    errs = 0
    for line in open(gens):
        r = json.loads(line)
        it = items.get(r["id"])
        if not it:
            continue
        text = r["text"]
        closed = "</think>" in text
        cut = r["fr"] == "length"
        try:
            raw = score_answer(it, text)                      # platform: raw output, last segment
        except Exception as e:
            raw = 0.0
            errs += 1
            if errs <= 3:
                print("SCORE_ERROR", it["scorer"], repr(e)[:200], file=sys.stderr)
        strict = raw if closed else 0.0                       # public evaluator: no </think> means empty answer
        f = fam[it["family"]]
        f["n"] += 1
        f["pass_raw"] += raw
        f["pass_strict"] += strict
        f["cut"] += cut
        f["cut_pass_raw"] += cut and raw
        f["unclosed_not_cut"] += (not closed) and (not cut)
        f["two_closes"] += text.count("</think>") > 1
        if closed and not cut and not raw:
            ans = text.split("</think>")[-1].strip()
            f["answered_format_fail" if not has_parsable(it, ans) else "answered_wrong"] += 1
        toks[it["family"]].append(r["ct"])
        per_item[r["id"]].append(raw)
    rows = {}
    for name, f in sorted(fam.items()):
        n = f["n"]
        ids = [i for i in per_item if items[i]["family"] == name]
        rates = [sum(per_item[i]) / len(per_item[i]) for i in ids]
        multi = [r for i, r in zip(ids, rates) if len(per_item[i]) > 1]
        rows[name] = {
            "n": int(n), "items": len(ids),
            "pass_raw": round(100 * f["pass_raw"] / n, 1), "pass_strict": round(100 * f["pass_strict"] / n, 1),
            "cut": round(100 * f["cut"] / n, 1), "cut_but_pass_raw": round(100 * f["cut_pass_raw"] / n, 1),
            "wrong": round(100 * f["answered_wrong"] / n, 1), "format_fail": round(100 * f["answered_format_fail"] / n, 1),
            "unclosed_not_cut": round(100 * f["unclosed_not_cut"] / n, 1), "two_closes": round(100 * f["two_closes"] / n, 1),
            "tok_mean": round(statistics.mean(toks[name])), "tok_median": round(statistics.median(toks[name])),
            "always": round(100 * sum(r == 1 for r in multi) / len(multi), 1) if multi else None,
            "never": round(100 * sum(r == 0 for r in multi) / len(multi), 1) if multi else None,
            "sometimes": round(100 * sum(0 < r < 1 for r in multi) / len(multi), 1) if multi else None,
        }
    json.dump({"families": rows, "per_item": {i: per_item[i] for i in per_item}, "score_errors": errs}, open(out, "w"))
    cols = ["n", "pass_raw", "pass_strict", "cut", "cut_but_pass_raw", "wrong", "format_fail", "unclosed_not_cut", "two_closes", "tok_mean", "tok_median", "always", "sometimes", "never"]
    print("family".ljust(16) + " ".join(c[:11].rjust(11) for c in cols))
    for name, r in rows.items():
        print(name.ljust(16) + " ".join(str(r[c]).rjust(11) for c in cols))
    print("score_errors:", errs)


if __name__ == "__main__":
    main(*sys.argv[1:4])
