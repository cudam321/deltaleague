#!/usr/bin/env python3
"""Build supervised training rows from public TRAIN data only (venv-score: needs `datasets`).

Each output row: {"id", "family", "kind", "prompt", "completion"}. `completion` is the text that follows the chat
template's opening "<think>\\n"; the trainer adds the end-of-turn token. Every completion ends in the one answer
shape all scorers accept: a bare letter, word or plain number, or \\boxed{...} for a LaTeX expression.

kinds
  direct  empty thinking, then the answer          --direct fam,fam (items from pool.json)
  trace   the model's own correct sample: its thinking kept, the text after </think> replaced by the answer
          (instruction-following items keep the model's own answer text: the checkers passed it as written)
          --trace-items ITEMS.json --gens GENS.jsonl --rewards REWARDS.jsonl (files of a sampling pass)
  solved  a public dataset's worked solution as the thinking, then the answer
          --math-train N  --gsm8k-train N  --omi2 N        (N = -1 takes all)
  exact   the "give only the final answer" wording with a bare answer
          --exact-boolq N (passage + question -> yes/no)  --exact-gsm8k N

A question is dropped when its text is also in an eval split of the same benchmark (MATH test, MATH-500, GSM8K test,
BoolQ validation); pool.json items were already filtered that way by build_pool.py.
"""
import argparse, collections, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "diag")]
import build_pool as bp
import build_replica as br

CLOSE = "\n</think>\n\n"
MATH_SOURCE = "EleutherAI/hendrycks_math"


def final(answer, scorer):
    """Bare letters, words and plain numbers; \\boxed{} for anything else under the math scorer."""
    a = str(answer).strip()
    if scorer != "math":
        return a
    plain = a.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", a) else a
    return plain if re.fullmatch(r"-?\d+(\.\d+)?", plain) else "\\boxed{" + a + "}"


def thinking_of(text):
    """The thinking part of a generated answer, without the opening tag. None unless there is exactly one </think>."""
    if text.count("</think>") != 1:
        return None
    t = text.split("</think>")[0].strip()
    if t.startswith("<think>"):
        t = t[len("<think>"):].strip()
    return t or None


def direct(it):
    return {"id": it["id"] + "-d", "family": it["family"], "kind": "direct", "prompt": it["prompt"],
            "completion": CLOSE + final(it["answer"], it["scorer"])}


def trace_rows(items, gens_path, rewards_path, per_item, max_tokens):
    """items: {id: item}. Keeps, per question, the shortest correct samples that finished on their own."""
    ok = set()
    for line in open(rewards_path):
        r = json.loads(line)
        if r["correct"] >= 1:
            ok.add((r["id"], r["s"]))
    best = collections.defaultdict(list)
    for line in open(gens_path):
        r = json.loads(line)
        if (r["id"], r["s"]) in ok and r["fr"] == "stop" and r["ct"] <= max_tokens and r["id"] in items:
            th = thinking_of(r["text"])
            if th:
                best[r["id"]].append((r["ct"], r["s"], th, r["text"].split("</think>")[1].strip()))
    rows = []
    for iid in sorted(best):
        it = items[iid]
        for ct, s, th, own in sorted(best[iid])[:per_item]:
            ans = own if it["scorer"] == "ifeval" else final(it["answer"], it["scorer"])   # free-form answers stay as written
            rows.append({"id": f"{iid}-t{s}", "family": it["family"], "kind": "trace", "prompt": it["prompt"],
                         "completion": th + CLOSE + ans})
    return rows


def solved(rid, family, problem, think, answer, suffix=br.SUFFIX, kind="solved"):
    problem, think, answer = str(problem or "").strip(), str(think or "").strip(), str(answer or "").strip()
    if not problem or not think or not answer or "</think>" in think:
        return None
    return {"id": rid, "family": family, "kind": kind, "prompt": problem + suffix,
            "completion": think + CLOSE + final(answer, "math")}


def gsm_steps(answer_field):
    """GSM8K's answer field -> (worked steps without the <<calculator>> marks, final number)."""
    body, _, ans = answer_field.partition("####")
    return re.sub(r"<<[^>]*>>", "", body).strip(), ans.strip().replace(",", "")


def eval_keys(dataset, config, column):
    keys = set()
    for split in bp.eval_splits(dataset, config):
        keys.update(bp.key(x) for x in bp.load(dataset, config, split)[column])
    keys.discard("")
    return keys


def take(n, seq):
    seq = list(seq)
    return seq if n < 0 else seq[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", default="pool.json")
    ap.add_argument("--direct", default="", help="comma-separated families from the pool, answered with empty thinking")
    ap.add_argument("--direct-cap", type=int, default=20000, help="rows per direct family")
    ap.add_argument("--trace-items", default="")
    ap.add_argument("--gens", default="")
    ap.add_argument("--rewards", default="")
    ap.add_argument("--traces-per-item", type=int, default=1)
    ap.add_argument("--trace-max-tokens", type=int, default=3000)
    ap.add_argument("--math-train", type=int, default=0)
    ap.add_argument("--gsm8k-train", type=int, default=0)
    ap.add_argument("--omi2", type=int, default=0)
    ap.add_argument("--omi2-split", default="train_1M")
    ap.add_argument("--omi2-max-chars", type=int, default=2000)
    ap.add_argument("--exact-boolq", type=int, default=0)
    ap.add_argument("--exact-gsm8k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []

    if a.direct:
        fams = set(a.direct.split(","))
        by = collections.defaultdict(list)
        for it in json.load(open(a.pool))["items"]:
            if it["family"] in fams:
                by[it["family"]].append(it)
        missing = fams - set(by)
        assert not missing, f"families not in the pool: {sorted(missing)}"
        for fam in sorted(by):
            rows += [direct(it) for it in by[fam][:a.direct_cap]]     # pool order is already a seeded shuffle

    if a.trace_items:
        items = {it["id"]: it for it in json.load(open(a.trace_items))["items"]}
        rows += trace_rows(items, a.gens, a.rewards, a.traces_per_item, a.trace_max_tokens)

    if a.math_train or a.gsm8k_train or a.omi2 or a.exact_gsm8k:
        math_keys = set()
        for cfg in bp.MATH_CONFIGS:
            math_keys |= eval_keys(MATH_SOURCE, cfg, "problem")
        math_keys |= eval_keys("HuggingFaceH4/MATH-500", "default", "problem")
        math_keys |= eval_keys("openai/gsm8k", "main", "question")
        print("math and GSM8K eval question keys:", len(math_keys))
        seen = set()

        def keep(problem):
            k = bp.key(problem)
            if not k or k in math_keys or k in seen:
                return False
            seen.add(k)
            return True

        if a.math_train:
            got = []
            for cfg in bp.MATH_CONFIGS:
                ds = bp.load(MATH_SOURCE, cfg, "train")
                for i in range(len(ds)):
                    r = ds[i]
                    ans = bp.last_boxed(r["solution"])
                    if ans and keep(r["problem"]):
                        got.append(solved(f"math-tr-{cfg}-{i}", "math_train", r["problem"], r["solution"], ans))
            got = [x for x in got if x]
            rng.shuffle(got)
            rows += take(a.math_train, got)

        if a.gsm8k_train or a.exact_gsm8k:
            ds = bp.load("openai/gsm8k", "main", "train")
            idx = list(range(len(ds)))
            rng.shuffle(idx)
            n_solved = len(idx) if a.gsm8k_train < 0 else a.gsm8k_train
            n_exact = len(idx) if a.exact_gsm8k < 0 else a.exact_gsm8k
            for j, i in enumerate(idx):
                r = ds[i]
                steps, ans = gsm_steps(r["answer"])
                if bp.key(r["question"]) in math_keys:
                    continue
                if j < n_solved and keep(r["question"]):
                    rows.append(solved(f"gsm8k-tr-{i}", "gsm8k", r["question"], steps, ans))
                if j >= len(idx) - n_exact:                 # the exact wording uses the other end of the shuffle
                    rows.append(solved(f"gsm8k-ex-{i}", "gsm8k", r["question"], steps, ans, br.EXACT_SUFFIX, "exact"))

        if a.omi2:
            from datasets import load_dataset
            ds = load_dataset("nvidia/OpenMathInstruct-2", split=a.omi2_split, streaming=True)
            n, scanned = 0, 0
            for r in ds:
                scanned += 1
                sol = r.get("generated_solution") or ""
                if len(sol) > a.omi2_max_chars or not keep(r.get("problem", "")):
                    continue
                row = solved(f"omi2-{scanned}", "omi2", r["problem"], sol, r.get("expected_answer", ""))
                if row:
                    rows.append(row)
                    n += 1
                    if 0 <= a.omi2 <= n:
                        break
            print(f"OpenMathInstruct-2 ({a.omi2_split}): scanned {scanned}, kept {n}")

    if a.exact_boolq:
        held = eval_keys("aps/super_glue", "boolq", "question")
        ds = bp.load("aps/super_glue", "boolq", "train")
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        n = 0
        for i in idx:
            r = ds[i]
            q = r["question"].strip()
            if not q or bp.key(q) in held:
                continue
            q = q[0].upper() + q[1:] + ("" if q.endswith("?") else "?")
            rows.append({"id": f"boolq-ex-{i}", "family": "boolq", "kind": "exact",
                         "prompt": r["passage"] + "\n" + q + br.EXACT_SUFFIX,
                         "completion": CLOSE + ("yes" if r["label"] else "no")})
            n += 1
            if 0 <= a.exact_boolq <= n:
                break

    rows = [r for r in rows if r]
    assert rows, "no rows built"
    assert len({r["id"] for r in rows}) == len(rows), "duplicate row ids"
    rng.shuffle(rows)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stat = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        s = stat[r["kind"] + "/" + r["family"]]
        s[0] += 1
        s[1] += len(r["prompt"]) + len(r["completion"])
    for k in sorted(stat):
        print(f"{k:28s} rows {stat[k][0]:7d}  mean chars {stat[k][1] / stat[k][0]:7.0f}")
    print("SFT_DATA_DONE rows", len(rows), "chars", sum(s[1] for s in stat.values()))


if __name__ == "__main__":
    main()
