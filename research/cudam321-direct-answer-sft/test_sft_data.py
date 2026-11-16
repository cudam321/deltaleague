#!/usr/bin/env python3
"""CPU checks of the data-building and scheduling logic (no datasets, no torch). usage: python3 test_sft_data.py"""
import json, os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pick, sft_data as d


def test_final():
    assert d.final("C", "choice") == "C" and d.final("yes", "exact") == "yes"
    assert d.final("18", "math") == "18" and d.final("-3.5", "math") == "-3.5" and d.final("1,250", "math") == "1250"
    assert d.final("\\frac{3}{4}", "math") == "\\boxed{\\frac{3}{4}}"
    assert d.final("(1,2)", "math") == "\\boxed{(1,2)}" and d.final("10\\%", "math") == "\\boxed{10\\%}"


def test_thinking_of():
    assert d.thinking_of("<think>\nstep one\n</think>\n\nThe answer is B") == "step one"
    assert d.thinking_of("step one\n</think>\n\nB") == "step one"
    assert d.thinking_of("no close tag") is None and d.thinking_of("a</think>b</think>c") is None
    assert d.thinking_of("\n</think>\n\nB") is None


def test_rows():
    it = {"id": "x-1", "family": "piqa", "prompt": "Q?\nA. a\nB. b", "answer": "B", "scorer": "choice"}
    r = d.direct(it)
    assert r["completion"] == "\n</think>\n\nB" and r["kind"] == "direct" and r["id"] == "x-1-d"
    s = d.solved("m-1", "math_train", "What is 3/4 of 1?", "It is three quarters.", "\\frac{3}{4}")
    assert s["completion"] == "It is three quarters.\n</think>\n\n\\boxed{\\frac{3}{4}}" and s["prompt"].endswith("\\boxed{}.")
    assert d.solved("m-2", "math_train", "Q", "", "1") is None and d.solved("m-3", "math_train", "Q", "a</think>b", "1") is None
    assert d.gsm_steps("He has 2+3=<<2+3=5>>5 apples.\n#### 1,005") == ("He has 2+3=5 apples.", "1005")


def test_traces():
    tmp = tempfile.mkdtemp(prefix="sft-test-")
    items = {"q1": {"id": "q1", "family": "gsm8k", "prompt": "P", "answer": "7", "scorer": "math"},
             "q2": {"id": "q2", "family": "mmlu", "prompt": "P2", "answer": "D", "scorer": "choice"}}
    gens = [{"id": "q1", "s": 0, "fr": "stop", "ct": 50, "text": "long way\n</think>\n\nThe answer is 7."},
            {"id": "q1", "s": 1, "fr": "stop", "ct": 20, "text": "short way\n</think>\n\n7"},
            {"id": "q1", "s": 2, "fr": "length", "ct": 10, "text": "cut\n</think>\n\n7"},
            {"id": "q2", "s": 0, "fr": "stop", "ct": 30, "text": "wrong\n</think>\n\nA"},
            {"id": "q2", "s": 1, "fr": "stop", "ct": 9000, "text": "too long\n</think>\n\nD"},
            {"id": "zz", "s": 0, "fr": "stop", "ct": 5, "text": "not an item\n</think>\n\n1"}]
    rewards = [{"id": "q1", "s": 0, "correct": 1.0}, {"id": "q1", "s": 1, "correct": 1.0}, {"id": "q1", "s": 2, "correct": 1.0},
               {"id": "q2", "s": 0, "correct": 0.0}, {"id": "q2", "s": 1, "correct": 1.0}, {"id": "zz", "s": 0, "correct": 1.0}]
    gp, rp = os.path.join(tmp, "g.jsonl"), os.path.join(tmp, "r.jsonl")
    open(gp, "w").write("\n".join(map(json.dumps, gens)) + "\n")
    open(rp, "w").write("\n".join(map(json.dumps, rewards)) + "\n")
    rows = d.trace_rows(items, gp, rp, per_item=1, max_tokens=3000)
    assert [r["id"] for r in rows] == ["q1-t1"], rows                       # shortest correct finished sample only
    assert rows[0]["completion"] == "short way\n</think>\n\n7"
    assert len(d.trace_rows(items, gp, rp, per_item=2, max_tokens=3000)) == 2


def test_pick():
    items = [{"id": i, "family": "a" if i % 2 else "b"} for i in range(20)]
    got = pick.pick(items, {"a": 3}, skip=2)
    assert [it["id"] for it in got] == [5, 7, 9]
    assert len(pick.pick(items, {"a": 100, "b": 1})) == 11


for name, fn in sorted(globals().items()):
    if name.startswith("test_"):
        fn()
        print("ok", name)
print("ALL_OK")
