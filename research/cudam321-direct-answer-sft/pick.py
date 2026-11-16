#!/usr/bin/env python3
"""Take the first N items of chosen families from a pool file (pool order is already a seeded shuffle).
usage: pick.py --pool pool.json --out items.json --take arc_challenge=1200,gsm8k=2500 [--skip N]"""
import argparse, collections, json


def pick(items, want, skip=0):
    got, seen = [], collections.Counter()
    for it in items:
        fam = it["family"]
        if fam in want:
            seen[fam] += 1
            if skip < seen[fam] <= skip + want[fam]:
                got.append(it)
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--take", required=True)
    ap.add_argument("--skip", type=int, default=0, help="skip the first N items of each family")
    a = ap.parse_args()
    want = {k: int(v) for k, v in (x.split("=") for x in a.take.split(","))}
    got = pick(json.load(open(a.pool))["items"], want, a.skip)
    json.dump({"items": got}, open(a.out, "w"), ensure_ascii=False)
    print("PICKED", len(got), json.dumps(collections.Counter(it["family"] for it in got)))


if __name__ == "__main__":
    main()
