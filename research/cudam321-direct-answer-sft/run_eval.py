#!/usr/bin/env python3
"""Generate answers for the replica against a local vLLM server, the way the evaluator does:
render the checkpoint's chat template ourselves, call /v1/completions, keep the raw text."""
import argparse, asyncio, hashlib, json, os, time
from openai import AsyncOpenAI
from transformers import AutoTokenizer

TEST = dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5)
MODES = {
    "sample": dict(think=True, **TEST),            # the hidden test's settings
    "nothink": dict(think=False, **TEST),          # same sampler, thinking closed up front
    "greedy_pp": dict(think=True, temperature=0.0, presence_penalty=1.5),
    "greedy": dict(think=True, temperature=0.0, presence_penalty=0.0),
}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--replica", default="replica.json")
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--every", type=int, default=1, help="use every Nth item")
    ap.add_argument("--families", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=8192)
    a = ap.parse_args()

    mode = dict(MODES[a.mode])
    think = mode.pop("think")
    top_k = mode.pop("top_k", None)
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    items = json.load(open(a.replica))["items"]
    if a.families:
        keep = set(a.families.split(","))
        items = [it for it in items if it["family"] in keep]
    items = items[:: a.every]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
                done.add((r["id"], r["s"]))
            except Exception:
                pass
    jobs = [(it, s) for it in items for s in range(a.samples) if (it["id"], s) not in done]
    print(f"mode={a.mode} items={len(items)} samples={a.samples} todo={len(jobs)} already={len(done)}", flush=True)

    client = AsyncOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="x", timeout=3600, max_retries=2)
    sem = asyncio.Semaphore(a.concurrency)
    out = open(a.out, "a")
    stats = {"n": 0, "tok": 0, "t0": time.time(), "err": 0}

    async def one(it, s):
        prompt = tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=think)
        seed = int(hashlib.sha256(f"{it['id']}|{s}|{a.mode}".encode()).hexdigest()[:8], 16)
        extra = {"top_k": top_k} if top_k else {}
        async with sem:
            try:
                r = await client.completions.create(model="m", prompt=prompt, max_tokens=a.max_tokens, seed=seed,
                                                    extra_body=extra, **mode)
                c = r.choices[0]
                rec = {"id": it["id"], "s": s, "fr": c.finish_reason, "ct": r.usage.completion_tokens, "text": c.text}
            except Exception as e:
                stats["err"] += 1
                rec = {"id": it["id"], "s": s, "fr": "error", "ct": 0, "text": "", "error": str(e)[:300]}
                if stats["err"] <= 3:
                    print("ERROR", rec["error"], flush=True)
        if rec["fr"] != "error":
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
        stats["n"] += 1
        stats["tok"] += rec["ct"]
        if stats["n"] % 100 == 0 or stats["n"] == len(jobs):
            dt = time.time() - stats["t0"]
            print(f"{stats['n']}/{len(jobs)} done, {stats['tok']/dt:.0f} tok/s, {dt:.0f}s, errors={stats['err']}", flush=True)

    await asyncio.gather(*(one(it, s) for it, s in jobs))
    print("FINISHED", a.mode, flush=True)


asyncio.run(main())
