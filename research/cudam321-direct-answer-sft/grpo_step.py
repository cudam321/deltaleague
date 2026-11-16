#!/usr/bin/env python3
"""One GRPO-style policy-improvement step on answers sampled under the test sampler.

  grpo_step.py rewards --items ROUND.json --gens GENS.jsonl --out REWARDS.jsonl
      CPU. Run it in the environment that has math_verify (venv-score). Reward = 1 when the answer finished
      on its own (finish_reason stop), closed its thinking, and passes the competition scorer.

  grpo_step.py train --ckpt IN_DIR --out OUT_DIR --items ROUND.json --gens GENS.jsonl --rewards REWARDS.jsonl
                     [--master-in M.safetensors] --master-out M.safetensors
      GPU (venv-vllm; stop the vLLM server first). Token-level PPO-clipped surrogate with per-question
      advantages. Weights are trained in float32 ("master") and written back to the checkpoint's own dtype
      by stochastic rounding, because the updates are smaller than one bfloat16 step.
"""
import argparse, collections, gc, hashlib, json, os, random, re, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
HEAD_CHUNK = 2048          # positions per output-head pass (keeps the vocabulary-sized logits small)
MAX_TOKENS = 8192.0        # the test's token cap; the length cost is measured against it


# ----------------------------------------------------------------------------- rewards and advantages
def load_score_answer():
    for d in (HERE, os.path.join(HERE, "..", "diag")):
        path = os.path.join(d, "score.py")
        if os.path.exists(path):
            ns = {}
            exec(open(path).read().split("def has_parsable(")[0], ns)
            return ns["score_answer"]
    sys.exit("score.py not found next to grpo_step.py")


def cmd_rewards(a):
    items = {it["id"]: it for it in json.load(open(a.items))["items"]}
    score_answer = load_score_answer()
    fam = collections.defaultdict(lambda: [0, 0.0])
    per_item = collections.defaultdict(list)
    n = errs = max_s = 0
    with open(a.out, "w") as out:
        for line in open(a.gens):
            r = json.loads(line)
            it = items.get(r["id"])
            if it is None:
                continue
            ok = 0.0
            if r["fr"] == "stop" and "</think>" in r["text"]:
                try:
                    ok = float(score_answer(it, r["text"]))
                except Exception:
                    errs += 1
            out.write(json.dumps({"id": r["id"], "s": r["s"], "correct": ok}) + "\n")
            fam[it["family"]][0] += 1
            fam[it["family"]][1] += ok
            per_item[r["id"]].append(ok)
            n += 1
            max_s = max(max_s, r["s"])
    for name in sorted(fam):
        ids = [i for i in per_item if items[i]["family"] == name]
        mixed = sum(0 < sum(per_item[i]) < len(per_item[i]) for i in ids)
        print(f"{name:16s} samples {fam[name][0]:6d}  reward {100 * fam[name][1] / fam[name][0]:5.1f}%  questions {len(ids):5d}  mixed {mixed:5d}")
    expected = len(items) * (max_s + 1)
    print(f"REWARDS samples {n} of {expected} expected, score_errors {errs}")
    if n < 0.9 * expected:
        sys.exit("REWARDS_INCOMPLETE: too many samples are missing (did the server die?)")


def advantages(groups, len_coef, eps=1e-6):
    """groups: {question id: [(sample key, correct 0/1, completion tokens)]} -> {sample key: advantage}.
    advantage = (reward - question mean) / (std of correctness + eps), reward = correct - len_coef * tokens / 8192.
    Questions whose samples are all right or all wrong carry no signal and are skipped; the length cost only
    tilts the advantage inside mixed questions (it is not allowed to become the whole signal)."""
    adv = {}
    for rows in groups.values():
        c = [x[1] for x in rows]
        if len(rows) < 2 or max(c) == min(c):
            continue
        mean_c = sum(c) / len(c)
        std = (sum((x - mean_c) ** 2 for x in c) / len(c)) ** 0.5
        r = [x[1] - len_coef * x[2] / MAX_TOKENS for x in rows]
        mean_r = sum(r) / len(r)
        for x, ri in zip(rows, r):
            adv[x[0]] = (ri - mean_r) / (std + eps)
    return adv


def pack(lengths, order, budget):
    """Token-budget batches over the indices in order, longest first: sequences * longest <= budget."""
    batches, cur = [], []
    for i in sorted(order, key=lambda i: -lengths[i]):
        if cur and (len(cur) + 1) * lengths[cur[0]] > budget:
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


# ----------------------------------------------------------------------------- precision and export
def stochastic_round_bf16(x, gen):
    """float32 -> bfloat16. A value between two bf16 neighbours goes to the farther-from-zero one with
    probability equal to its fractional position, so the mean is unbiased. Values already representable
    in bf16 (anything never trained) come out bit-identical."""
    import torch
    bits = x.contiguous().view(torch.int32)          # bf16 is the top 16 bits of float32
    low = bits & 0xFFFF
    up = torch.rand(x.shape, generator=gen) * 65536.0 < low.float()
    return ((bits & -65536) + up.to(torch.int32) * 65536).view(torch.float32).to(torch.bfloat16)


def same_bits(a, b):
    import torch
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


def master_matches(master, stored):
    """True when `stored` is what export() would write for `master`: each bf16 value is one of the master's two neighbours."""
    import torch
    if stored.dtype != torch.bfloat16:
        return same_bits(master.to(stored.dtype), stored)
    floor = master.contiguous().view(torch.int32) & -65536
    got = stored.float().view(torch.int32)
    return master.shape == stored.shape and bool(((got == floor) | (got == floor + 65536)).all())


def export(st, meta, new_f32, out_path, seed):
    """Write st with the tensors in new_f32 replaced, in each tensor's original dtype. Names, shapes, dtypes and
    metadata must come out identical to the input and every other tensor bit-identical; this is re-checked from disk."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    out, rep = dict(st), {"tensors_trained": len(new_f32), "tensors_changed": 0, "elements_changed": 0, "max_abs_delta": 0.0}
    for name, x in new_f32.items():
        old = st[name]
        x = x.detach().float().cpu()
        assert x.shape == old.shape, name
        assert bool(torch.isfinite(x).all()), f"non-finite values in {name}"
        if old.dtype == torch.bfloat16:
            gen = torch.Generator().manual_seed(int(hashlib.sha256(f"{seed}|{name}".encode()).hexdigest()[:8], 16))
            new = stochastic_round_bf16(x, gen)
            k = int((new.view(torch.int16) != old.view(torch.int16)).sum())
        elif old.dtype == torch.float32:
            new = x.clone()
            k = int((new.view(torch.int32) != old.view(torch.int32)).sum())
        else:
            raise ValueError(f"cannot export {name}: dtype {old.dtype}")
        if k:
            rep["tensors_changed"] += 1
            rep["elements_changed"] += k
            rep["max_abs_delta"] = max(rep["max_abs_delta"], float((new.float() - old.float()).abs().max()))
        out[name] = new
    save_file(out, out_path, metadata=meta)
    with safe_open(out_path, framework="pt") as f:
        assert f.metadata() == meta, "metadata changed"
        assert set(f.keys()) == set(st), "tensor names changed"
        for name in st:
            t = f.get_tensor(name)
            assert t.shape == st[name].shape and t.dtype == st[name].dtype, f"shape or dtype changed: {name}"
            assert same_bits(t, out[name]), f"written bytes differ: {name}"
            if name not in new_f32:
                assert same_bits(t, st[name]), f"untrained tensor changed: {name}"
    return rep


# ----------------------------------------------------------------------------- model
def load_model(path):
    import torch, transformers
    err = None
    for name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        try:   # load straight into float32: the file mixes bf16 and float32 tensors, and a bf16 load would degrade the latter
            return getattr(transformers, name).from_pretrained(path, dtype=torch.float32).float()
        except Exception as e:
            err = e
            print("LOAD_FALLBACK", name, repr(e)[:300], flush=True)
    raise err


def map_params(model, st, regex):
    """{safetensors key: model parameter} for every floating tensor whose key matches regex. Each pairing is proven by
    equal shape and bit-equal values, so a renamed or badly loaded model cannot be trained by mistake."""
    params = dict(model.named_parameters())

    def same(n, k):
        return params[n].shape == st[k].shape and same_bits(params[n].detach().cpu().to(st[k].dtype), st[k])

    mapping, used, missing = {}, set(), []
    for k in [k for k in st if re.search(regex, k) and st[k].is_floating_point()]:
        cands = [k] if k in params else [n for n in params if n.endswith("." + k) or k.endswith("." + n)]
        cands = [n for n in cands if n not in used and same(n, k)]
        if not cands:                                  # other naming: same last name part, shape and values
            tail = k.rsplit(".", 1)[-1]
            cands = [n for n in params if n not in used and n.rsplit(".", 1)[-1] == tail and same(n, k)]
        if len(cands) != 1:
            missing.append(k)
            continue
        mapping[k] = params[cands[0]]
        used.add(cands[0])
    if missing or not mapping:
        sys.exit(f"PARAM_MAP_FAILED {len(missing)} of {len(missing) + len(mapping)} tensors matching '{regex}' have no "
                 f"proven parameter, e.g. {missing[:5]}; first keys in the file: {list(st)[:5]}")
    head = getattr(model.get_output_embeddings(), "weight", None)
    for k in [k for k, p in mapping.items() if p is head]:
        twins = [o for o in st if o not in mapping and same_bits(st[o], st[k])]
        if twins:                                      # the file stores the tied head twice: keep both copies as they are
            print("FROZEN_TIED", k, "is tied to the output head, also stored as", twins, "- left untrained")
            del mapping[k]
    return mapping


class Policy:
    """Splits a causal LM into trunk (up to the final hidden states) and output head without relying on class
    internals: the head is swapped for a recorder during the trunk pass, so full-vocabulary logits are never built."""

    def __init__(self, model):
        import torch
        self.model, self.head = model, model.get_output_embeddings()
        if self.head is None:
            self.head = model.lm_head
        self.owner = next((m, n) for m in model.modules() for n, c in m.named_children() if c is self.head)

        class Recorder(torch.nn.Module):
            def forward(self, x):
                self.h = x
                return x[..., :1]

        self.rec = Recorder()

    def trunk(self, ids, mask):
        m, n = self.owner
        setattr(m, n, self.rec)
        try:
            self.model(input_ids=ids, attention_mask=mask, use_cache=False)
        finally:
            setattr(m, n, self.head)
        h, self.rec.h = self.rec.h, None
        return h


def forward_selected(policy, batch, dev):
    """Final hidden states at the positions that predict completion tokens, and those target tokens (sequence order)."""
    import torch
    T = max(len(s["ids"]) for s in batch)
    ids = torch.zeros(len(batch), T, dtype=torch.long)
    mask = torch.zeros(len(batch), T, dtype=torch.long)
    sel = torch.zeros(len(batch), T, dtype=torch.bool)
    for b, s in enumerate(batch):
        n = len(s["ids"])
        ids[b, :n] = torch.tensor(s["ids"])
        mask[b, :n] = 1
        sel[b, s["p"] - 1:n - 1] = True               # position t predicts token t + 1
    tgt = torch.roll(ids, -1, 1)[sel].to(dev)
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
        h = policy.trunk(ids.to(dev), mask.to(dev))
    return h[sel.to(dev)], tgt


def head_logp(policy, h, tgt):
    import torch
    with torch.autocast(h.device.type, enabled=False):    # the head runs in float32 so small ratio changes are visible
        return torch.log_softmax(policy.head(h.float()), -1).gather(1, tgt[:, None])[:, 0]


def ppo_terms(lp, old, adv, clip, neg_weight):
    """Summed clipped-surrogate loss over tokens, and the per-token probability ratios."""
    import torch
    ratio = torch.exp(lp - old)
    obj = torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
    w = torch.where(adv > 0, torch.ones_like(adv), torch.full_like(adv, neg_weight))
    return -(w * obj).sum(), ratio


def old_logps(policy, seqs, budget, dev):
    """Per-token log-probs of every completion under the starting weights, stored on each sequence."""
    import torch
    lengths = [len(s["ids"]) for s in seqs]
    with torch.no_grad():
        for b in pack(lengths, range(len(seqs)), budget):
            batch = [seqs[i] for i in b]
            hs, tgt = forward_selected(policy, batch, dev)
            lp = torch.cat([head_logp(policy, hs[i:i + HEAD_CHUNK], tgt[i:i + HEAD_CHUNK])
                            for i in range(0, len(tgt), HEAD_CHUNK)]).float().cpu()
            o = 0
            for s in batch:
                n = len(s["ids"]) - s["p"]
                s["old"] = lp[o:o + n]
                o += n


def run_step(policy, seqs, group, budget, a, dev):
    """Accumulate the gradient of one optimizer step (token-mean loss over the group's sequences). Returns sums."""
    import torch
    lengths = {i: len(seqs[i]["ids"]) for i in group}
    norm = sum(len(seqs[i]["ids"]) - seqs[i]["p"] for i in group)
    tot = {"n": norm, "loss": 0.0, "ratio": 0.0, "clip": 0.0, "kl": 0.0}
    for b in pack(lengths, group, budget):
        batch = [seqs[i] for i in b]
        hs, tgt = forward_selected(policy, batch, dev)
        old = torch.cat([s["old"] for s in batch]).to(dev)
        adv = torch.cat([torch.full((len(s["ids"]) - s["p"],), s["adv"]) for s in batch]).to(dev)
        hd = hs.detach().float().requires_grad_(True)    # head in chunks; its gradient is handed to the trunk once
        for i in range(0, len(tgt), HEAD_CHUNK):
            j = slice(i, i + HEAD_CHUNK)
            loss, ratio = ppo_terms(head_logp(policy, hd[j], tgt[j]), old[j], adv[j], a.clip, a.neg_weight)
            (loss / norm).backward()
            with torch.no_grad():
                tot["loss"] += float(loss)
                tot["ratio"] += float(ratio.sum())
                tot["clip"] += float(((ratio - 1).abs() > a.clip).sum())
                tot["kl"] += float((ratio - 1 - torch.log(ratio)).sum())
        hs.backward(hd.grad.to(hs.dtype))
    return tot


def train_loop(policy, params, seqs, a, dev, log=print):
    """K epochs of clipped-surrogate steps over seqs ({ids, p, adv}). Returns the number of optimizer steps taken."""
    import torch
    oom = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    rng, budget, step, t0 = random.Random(a.seed), a.batch_tokens, 0, time.time()
    old_logps(policy, seqs, budget, dev)
    log(f"old log-probs done, {time.time() - t0:.0f}s")
    for ep in range(a.epochs):
        order = list(range(len(seqs)))
        rng.shuffle(order)
        groups = [order[i:i + a.seqs_per_step] for i in range(0, len(order), a.seqs_per_step)]
        if len(groups) > 1 and len(groups[-1]) < a.seqs_per_step // 2:
            tail = groups.pop()                               # fold a small last group into the one before it
            groups[-1] += tail
        for g in groups:
            tot = None
            while tot is None:
                try:
                    tot = run_step(policy, seqs, g, budget, a, dev)
                except oom:
                    pass
                if tot is None:                           # out of memory: drop the partial gradient, halve, retry
                    opt.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    if budget <= a.max_seq_tokens:
                        raise RuntimeError("out of GPU memory even with one sequence per batch")
                    budget = max(a.max_seq_tokens, budget // 2)
                    log(f"OOM_HALVE batch_tokens={budget}")
            kl = tot["kl"] / tot["n"]
            if kl > a.max_kl:
                opt.zero_grad(set_to_none=True)
                log(f"EARLY_STOP approx_kl {kl:.5f} > {a.max_kl} before step {step + 1}")
                return step
            gn = float(torch.nn.utils.clip_grad_norm_(params, a.grad_clip))
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            log(f"STEP {step} epoch {ep + 1} seqs {len(g)} tokens {tot['n']} loss {tot['loss'] / tot['n']:.5f} "
                f"ratio {tot['ratio'] / tot['n']:.4f} clipfrac {tot['clip'] / tot['n']:.4f} kl {kl:.5f} "
                f"gradnorm {gn:.3f} t {time.time() - t0:.0f}s")
            if a.max_steps and step >= a.max_steps:
                return step
    return step


def cmd_train(a):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer
    assert os.path.abspath(a.ckpt) != os.path.abspath(a.out), "--out must be a new directory"
    weights = os.path.join(a.ckpt, "model.safetensors")
    assert os.path.exists(weights), f"no {weights}"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log = lambda *x: print(*x, flush=True)

    # 1. advantages
    items = {it["id"]: it for it in json.load(open(a.items))["items"]}
    gens = {}
    for line in open(a.gens):
        r = json.loads(line)
        gens[(r["id"], r["s"])] = r
    groups = collections.defaultdict(list)
    for line in open(a.rewards):
        r = json.loads(line)
        k = (r["id"], r["s"])
        if k in gens and r["id"] in items:
            groups[r["id"]].append((k, r["correct"], gens[k]["ct"]))
    adv = advantages(groups, a.len_coef)
    used_groups = len({k[0] for k in adv})
    if a.neg_weight == 0:
        adv = {k: v for k, v in adv.items() if v > 0}
    if not adv:
        sys.exit("NO_TRAINABLE_GROUPS: no question has both right and wrong samples")

    # 2. sequences = chat-template prompt + completion (+ end-of-turn token when the model stopped by itself)
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    eot = tok.convert_tokens_to_ids("<|im_end|>")
    if eot is None or eot == tok.unk_token_id:
        eot = tok.eos_token_id
    prompts, seqs, fam, skipped, drift = {}, [], collections.Counter(), 0, 0
    for (iid, s), v in sorted(adv.items()):
        if iid not in prompts:
            text = tok.apply_chat_template([{"role": "user", "content": items[iid]["prompt"]}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True)   # as run_eval.py
            prompts[iid] = tok(text)["input_ids"]
        r = gens[(iid, s)]
        comp = tok(r["text"], add_special_tokens=False)["input_ids"] + ([eot] if r["fr"] == "stop" else [])
        drift += abs(len(comp) - r["ct"]) > 2
        ids = (prompts[iid] + comp)[:a.max_seq_tokens]
        if len(ids) <= len(prompts[iid]):
            skipped += 1
            continue
        seqs.append({"ids": ids, "p": len(prompts[iid]), "adv": float(v)})
        fam[items[iid]["family"]] += 1
    log(f"questions {len(groups)}, with mixed outcomes {used_groups}; sequences {len(seqs)} "
        f"(+{sum(s['adv'] > 0 for s in seqs)} / -{sum(s['adv'] < 0 for s in seqs)}), skipped {skipped}, "
        f"completion tokens {sum(len(s['ids']) - s['p'] for s in seqs)}, end-of-turn id {eot}, "
        f"re-tokenized length differs from the server's count in {drift} sequences")
    log("sequences by family:", json.dumps(dict(fam)))

    # 3. model, parameter mapping, float32 master
    st = load_file(weights)
    with safe_open(weights, framework="pt") as f:
        meta = f.metadata()
    model = load_model(a.ckpt)
    mapping = map_params(model, st, a.train_regex)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in mapping.values():
        p.requires_grad_(True)
    if a.master_in:
        master = load_file(a.master_in)
        assert set(master) == set(mapping), "master file covers a different set of tensors"
        for k, p in mapping.items():
            assert master_matches(master[k], st[k]), f"master does not belong to this checkpoint: {k}"
            p.data.copy_(master[k])
        del master
    log(f"training {len(mapping)} of {len(st)} tensors ({sum(p.numel() for p in mapping.values()) / 1e6:.0f}M values), "
        f"master {'loaded' if a.master_in else 'started from the checkpoint'}, device {dev}")
    model.to(dev).eval()

    # 4. train, 5. export
    steps = train_loop(Policy(model), list(mapping.values()), seqs, a, dev, log)
    new = {k: p.detach().float().cpu() for k, p in mapping.items()}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    shutil.copytree(a.ckpt, a.out, dirs_exist_ok=True,
                    ignore=lambda d, names: ["model.safetensors"] if os.path.abspath(d) == os.path.abspath(a.ckpt) else [])
    rep = export(st, meta, new, os.path.join(a.out, "model.safetensors"), a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.master_out)), exist_ok=True)
    save_file(new, a.master_out)
    log("EXPORT", json.dumps(rep), "names/shapes/dtypes/metadata identical: yes")
    log(f"TRAIN_DONE steps={steps}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rewards")
    t = sub.add_parser("train")
    for p in (r, t):
        p.add_argument("--items", required=True)
        p.add_argument("--gens", required=True)
    r.add_argument("--out", required=True)
    t.add_argument("--rewards", required=True)
    t.add_argument("--ckpt", required=True, help="input checkpoint directory (model.safetensors + config + tokenizer)")
    t.add_argument("--out", required=True, help="output checkpoint directory")
    t.add_argument("--master-in", default="", help="float32 master weights from the previous round")
    t.add_argument("--master-out", required=True)
    t.add_argument("--train-regex", default="language_model", help="train tensors whose safetensors key matches this")
    t.add_argument("--lr", type=float, default=2e-6)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--clip", type=float, default=0.2)
    t.add_argument("--neg-weight", type=float, default=1.0, help="scale on negative-advantage terms; 0 = positives only")
    t.add_argument("--len-coef", type=float, default=0.1, help="reward cost per 8,192 completion tokens")
    t.add_argument("--max-seq-tokens", type=int, default=4096)
    t.add_argument("--batch-tokens", type=int, default=16000)
    t.add_argument("--seqs-per-step", type=int, default=64)
    t.add_argument("--max-kl", type=float, default=0.02)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--max-steps", type=int, default=0, help="0 = no limit (use 1 or 2 for a smoke test)")
    t.add_argument("--seed", type=int, default=0, help="shuffling and rounding seed; keep it fixed across rounds")
    a = ap.parse_args()
    (cmd_rewards if a.cmd == "rewards" else cmd_train)(a)


if __name__ == "__main__":
    main()
