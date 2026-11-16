#!/usr/bin/env python3
"""Supervised fine-tuning on rows from sft_data.py, plus three checkpoint tools.

  sft.py train --ckpt IN_DIR --out OUT_DIR --data ROWS.jsonl [--lr 5e-6 --epochs 1 ...]
      GPU (venv-vllm; stop the vLLM server first). Cross-entropy on the completion tokens only. Weights are trained
      in float32 and written back in the checkpoint's own dtypes by stochastic rounding (grpo_step.export).
  sft.py mix --a DIR --b DIR --alpha X --out DIR
      CPU. (1 - X) * a + X * b on every floating tensor that differs between the two, written in a's dtypes.
  sft.py scale --ckpt DIR --out DIR --key NAME --factor F
      CPU. Multiply one tensor by F (the final norm weight rescales all logits).
  sft.py keys --ckpt DIR [--grep REGEX]
      CPU. List tensors (name, shape, dtype, mean) whose name matches REGEX.
"""
import argparse, collections, gc, json, math, os, random, re, shutil, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grpo_step as g


def lr_at(step, total, peak, warmup, floor):
    """Linear warm-up over `warmup` steps, then cosine from peak down to floor * peak. step counts from 0."""
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def bucket_of(n):
    """Padded length for a sequence of n tokens. Few distinct lengths, so the kernels and the allocator see few shapes."""
    step = 32 if n <= 256 else 64 if n <= 512 else 128
    return -(-n // step) * step


def plan_steps(lengths, batch_tokens, batches_per_step, epochs, seed):
    """Optimizer steps, each a list of batches (padded length, row indices). Rows are grouped by padded length and
    every batch of a group holds the same number of rows (the last may be smaller): varying shapes cost seconds each.
    Membership and order are reshuffled every epoch."""
    by = collections.defaultdict(list)
    for i, n in enumerate(lengths):
        by[bucket_of(n)].append(i)
    rng, steps = random.Random(seed), []
    for _ in range(epochs):
        batches = []
        for T in sorted(by):
            idx = by[T][:]
            rng.shuffle(idx)
            bs = max(1, batch_tokens // T)
            batches += [(T, idx[j:j + bs]) for j in range(0, len(idx), bs)]
        rng.shuffle(batches)
        steps += [batches[j:j + batches_per_step] for j in range(0, len(batches), batches_per_step)]
    return steps


def forward_batch(policy, batch, dev, T=0):
    """Final hidden states at the positions that predict completion tokens, and those target tokens. Padded to T."""
    import torch
    T = max(T, max(len(s["ids"]) for s in batch))
    ids = torch.zeros(len(batch), T, dtype=torch.long)
    mask = torch.zeros(len(batch), T, dtype=torch.long)
    sel = torch.zeros(len(batch), T, dtype=torch.bool)
    for b, s in enumerate(batch):
        n = len(s["ids"])
        ids[b, :n] = s["ids"]
        mask[b, :n] = 1
        sel[b, s["p"] - 1:n - 1] = True               # position t predicts token t + 1
    tgt = torch.roll(ids, -1, 1)[sel].to(dev)
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
        h = policy.trunk(ids.to(dev), mask.to(dev))
    return h[sel.to(dev)], tgt


def ce_backward(policy, batch, dev, weight, T=0):
    """Adds weight * (mean negative log-likelihood of the batch's completion tokens) to the gradients. Returns (nll sum, tokens)."""
    hs, tgt = forward_batch(policy, batch, dev, T)
    hd = hs.detach().float().requires_grad_(True)        # head in chunks; its gradient is handed to the trunk once
    total = 0.0
    for i in range(0, len(tgt), g.HEAD_CHUNK):
        j = slice(i, i + g.HEAD_CHUNK)
        nll = -g.head_logp(policy, hd[j], tgt[j]).sum()
        (nll * (weight / len(tgt))).backward()
        total += float(nll.detach())
    hs.backward(hd.grad.to(hs.dtype))
    return total, len(tgt)


def cmd_train(a):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer
    assert os.path.abspath(a.ckpt) != os.path.abspath(a.out), "--out must be a new directory"
    assert not os.path.exists(a.out), f"{a.out} already exists"
    weights = os.path.join(a.ckpt, "model.safetensors")
    assert os.path.exists(weights), f"no {weights}"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True          # float32 head matmul: 4x faster on an H100 (measured)
    log = lambda *x: print(*x, flush=True)
    t0 = time.time()

    # 1. sequences = chat-template prompt + completion + end-of-turn token
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    eot = tok.convert_tokens_to_ids("<|im_end|>")
    if eot is None or eot == tok.unk_token_id:
        eot = tok.eos_token_id
    render = lambda content: tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                     add_generation_prompt=True, enable_thinking=True)   # as run_eval.py
    opened = render("x").rstrip().endswith("<think>")
    log("prompt tail:", repr(render("x")[-40:]), "| template opens the thinking itself:", opened)
    seqs, kinds, dropped = [], collections.Counter(), 0
    with open(a.data) as f:
        for n, line in enumerate(f):
            if a.limit and n >= a.limit:
                break
            r = json.loads(line)
            p = tok(render(r["prompt"]))["input_ids"]
            comp = r["completion"] if opened else "<think>\n" + r["completion"]
            c = tok(comp, add_special_tokens=False)["input_ids"] + [eot]
            if len(p) + len(c) > a.max_seq_tokens:
                dropped += 1
                continue
            seqs.append({"ids": torch.tensor(p + c, dtype=torch.int32), "p": len(p)})
            kinds[r["kind"] + "/" + r["family"]] += 1
    lengths = [len(s["ids"]) for s in seqs]
    targets = sum(len(s["ids"]) - s["p"] for s in seqs)
    log(f"sequences {len(seqs)}, dropped as longer than {a.max_seq_tokens} tokens {dropped}, tokens {sum(lengths)}, "
        f"completion tokens {targets}, end-of-turn id {eot}, tokenized in {time.time() - t0:.0f}s")
    log("rows by kind/family:", json.dumps(dict(sorted(kinds.items()))))
    steps = plan_steps(lengths, a.batch_tokens, a.batches_per_step, a.epochs, a.seed)
    if a.max_steps:
        steps = steps[:a.max_steps]
    warmup = int(a.warmup * len(steps))
    log(f"optimizer steps {len(steps)}, warm-up {warmup}, peak lr {a.lr}, batch tokens {a.batch_tokens} x {a.batches_per_step}")

    # 2. model, parameter mapping (float32 masters)
    st = load_file(weights)
    with safe_open(weights, framework="pt") as f:
        meta = f.metadata()
    model = g.load_model(a.ckpt)
    mapping = g.map_params(model, st, a.train_regex)
    if a.freeze_regex:
        mapping = {k: p for k, p in mapping.items() if not re.search(a.freeze_regex, k)}
    for p in model.parameters():
        p.requires_grad_(False)
    for p in mapping.values():
        p.requires_grad_(True)
    params = list(mapping.values())
    log(f"training {len(mapping)} of {len(st)} tensors ({sum(p.numel() for p in params) / 1e6:.0f}M values), device {dev}")
    model.to(dev).eval()
    policy = g.Policy(model)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    oom = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)

    def run_batch(idx, weight, T=0):
        """One batch; on out-of-memory it is split in two and retried (the weight is shared by token count)."""
        try:
            return ce_backward(policy, [seqs[i] for i in idx], dev, weight, T)
        except oom:
            if len(idx) == 1:
                raise RuntimeError("out of GPU memory with one sequence in the batch")
        gc.collect()
        torch.cuda.empty_cache()
        log(f"OOM_SPLIT batch of {len(idx)} sequences, longest {max(lengths[i] for i in idx)} tokens")
        half = len(idx) // 2
        n_all = sum(lengths[i] - seqs[i]["p"] for i in idx)
        out = [0.0, 0]
        for part in (idx[:half], idx[half:]):
            n_part = sum(lengths[i] - seqs[i]["p"] for i in part)
            nll, n = run_batch(part, weight * n_part / n_all, T)
            out[0] += nll
            out[1] += n
        return out[0], out[1]

    # 3. train
    t1, seen_tok, win_nll, win_n, done = time.time(), 0, 0.0, 0, 0
    for step, group in enumerate(steps):
        lr = lr_at(step, len(steps), a.lr, warmup, a.lr_floor)
        for pg in opt.param_groups:
            pg["lr"] = lr
        for T, idx in group:
            nll, n = run_batch(idx, 1.0 / len(group), T)
            win_nll += nll
            win_n += n
            seen_tok += sum(lengths[i] for i in idx)
        gn = float(torch.nn.utils.clip_grad_norm_(params, a.grad_clip))
        opt.step()
        opt.zero_grad(set_to_none=True)
        done = step + 1
        if done % a.log_every == 0 or done == len(steps):
            dt = time.time() - t1
            log(f"STEP {done}/{len(steps)} lr {lr:.2e} loss {win_nll / max(1, win_n):.4f} gradnorm {gn:.3f} "
                f"tokens/s {seen_tok / dt:.0f} elapsed {dt:.0f}s")
            win_nll, win_n = 0.0, 0
        if a.max_minutes and time.time() - t1 > 60 * a.max_minutes:
            log(f"TIME_LIMIT reached after step {done}; exporting what is trained so far")
            break

    # 4. export
    new = {k: p.detach().float().cpu() for k, p in mapping.items()}
    del model, policy, opt
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    shutil.copytree(a.ckpt, a.out, ignore=lambda d, names: ["model.safetensors"] if os.path.abspath(d) == os.path.abspath(a.ckpt) else [])
    rep = g.export(st, meta, new, os.path.join(a.out, "model.safetensors"), a.seed)
    if a.master_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.master_out)), exist_ok=True)
        save_file(new, a.master_out)
    log("EXPORT", json.dumps(rep), "names/shapes/dtypes/metadata identical: yes")
    log(f"SFT_DONE steps={done} minutes={(time.time() - t0) / 60:.1f}")


def read_ckpt(d):
    from safetensors import safe_open
    from safetensors.torch import load_file
    path = os.path.join(d, "model.safetensors")
    with safe_open(path, framework="pt") as f:
        meta = f.metadata()
    return load_file(path), meta


def write_ckpt(src_dir, out_dir, st, meta, new, seed):
    assert not os.path.exists(out_dir), f"{out_dir} already exists"
    shutil.copytree(src_dir, out_dir, ignore=lambda d, names: ["model.safetensors"] if os.path.abspath(d) == os.path.abspath(src_dir) else [])
    rep = g.export(st, meta, new, os.path.join(out_dir, "model.safetensors"), seed)
    print("EXPORT", json.dumps(rep), flush=True)


def cmd_mix(a):
    sa, meta = read_ckpt(a.a)
    sb, _ = read_ckpt(a.b)
    assert set(sa) == set(sb), "the two checkpoints hold different tensors"
    new = {k: (1 - a.alpha) * sa[k].float() + a.alpha * sb[k].float()
           for k in sa if sa[k].is_floating_point() and not g.same_bits(sa[k], sb[k])}
    assert new, "the two checkpoints are identical"
    write_ckpt(a.a, a.out, sa, meta, new, a.seed)
    print(f"MIX_DONE alpha={a.alpha} tensors={len(new)}")


def cmd_scale(a):
    st, meta = read_ckpt(a.ckpt)
    assert a.key in st, f"no tensor named {a.key}"
    write_ckpt(a.ckpt, a.out, st, meta, {a.key: st[a.key].float() * a.factor}, a.seed)
    print(f"SCALE_DONE {a.key} x {a.factor}: mean {float(st[a.key].float().mean()):.4f} -> {float(st[a.key].float().mean()) * a.factor:.4f}")


def cmd_keys(a):
    st, _ = read_ckpt(a.ckpt)
    for k in st:
        if re.search(a.grep, k):
            t = st[k]
            mean = f"{float(t.float().mean()):.5f}" if t.is_floating_point() else "-"
            print(k, tuple(t.shape), str(t.dtype).replace("torch.", ""), mean)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--ckpt", required=True, help="input checkpoint directory (model.safetensors + config + tokenizer)")
    t.add_argument("--out", required=True, help="output checkpoint directory (must not exist)")
    t.add_argument("--data", required=True, help="rows from sft_data.py")
    t.add_argument("--master-out", default="", help="also save the trained tensors in float32")
    t.add_argument("--train-regex", default="language_model", help="train tensors whose safetensors key matches this")
    t.add_argument("--freeze-regex", default="", help="but not the ones matching this")
    t.add_argument("--lr", type=float, default=5e-6)
    t.add_argument("--lr-floor", type=float, default=0.1, help="the cosine ends at this fraction of the peak")
    t.add_argument("--warmup", type=float, default=0.05, help="fraction of the steps")
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--batch-tokens", type=int, default=16000, help="sequences x longest sequence per batch")
    t.add_argument("--batches-per-step", type=int, default=1)
    t.add_argument("--max-seq-tokens", type=int, default=2048, help="longer rows are dropped, never cut")
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--max-steps", type=int, default=0, help="0 = no limit (use a few for a speed test)")
    t.add_argument("--max-minutes", type=float, default=0, help="stop and export after this much training time")
    t.add_argument("--limit", type=int, default=0, help="read only the first N rows")
    t.add_argument("--log-every", type=int, default=20)
    t.add_argument("--seed", type=int, default=0)
    m = sub.add_parser("mix")
    m.add_argument("--a", required=True)
    m.add_argument("--b", required=True)
    m.add_argument("--alpha", type=float, required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("scale")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--key", required=True)
    s.add_argument("--factor", type=float, required=True)
    s.add_argument("--seed", type=int, default=0)
    k = sub.add_parser("keys")
    k.add_argument("--ckpt", required=True)
    k.add_argument("--grep", default="norm")
    a = ap.parse_args()
    {"train": cmd_train, "mix": cmd_mix, "scale": cmd_scale, "keys": cmd_keys}[a.cmd](a)


if __name__ == "__main__":
    main()
