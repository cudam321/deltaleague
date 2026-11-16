#!/usr/bin/env python3
"""Package a full fine-tuned checkpoint as a Delta League `full`-format submission, and check it against its parent.

`delta patch build` writes a sparse diff: 8 bytes of index plus the value for every changed element. After full
fine-tuning most of the 752M trained values differ, so the sparse file would be several times the size of the
checkpoint itself. ARTIFACTS.md also allows format "full": weights.safetensors is the whole checkpoint.

Needs numpy only (no torch). Reads both files through mmap, so memory use stays small.

Checks before anything is written:
  1. same tensor names, shapes and dtypes as the parent
  2. no NaN or Inf in any floating tensor of the modified file
  3. which tensors differ from the parent, byte for byte (printed by top-level prefix)

usage: pack_full.py --parent PARENT.safetensors --modified EDITED.safetensors --out DIR
                    --parent-id UUID --parent-hash CONTENT_HASH [--model-id qwen35-08b-v1] [--frozen-outside PREFIX]
"""
import argparse, collections, hashlib, json, mmap, os, struct, sys

import numpy as np

EXP_MASK = {"BF16": (np.uint16, 0x7F80), "F16": (np.uint16, 0x7C00), "F32": (np.uint32, 0x7F800000),
            "F64": (np.uint64, 0x7FF0000000000000)}
CHUNK = 1 << 26


def open_st(path):
    """-> (header without __metadata__, offset of the byte buffer, mmap)"""
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)


def span(header, base, name):
    a, b = header[name]["data_offsets"]
    return base + a, base + b


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True)
    ap.add_argument("--modified", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parent-id", required=True)
    ap.add_argument("--parent-hash", required=True)
    ap.add_argument("--model-id", default="qwen35-08b-v1")
    ap.add_argument("--frozen-outside", default="model.language_model.",
                    help="fail if a tensor whose name does not start with this differs from the parent")
    a = ap.parse_args()

    hp, bp, mp = open_st(a.parent)
    hm, bm, mm = open_st(a.modified)
    assert set(hp) == set(hm), "tensor names differ from the parent"
    for k in hp:
        assert hp[k]["shape"] == hm[k]["shape"] and hp[k]["dtype"] == hm[k]["dtype"], f"shape or dtype differs: {k}"
    print(f"tensors {len(hp)}: names, shapes and dtypes match the parent")

    changed, same, bad = collections.Counter(), collections.Counter(), []
    changed_values = total_values = 0
    for k in sorted(hp):
        s0, e0 = span(hp, bp, k)
        s1, e1 = span(hm, bm, k)
        assert e0 - s0 == e1 - s1, f"byte length differs: {k}"
        differs = False
        kind = EXP_MASK.get(hm[k]["dtype"])
        for off in range(0, e1 - s1, CHUNK):
            n = min(CHUNK, e1 - s1 - off)
            x = mm[s1 + off:s1 + off + n]
            if not differs and x != mp[s0 + off:s0 + off + n]:
                differs = True
            if kind:
                v = np.frombuffer(x, dtype=kind[0])
                if ((v & kind[1]) == kind[1]).any():
                    bad.append(k)
                    break
        top = ".".join(k.split(".")[:2])
        (changed if differs else same)[top] += 1
        count = int(np.prod(hm[k]["shape"])) if hm[k]["shape"] else 1
        total_values += count
        if differs:
            changed_values += count
            assert k.startswith(a.frozen_outside), f"a tensor outside {a.frozen_outside} differs from the parent: {k}"
    assert not bad, f"non-finite values in: {bad[:5]}"
    print("no NaN or Inf in any floating tensor")
    print("tensors that differ from the parent, by prefix:", dict(changed))
    print("tensors identical to the parent, by prefix:", dict(same))
    print(f"values in tensors that differ: {changed_values:,} of {total_values:,}")

    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "weights.safetensors")
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(a.modified, dst)           # same bytes, no second copy on disk
    except OSError:
        import shutil
        shutil.copyfile(a.modified, dst)
    digest = file_hash(dst)
    manifest = {"schema_version": "delta.submission.v1", "reconstruction_version": "delta-reconstruct-v1",
                "model_config_id": a.model_id, "parent_version_id": a.parent_id, "parent_content_hash": a.parent_hash,
                "format": "full", "operation": "replace",
                "artifacts": [{"name": "weights.safetensors", "sha256": digest, "size_bytes": os.path.getsize(dst)}]}
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("weights.safetensors sha256", digest, "bytes", os.path.getsize(dst))
    print("PACK_DONE", a.out)


if __name__ == "__main__":
    sys.exit(main())
