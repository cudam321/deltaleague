#!/usr/bin/env python3
"""Download one Delta League model version (public endpoint, no login) and verify its hashes."""
import hashlib, json, os, subprocess, sys, urllib.request

vid, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
req = urllib.request.Request(f"https://deltaleague.ai/api/v1/model-versions/{vid}", headers={"User-Agent": "curl/8"})
meta = json.load(urllib.request.urlopen(req, timeout=60))
for f in meta["files"]:
    path = os.path.join(out, f["name"])
    if not (os.path.exists(path) and os.path.getsize(path) == f["size_bytes"]):
        subprocess.run(["curl", "-sS", "-L", "--retry", "3", "-o", path, f["url"]], check=True)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    assert h.hexdigest() == f["sha256"], f"hash mismatch: {f['name']}"
print("OK", vid, meta.get("name", "")[:80])
