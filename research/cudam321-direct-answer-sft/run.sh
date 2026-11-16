#!/bin/bash
# The commands behind this experiment, in the order they ran on one H100 box (Ubuntu, 2026-11-16).
# Collected from shell history. I have NOT re-run this file top to bottom as a single script, so treat it as an
# exact record of the steps and not as a tested one-click pipeline. All scripts sit together in ~/dl.
# The sampling and scoring helpers start and stop a vLLM server themselves.
# apt packages first: gcc libc6-dev python3-dev ninja-build g++ make cmake
set -e
cd "$HOME/dl"; mkdir -p ckpt rounds data out logs
TRAIN="env FLA_DISABLE_BACKEND_DISPATCH=1 PYTHONPATH=$PWD/extra venv-vllm/bin/python sft.py train"
PY=venv-score/bin/python

# 0. two environments: vLLM 0.32.0 (sampling, training) and lm-eval 0.4.13 + math-verify 0.9.0 (scoring)
bash setup.sh
uv pip install --python venv-vllm/bin/python --target extra --no-deps fla-core   # fast gated-delta kernels, training only

# 1. parent, replica, parent's score (2 samples per task)
$PY fetch_ckpt.py eff1e1b7-18ef-4c77-83ef-25fdcc027e75 ckpt/r0
$PY build_replica.py replica.json                 # 1,992 tasks from public eval splits; measurement only
bash evalck.sh ckpt/r0 r0 2                       # the parent's table in results/ is from 2026-11-15, same files

# 2. training pool: train splits only; a question whose text is in any eval split, or in MATH-500, is dropped
$PY build_pool.py --out pool.json
$PY build_pool.py --out pool2.json --cap-mc 45000 # same pool with more HellaSwag/WinoGrande rows, for run D

# 3. "direct" rows: empty thinking, then the bare answer
$PY sft_data.py --out data/direct.jsonl --pool pool.json \
  --direct hellaswag,winogrande,piqa,commonsense_qa,boolq --direct-cap 20000 --exact-boolq 3000

# 4. "trace" rows: the parent's own correct samples on train questions, thinking kept
#    (this first pass called run_eval.py by hand with the arguments sample.sh uses; sample.sh came later)
$PY pick.py --pool pool.json --out rounds/t1.json --take arc_challenge=1200,arc_easy=2300,openbookqa=2000,mmlu=2000,gsm8k=2500
bash sample.sh ckpt/r0 rounds/t1.json rounds/t1.gens.jsonl 1 8192
$PY grpo_step.py rewards --items rounds/t1.json --gens rounds/t1.gens.jsonl --out rounds/t1.rewards.jsonl
$PY sft_data.py --out data/trace.jsonl --trace-items rounds/t1.json --gens rounds/t1.gens.jsonl --rewards rounds/t1.rewards.jsonl

# 5. run A: direct + trace, two passes  ->  instruction-following collapses (70.0 -> 32.0)
cat data/direct.jsonl data/trace.jsonl > data/runA.jsonl
$TRAIN --ckpt ckpt/r0 --out ckpt/A --data data/runA.jsonl --epochs 2 --lr 5e-6 --log-every 50
bash evalck.sh ckpt/A A 2

# 6. rehearsal rows for free-form prompts: the parent's own answers that pass the real IFEval checkers
$PY ifl_items.py --out rounds/ifl.json --take 5000   # argilla/ifeval-like-data minus every prompt in google/IFEval
bash sample.sh ckpt/r0 rounds/ifl.json rounds/ifl.gens.jsonl 1 8192
$PY grpo_step.py rewards --items rounds/ifl.json --gens rounds/ifl.gens.jsonl --out rounds/ifl.rewards.jsonl
$PY sft_data.py --out data/ifl.jsonl --trace-items rounds/ifl.json --gens rounds/ifl.gens.jsonl \
  --rewards rounds/ifl.rewards.jsonl --traces-per-item 1 --trace-max-tokens 2400

# (run B1, discarded: run A's data + ifl rows twice + dataset worked solutions as the thinking text, built with
#  sft_data.py --math-train / --gsm8k-train / --exact-gsm8k / --omi2. Row counts are in results/train-runB1.log.
#  One pass, 5,154 steps. MATH-500 17.3, GSM8K 36.3. Not part of the lineage of the submitted checkpoint.)

# 7. the parent's own correct MATH-train solutions, sampled with a 3,000-token cap
$PY pick.py --pool pool.json --out rounds/m1.json --take math_train=7500   # all 7,401 MATH train problems in the pool
bash sample.sh ckpt/r0 rounds/m1.json rounds/m1.gens.jsonl 1 3000
$PY grpo_step.py rewards --items rounds/m1.json --gens rounds/m1.gens.jsonl --out rounds/m1.rewards.jsonl
$PY sft_data.py --out data/mtrace.jsonl --trace-items rounds/m1.json --gens rounds/m1.gens.jsonl \
  --rewards rounds/m1.rewards.jsonl --traces-per-item 1 --trace-max-tokens 2900

# 8. run C: from the parent again, two passes
cat data/direct.jsonl data/trace.jsonl data/ifl.jsonl data/mtrace.jsonl > data/runC.jsonl
$TRAIN --ckpt ckpt/r0 --out ckpt/C --data data/runC.jsonl --epochs 2 --lr 5e-6 --max-seq-tokens 3200 --log-every 100
bash evalck.sh ckpt/C C 2

# 9. final-norm scale sweep on run C (logits x f, which is sampling temperature 1/f)
for f in 1.3 1.6 2.0; do
  t=Cs${f/./}
  venv-vllm/bin/python sft.py scale --ckpt ckpt/C --out ckpt/$t --key model.language_model.norm.weight --factor $f
  bash evalck.sh ckpt/$t $t 2
done

# 10. long math solutions: the problems cut off at 3,000 tokens in step 7, resampled at the full 8,192
$PY - <<'PY'
import json
items = json.load(open("rounds/m1.json"))["items"]
fr = {}
for l in open("rounds/m1.gens.jsonl"):
    r = json.loads(l); fr[r["id"]] = r["fr"]
cut = [it for it in items if fr.get(it["id"], "length") != "stop"]
json.dump({"items": cut}, open("rounds/m2.json", "w"))
print("m2 items:", len(cut))          # 4,593
PY
bash sample.sh ckpt/r0 rounds/m2.json rounds/m2.gens.jsonl 1 8192
$PY grpo_step.py rewards --items rounds/m2.json --gens rounds/m2.gens.jsonl --out rounds/m2.rewards.jsonl
$PY sft_data.py --out data/mlong.jsonl --trace-items rounds/m2.json --gens rounds/m2.gens.jsonl \
  --rewards rounds/m2.rewards.jsonl --traces-per-item 1 --trace-max-tokens 7000

# 11. HellaSwag/WinoGrande train rows that run C had not seen
$PY sft_data.py --out data/direct2-all.jsonl --pool pool2.json --direct hellaswag,winogrande --direct-cap 45000
$PY - <<'PY'
import json
old = {json.loads(l)["id"] for l in open("data/direct.jsonl")}
oldp = {json.loads(l)["prompt"] for l in open("data/direct.jsonl")}
n = 0
with open("data/direct-new.jsonl", "w") as f:
    for l in open("data/direct2-all.jsonl"):
        r = json.loads(l)
        if r["id"] in old or r["prompt"] in oldp:
            continue
        f.write(l); n += 1
print("new direct rows", n)           # 40,235
PY

# 12. run D: from run C, one pass; then the x1.6 copy that is the submission
cat data/direct.jsonl data/direct-new.jsonl data/trace.jsonl data/ifl.jsonl data/mtrace.jsonl data/mlong.jsonl data/mlong.jsonl > data/runD.jsonl
$TRAIN --ckpt ckpt/C --out ckpt/D --data data/runD.jsonl --epochs 1 --lr 4e-6 --max-seq-tokens 7424 --log-every 100
venv-vllm/bin/python sft.py scale --ckpt ckpt/D --out ckpt/Ds16 --key model.language_model.norm.weight --factor 1.6
bash evalck.sh ckpt/Ds16 Ds16 6
sha256sum ckpt/Ds16/model.safetensors   # 3ce20e17b593df8e040c27e5c2de81b3ab8819968e016e22028b04169b97821d

# 13. package as a full-format artifact (no torch needed) and check it against the parent
python3 pack_full.py --parent ckpt/r0/model.safetensors --modified ckpt/Ds16/model.safetensors --out submission \
  --parent-id eff1e1b7-18ef-4c77-83ef-25fdcc027e75 --parent-hash 6ba2eed0971dcc9fbb9e20d8a44128580fcdc1d1d8c3b774470c8d6676006fc2
delta validate submission
