#!/bin/bash
# Score any checkpoint directory on the replica. usage: evalck.sh <ckpt dir> <tag> [samples=2]
cd "$HOME/dl" || exit 1
D=$1; TAG=$2; S=${3:-2}
stop_server() { pkill -f "[v]llm ser""ve"; for i in $(seq 1 30); do pgrep -f "[v]llm ser""ve" >/dev/null || break; sleep 2; done; sleep 3; }
stop_server
VLLM_USE_FLASHINFER_SAMPLER=0 nohup venv-vllm/bin/vllm ser""ve "$PWD/$D" --served-model-name m --host 127.0.0.1 --port 8000 \
  --language-model-only --dtype bfloat16 --max-model-len 34816 --max-num-seqs 384 \
  --gpu-memory-utilization 0.85 --no-enable-log-requests > "logs/vllm-$TAG.log" 2>&1 &
for i in $(seq 1 200); do curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null && break; sleep 3; done
curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null || { echo "EVAL_FAILED server $TAG"; exit 1; }
venv-vllm/bin/python run_eval.py --model-dir "$D" --concurrency 384 --mode sample --samples "$S" --out "out/$TAG-sample.jsonl"
stop_server
venv-score/bin/python score.py replica.json "out/$TAG-sample.jsonl" "out/$TAG-sample.summary.json" > "out/$TAG-sample.table.txt" 2> "logs/score-$TAG.err"
venv-score/bin/python - "out/$TAG-sample.summary.json" "$TAG" <<'PY'
import json, sys
f = json.load(open(sys.argv[1]))["families"]
rest = ["arc_challenge", "arc_easy", "boolq", "commonsense_qa", "gsm8k", "hellaswag", "ifeval", "mmlu", "openbookqa", "piqa", "winogrande"]
m, r = f["math500"]["pass_strict"], sum(f[k]["pass_strict"] for k in rest) / len(rest)
print("BLEND %s %.1f  (0.4 x math500 %.1f + 0.6 x rest %.1f; math500 cut off %.1f%%)" % (sys.argv[2], 0.4 * m + 0.6 * r, m, r, f["math500"]["cut"]))
PY
echo "EVAL_DONE $TAG"
