#!/bin/bash
# Sample a checkpoint on any items file under the test sampler. usage: sample.sh <ckpt dir> <items.json> <out.jsonl> [samples=1] [max tokens=8192]
cd "$HOME/dl" || exit 1
D=$1; ITEMS=$2; OUT=$3; S=${4:-1}; MAXTOK=${5:-8192}
stop_server() { pkill -f "[v]llm ser""ve"; for i in $(seq 1 30); do pgrep -f "[v]llm ser""ve" >/dev/null || break; sleep 2; done; sleep 3; }
stop_server
VLLM_USE_FLASHINFER_SAMPLER=0 nohup venv-vllm/bin/vllm ser""ve "$PWD/$D" --served-model-name m --host 127.0.0.1 --port 8000 \
  --language-model-only --dtype bfloat16 --max-model-len 34816 --max-num-seqs 384 \
  --gpu-memory-utilization 0.85 --no-enable-log-requests > "logs/vllm-sample.log" 2>&1 &
for i in $(seq 1 200); do curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null && break; sleep 3; done
curl -sf -m 2 http://127.0.0.1:8000/health >/dev/null || { echo "SAMPLE_FAILED server"; exit 1; }
venv-vllm/bin/python run_eval.py --model-dir "$D" --replica "$ITEMS" --concurrency 384 --mode sample --samples "$S" --max-tokens "$MAXTOK" --out "$OUT"
stop_server
echo "SAMPLE_DONE $OUT"
