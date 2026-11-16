#!/bin/bash
# One-time environment setup on the rented box. Two environments, as the competition repo advises.
cd "$HOME/dl" || exit 1
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv venv venv-vllm --python 3.12
uv pip install -p venv-vllm/bin/python "vllm==0.32.0" || uv pip install -p venv-vllm/bin/python vllm
echo VLLM_ENV_DONE
uv venv venv-score --python 3.12
uv pip install -p venv-score/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu \
  "torch==2.10.0" "lm-eval[ifeval]==0.4.13" "math-verify==0.9.0" "antlr4-python3-runtime==4.11" datasets \
  || uv pip install -p venv-score/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu \
  torch "lm-eval[ifeval]==0.4.13" "math-verify==0.9.0" "antlr4-python3-runtime==4.11" datasets
echo SETUP_DONE
