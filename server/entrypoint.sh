#!/bin/bash
set -e

MODEL_ID=${MODEL_ID:-meta-llama/Meta-Llama-3-8B-Instruct}
MODEL_DIR=/models/Meta-Llama-3-8B-Instruct

if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: Hugging Face token not provided! Set HF_TOKEN as environment variable."
  exit 1
fi

mkdir -p /models

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "Downloading model $MODEL_ID to $MODEL_DIR ..."
  python3 - <<PYTHON
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$MODEL_ID",
    token="$HF_TOKEN",
    local_dir="$MODEL_DIR",
)
PYTHON
else
  echo "Model already present in $MODEL_DIR"
fi

echo "Model ready, starting app..."
exec python3 app.py
