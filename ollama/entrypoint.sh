#!/bin/bash

echo "[entrypoint] Starting Ollama server in background..."

ollama serve &

echo "[entrypoint] Waiting for Ollama to be ready..."
until curl -s http://localhost:11434 > /dev/null; do
  sleep 2
done

echo "[entrypoint] Ollama server ready, running generating_prompts.py..."

python3 /app/generating_prompts.py

echo "[entrypoint] Prompt generation done, keeping Ollama running."

wait