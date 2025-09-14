FROM ubuntu:22.04

ARG USE_CUDA=true

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    sudo \
    python3 \
    python3-distutils \
    python3-pip \
    git \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

WORKDIR /app

# Install PyTorch
RUN if [ "$USE_CUDA" = "true" ]; then \
      pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 ; \
    else \
      pip install torch torchvision torchaudio ; \
    fi

# Install project dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Hugging Face deps
RUN pip install huggingface_hub transformers accelerate safetensors peft

# Copy your code
COPY . .

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Hugging Face cache folder
ENV HF_HOME=/models/hf_cache
ENV MODEL_ID=meta-llama/Meta-Llama-3-8B-Instruct
ENV OUTPUT_FILE=/app/output.txt

ENTRYPOINT ["/entrypoint.sh"]
