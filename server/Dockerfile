FROM ubuntu:22.04

ARG USE_CUDA=true

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    sudo \
    python3 \
    python3-distutils \
    python3-pip \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN pip install -U openai-whisper

WORKDIR /app

COPY requirements.txt .

RUN if [ "$USE_CUDA" = "true" ]; then \
      pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 ; \
    else \
      pip install torch torchvision torchaudio ; \
    fi

RUN pip install --break-system-packages -r requirements.txt

COPY . .

CMD ["python3", "app.py"]
