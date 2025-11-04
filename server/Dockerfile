FROM ubuntu:22.04

ARG USE_CUDA=true

# Usar mirror HTTP que não precisa de certificados
RUN printf "deb http://archive.ubuntu.com/ubuntu jammy main restricted universe multiverse\n\
deb http://archive.ubuntu.com/ubuntu jammy-updates main restricted universe multiverse\n\
deb http://archive.ubuntu.com/ubuntu jammy-backports main restricted universe multiverse\n\
deb http://security.ubuntu.com/ubuntu jammy-security main restricted universe multiverse\n" \
> /etc/apt/sources.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        sudo \
        python3 \
        python3-distutils \
        python3-pip \
        python3-venv \
        ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip --no-cache-dir
RUN pip install --no-cache-dir -U openai-whisper

WORKDIR /app
COPY requirements.txt .

RUN if [ "$USE_CUDA" = "true" ]; then \
        pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 ; \
    else \
        pip install --no-cache-dir torch torchvision torchaudio ; \
    fi

RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY . .

CMD ["python3", "app.py"]

