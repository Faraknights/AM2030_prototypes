FROM ollama/ollama

# Install Python and dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip curl dos2unix && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN PIP_BREAK_SYSTEM_PACKAGES=1 pip install --no-cache-dir pexpect

# Create app directory and copy files
WORKDIR /app
COPY . .

# Convert entrypoint.sh to Unix line endings and make it executable
RUN dos2unix entrypoint.sh && chmod +x entrypoint.sh

# Use custom entrypoint
ENTRYPOINT ["bash", "entrypoint.sh"]
