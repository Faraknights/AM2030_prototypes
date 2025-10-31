FROM ollama/ollama

# Install Python and dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --break-system-packages pexpect

# Create app directory and copy files
WORKDIR /app
COPY . .

# Add custom entrypoint script
RUN chmod +x entrypoint.sh

# Use custom entrypoint
ENTRYPOINT ["bash", "entrypoint.sh"]
