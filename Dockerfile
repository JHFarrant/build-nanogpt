FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Copy repo files
COPY . /app

# Pre-install dependencies to speed up container startup
RUN pip install --no-cache-dir -q --upgrade pip && \
    pip install --no-cache-dir -q -r requirements.txt

# Entry point
CMD ["/bin/bash", "run_on_vastai.sh"]
