FROM python:3.10 as builder

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama (needed for LLM inference)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create a standalone binary using PyInstaller
RUN pip install pyinstaller && \
    pyinstaller --onefile --name=local_rag local_rag_markdown.py

# Final minimal scratch container
FROM scratch

# Copy the built binary
COPY --from=builder /app/dist/local_rag /local_rag

# Set entrypoint
ENTRYPOINT ["/local_rag"]

