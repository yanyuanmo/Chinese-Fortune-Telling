FROM python:3.10-slim

WORKDIR /app

# Create a non-root user (Hugging Face Spaces requirement: UID 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install CPU-only PyTorch first to avoid pulling large CUDA libraries
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
COPY api/requirements.txt ./api_requirements.txt
COPY app/requirements.txt ./app_requirements.txt
RUN pip install --no-cache-dir -r api_requirements.txt && \
    pip install --no-cache-dir -r app_requirements.txt

# Copy application files
COPY . .

# Ensure correct permissions for the non-root user
RUN chown -R user:user /app && chmod +x /app/start.sh

USER user

# Hugging Face Spaces expects traffic on port 7860
EXPOSE 7860

ENV API_URL="http://127.0.0.1:8000" \
    CHROMA_DIR="/app/chroma_db" \
    PYTHONUNBUFFERED=1

CMD ["/app/start.sh"]
