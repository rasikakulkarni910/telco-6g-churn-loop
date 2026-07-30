# Deploy-from-source image for Cloud Run (built remotely by Cloud Build — no local Docker required).
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Minimal system deps for scientific wheels / SSL.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY agents ./agents
COPY models ./models
COPY data ./data
COPY main.py Procfile ./

# Drop unused local artifacts if copied
RUN find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

EXPOSE 8080
CMD ["python", "main.py"]
