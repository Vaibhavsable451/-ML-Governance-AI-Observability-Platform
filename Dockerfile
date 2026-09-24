# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# Native build tools for SHAP and other scientific packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip

# Use pip cache mount so wheels persist across build attempts
RUN pip install --no-cache-dir --prefer-binary --default-timeout=1000 --retries 20 -r requirements.txt

COPY . .

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]