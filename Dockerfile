FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for OpenCV and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, models, and static assets
COPY . .

ENV PORT=7860
EXPOSE 7860

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120"]
