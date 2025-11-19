# ---------- Base Image ----------
FROM python:3.11-slim

# Avoid Python writing .pyc and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# Install system dependencies required by OpenCV etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---------- Python dependencies ----------
COPY requirements.txt .

# Increase pip timeout to handle large wheels / slow network
ENV PIP_DEFAULT_TIMEOUT=120

RUN pip install --no-cache-dir -r requirements.txt

# ---------- Application code ----------
# .dockerignore will exclude .env, .git, venv, etc.
COPY . .

# Ensure start script is executable
RUN chmod +x start_app.sh

# Expose application port (Flask/Gunicorn uses 5001)
EXPOSE 5001

# Default command: run Gunicorn via your start script
CMD ["./start_app.sh"]
