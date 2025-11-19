# Use a lightweight Python base image
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (for numpy, opencv, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better for Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project
COPY . .

# Make sure Python can find your 'routes' package
ENV PYTHONPATH=/app

# Expose the port your app uses
EXPOSE 5001

# Default command: run with Gunicorn
# "app:app" = <filename_without_py>:<Flask_app_variable>
CMD ["gunicorn", "-b", "0.0.0.0:5001", "app:app"]
