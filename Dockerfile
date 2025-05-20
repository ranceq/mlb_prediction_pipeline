# Use official Python runtime base image
FROM python:3.10-slim

# Set environment variables
ENV PORT=8080 \
    PYTHONUNBUFFERED=1 \
    # Tell Flask to run in production
    FLASK_ENV=production \
    # Entry point for Cloud Run
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=8080

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
      curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py ./

# Expose the port
EXPOSE 8080

# Health check endpoint
# Start the Flask app
CMD ["flask", "run"]
