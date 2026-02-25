# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (none needed for basic python app, but good practice to clean up)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the internal port (application runs on 8000 internally)
EXPOSE 8000

# Command to run the application
# We use 0.0.0.0 to allow external connections into the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
