#!/bin/bash
set -e

# This script assumes you have Docker and Docker Compose installed on your VPS.

echo "Pulling latest image..."
docker pull anandkabirus/llm-gateway:latest

echo "Restarting service..."
# If docker-compose is not in path, try 'docker compose' (newer syntax)
if command -v docker-compose &> /dev/null; then
    docker-compose down
    docker-compose up -d
else
    docker compose down
    docker compose up -d
fi

echo "Deployment complete! App should be running on port 8089."
