#!/bin/bash
set -e

IMAGE_NAME="anandkabirus/llm-gateway"
TAG="latest"

echo "Building Docker image for linux/amd64 (VPS compatible)..."
# Using --platform linux/amd64 to ensure it runs on standard VPS even if built on Apple Silicon
docker build --platform linux/amd64 -t $IMAGE_NAME:$TAG .

echo "Pushing image to Docker Hub..."
docker push $IMAGE_NAME:$TAG

echo "Done! Image pushed to $IMAGE_NAME:$TAG"
