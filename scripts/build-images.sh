#!/bin/bash
set -e

# This script builds Docker images for the AutoVisionAI application locally.
#
# Usage: ./scripts/build-images.sh [image_tag] [repository_url]
#
# Arguments:
#   image_tag: The tag to apply to the images (e.g., the git commit SHA). Defaults to 'latest'.
#   repository_url: The repository URL to tag images with. Defaults to 'autovisionai'.

# --- Configuration ---
IMAGE_TAG=${1:-latest} # Default to 'latest' if no tag is provided
REPOSITORY_URL=${2:-autovisionai} # Default repository name

echo "--- Building Docker Images ---"
echo "Repository URL: $REPOSITORY_URL"
echo "Image Tag: $IMAGE_TAG"
echo "------------------------------------------"

# --- Main Application Image ---
echo "Building main application image..."
docker build -t "$REPOSITORY_URL:app" -t "$REPOSITORY_URL:app-$IMAGE_TAG" -f docker/Dockerfile .

# --- MLflow Image ---
echo "Building MLflow image..."
docker build -t "$REPOSITORY_URL:mlflow" -t "$REPOSITORY_URL:mlflow-$IMAGE_TAG" -f docker/Dockerfile.mlflow .

# --- TensorBoard Image ---
echo "Building TensorBoard image..."
docker build -t "$REPOSITORY_URL:tensorboard" -t "$REPOSITORY_URL:tensorboard-$IMAGE_TAG" -f docker/Dockerfile.tensorboard .

echo "------------------------------------------"
echo "All images built successfully."
echo "Images created:"
echo "  - $REPOSITORY_URL:latest"
echo "  - $REPOSITORY_URL:$IMAGE_TAG"
echo "  - $REPOSITORY_URL:mlflow"
echo "  - $REPOSITORY_URL:mlflow-$IMAGE_TAG"
echo "  - $REPOSITORY_URL:tensorboard"
echo "  - $REPOSITORY_URL:tensorboard-$IMAGE_TAG"
echo "------------------------------------------"
