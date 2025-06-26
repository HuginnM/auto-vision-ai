#!/bin/bash
set -e

# This script pushes Docker images to a specified ECR repository.
#
# Usage: ./scripts/push-images.sh <ecr_repository_url> [image_tag]
#
# Arguments:
#   ecr_repository_url: The full URL of the ECR repository to push the images to.
#   image_tag: The tag to push (e.g., the git commit SHA). Defaults to 'latest'.

# --- Configuration ---
ECR_REPOSITORY_URL=$1
IMAGE_TAG=${2:-latest} # Default to 'latest' if no tag is provided

# Check if required arguments are provided
if [ -z "$ECR_REPOSITORY_URL" ]; then
  echo "Error: ECR repository URL is required."
  echo "Usage: $0 <ecr_repository_url> [image_tag]"
  exit 1
fi

echo "--- Pushing Docker Images to ECR ---"
echo "Repository URL: $ECR_REPOSITORY_URL"
echo "Image Tag: $IMAGE_TAG"
echo "------------------------------------------"

# --- Main Application Image ---
echo "Pushing main application image..."
docker push "$ECR_REPOSITORY_URL:latest"
docker push "$ECR_REPOSITORY_URL:$IMAGE_TAG"

# --- MLflow Image ---
echo "Pushing MLflow image..."
docker push "$ECR_REPOSITORY_URL:mlflow"
docker push "$ECR_REPOSITORY_URL:mlflow-$IMAGE_TAG"

# --- TensorBoard Image ---
echo "Pushing TensorBoard image..."
docker push "$ECR_REPOSITORY_URL:tensorboard"
docker push "$ECR_REPOSITORY_URL:tensorboard-$IMAGE_TAG"

echo "------------------------------------------"
echo "✅ All images pushed successfully."
echo "------------------------------------------"
