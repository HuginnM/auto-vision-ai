#!/bin/bash
set -e

# This script pushes Docker images to a specified ECR repository.
#
# Usage: ./scripts/push-images.sh [image_tag] [ecr_repository_url]
#
# Arguments:
#   image_tag: The tag to push (e.g., the git commit SHA). Defaults to 'latest'.
#   ecr_repository_url: The full URL of the ECR repository. Defaults to constructed ECR URL.

# --- Configuration ---
IMAGE_TAG=${1:-latest} # Default to 'latest' if no tag is provided
AWS_REGION=us-west-1
ACCOUNT_ID=869935094020
ACCOUNT_URL="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
ECR_REPOSITORY_URL=${2:-"$ACCOUNT_URL/autovisionai"} # Default to full ECR URL

echo "--- Pushing Docker Images to ECR ---"
echo "Repository URL: $ECR_REPOSITORY_URL"
echo "Image Tag: $IMAGE_TAG"
echo "------------------------------------------"

# --- AWS ECR Authentication ---
echo "Authenticating with AWS ECR..."
# Get ECR login token and authenticate Docker
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ACCOUNT_URL"

if [ $? -ne 0 ]; then
  echo "Error: Failed to authenticate with ECR"
  echo "Please ensure:"
  echo "  1. AWS credentials are configured (aws configure)"
  echo "  2. You have permissions to access ECR"
  echo "  3. The ECR repository exists"
  exit 1
fi

echo "Successfully authenticated with ECR"

# --- Main Application Image ---
echo "Pushing main application image..."
docker push "$ECR_REPOSITORY_URL:app"
docker push "$ECR_REPOSITORY_URL:app-$IMAGE_TAG"

# --- MLflow Image ---
echo "Pushing MLflow image..."
docker push "$ECR_REPOSITORY_URL:mlflow"
docker push "$ECR_REPOSITORY_URL:mlflow-$IMAGE_TAG"

# --- TensorBoard Image ---
echo "Pushing TensorBoard image..."
docker push "$ECR_REPOSITORY_URL:tensorboard"
docker push "$ECR_REPOSITORY_URL:tensorboard-$IMAGE_TAG"

echo "------------------------------------------"
echo "All images pushed successfully."
echo "------------------------------------------"
