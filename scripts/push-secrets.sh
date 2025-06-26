#!/bin/bash
set -e

# This script pushes secrets from .env file to AWS Secrets Manager.
#
# Usage: ./scripts/push-secrets.sh [aws_region] [secret_name]
#
# Arguments:
#   aws_region: AWS region (defaults to us-west-1)
#   secret_name: Secret name in AWS Secrets Manager (defaults to autovisionai/wandb-api-key)

# --- Configuration ---
AWS_REGION=${1:-us-west-1}
WANDB_SECRET_NAME=${2:-autovisionai/wandb-api-key}

echo "--- Pushing Secrets from .env to AWS Secrets Manager ---"
echo "AWS Region: $AWS_REGION"
echo "WANDB Secret Name: $WANDB_SECRET_NAME"
echo "------------------------------------------"

# Check if .env file exists
if [ ! -f .env ]; then
  echo "Error: .env file not found."
  echo "Please create a .env file with your secrets:"
  echo "  WANDB_API_KEY=your_api_key_here"
  exit 1
fi

# Source .env file and check for required secrets
source .env

if [ -z "$WANDB_API_KEY" ]; then
  echo "Error: WANDB_API_KEY not found in .env file."
  echo "Please add it to your .env file:"
  echo "  WANDB_API_KEY=your_api_key_here"
  exit 1
fi

# Push WANDB API Key to AWS Secrets Manager
if aws secretsmanager describe-secret --secret-id "$WANDB_SECRET_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
  echo "Updating existing WANDB API Key secret..."
  aws secretsmanager update-secret \
    --secret-id "$WANDB_SECRET_NAME" \
    --secret-string "$WANDB_API_KEY" \
    --region "$AWS_REGION"
else
  echo "Creating new WANDB API Key secret..."
  aws secretsmanager create-secret \
    --name "$WANDB_SECRET_NAME" \
    --description "WANDB API Key for AutoVisionAI" \
    --secret-string "$WANDB_API_KEY" \
    --region "$AWS_REGION"
fi

echo "------------------------------------------"
echo "Secrets pushed successfully."
echo "------------------------------------------"
