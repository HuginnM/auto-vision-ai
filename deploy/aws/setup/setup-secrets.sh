#!/bin/bash

# This script populates AWS Secrets Manager from GitHub Secrets
# Run this in GitHub Actions with the secrets as environment variables
# In case if you need to update the secrets, you can run this script again

set -e

echo "Setting up AWS Secrets Manager from GitHub Secrets..."

# Create or update WANDB API Key secret
if aws secretsmanager describe-secret --secret-id "autovision/wandb-api-key" --region us-west-1 2>/dev/null; then
    echo "Updating existing WANDB API Key secret..."
    aws secretsmanager update-secret \
        --secret-id "autovision/wandb-api-key" \
        --secret-string "$WANDB_API_KEY" \
        --region us-west-1
else
    echo "Creating new WANDB API Key secret..."
    aws secretsmanager create-secret \
        --name "autovision/wandb-api-key" \
        --description "WANDB API Key for AutoVision AI" \
        --secret-string "$WANDB_API_KEY" \
        --region us-west-1
fi

echo "AWS Secrets Manager setup complete!"
