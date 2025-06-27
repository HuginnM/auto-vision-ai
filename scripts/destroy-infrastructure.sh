#!/bin/bash
set -e

# This script destroys the AutoVisionAI infrastructure and cleans up all resources.
#
# Usage: ./scripts/destroy-infrastructure.sh [aws_region]
#
# Arguments:
#   aws_region: AWS region (defaults to us-west-1)

# --- Configuration ---
AWS_REGION=${1:-us-west-1}
TF_WORKING_DIR=deploy/terraform

echo "--- Destroying AutoVisionAI Infrastructure ---"
echo "AWS Region: $AWS_REGION"
echo "Working Directory: $TF_WORKING_DIR"
echo "WARNING: This will permanently delete all resources!"
echo "------------------------------------------"

# Confirmation prompt
read -p "Type 'DESTROY' to confirm: " confirmation
if [ "$confirmation" != "DESTROY" ]; then
    echo "Confirmation failed. You must type 'DESTROY' to proceed."
    exit 1
fi

echo "Starting destruction process..."

# --- Pre-destruction cleanup ---
echo "--- Pre-destruction cleanup ---"

# Scale down ECS services to 0
echo "Scaling down ECS services..."
aws ecs update-service --cluster autovisionai --service autovision-api --desired-count 0 --region "$AWS_REGION" || echo "API service not found or already scaled down"
aws ecs update-service --cluster autovisionai --service autovision-ui --desired-count 0 --region "$AWS_REGION" || echo "UI service not found or already scaled down"
aws ecs update-service --cluster autovisionai --service autovision-mlflow --desired-count 0 --region "$AWS_REGION" || echo "MLflow service not found or already scaled down"
aws ecs update-service --cluster autovisionai --service autovision-tensorboard --desired-count 0 --region "$AWS_REGION" || echo "TensorBoard service not found or already scaled down"

# Wait for services to scale down
echo "Waiting 15s for services to scale down..."
sleep 15

# Empty S3 bucket (required for destruction)
echo "Emptying S3 bucket..."
aws s3 rm s3://autovision-mlflow-artifacts --recursive --region "$AWS_REGION" || echo "S3 bucket not found or already empty"

# Force delete the secret if it exists
echo "Force deleting WANDB secret..."
aws secretsmanager delete-secret \
    --secret-id autovisionai/wandb-api-key \
    --force-delete-without-recovery \
    --region "$AWS_REGION" || echo "Secret not found or already deleted"

# Clean up ECR images BEFORE destroying the repository
# List any remaining tagged images and delete them
echo "Cleaning up remaining ECR images..."
IMAGES=$(aws ecr list-images \
    --repository-name autovisionai \
    --region "$AWS_REGION" \
    --query 'imageIds[?imageTag!=null]' \
    --output json 2>/dev/null || echo "[]")

if [ ! -z "$IMAGES" ]; then
    aws ecr batch-delete-image \
        --repository-name autovisionai \
        --image-ids "$IMAGES" \
        --region "$AWS_REGION" || echo "No remaining images to delete"
fi

echo "Pre-destruction cleanup completed"

# --- Terraform destruction ---
echo "--- Terraform destruction ---"

# Initialize Terraform
echo "Initializing Terraform..."
terraform -chdir="$TF_WORKING_DIR" init

# Plan destruction
echo "Planning destruction..."
terraform -chdir="$TF_WORKING_DIR" plan -destroy -out=destroy-plan

# Apply destruction
echo "Applying destruction..."
terraform -chdir="$TF_WORKING_DIR" apply -auto-approve destroy-plan

# --- Post-destruction cleanup ---
echo "--- Post-destruction cleanup ---"
echo "Post-destruction cleanup completed"

# --- Verification ---
echo "--- Verification ---"

# Check if ECS cluster still exists
echo "Checking ECS cluster..."
CLUSTER=$(aws ecs describe-clusters --clusters autovisionai --region "$AWS_REGION" --query 'clusters[0].status' --output text 2>/dev/null || echo "NOTFOUND")
if [ "$CLUSTER" = "ACTIVE" ]; then
    echo "Warning: ECS cluster still exists"
else
    echo "ECS cluster destroyed"
fi

# Check if ECR repository still exists
echo "Checking ECR repository..."
REPO=$(aws ecr describe-repositories --repository-names autovisionai --region "$AWS_REGION" --query 'repositories[0].repositoryName' --output text 2>/dev/null || echo "NOTFOUND")
if [ "$REPO" = "autovisionai" ]; then
    echo "Warning: ECR repository still exists"
else
    echo "ECR repository destroyed"
fi

# Check if S3 bucket still exists
echo "Checking S3 bucket..."
BUCKET=$(aws s3api head-bucket --bucket autovision-mlflow-artifacts --region "$AWS_REGION" 2>/dev/null && echo "EXISTS" || echo "NOTFOUND")
if [ "$BUCKET" = "EXISTS" ]; then
    echo "Warning: S3 bucket still exists"
else
    echo "S3 bucket destroyed"
fi

# Check if secret still exists
echo "Checking WANDB secret..."
SECRET=$(aws secretsmanager describe-secret --secret-id autovisionai/wandb-api-key --region "$AWS_REGION" --query 'Name' --output text 2>/dev/null || echo "NOTFOUND")
if [ "$SECRET" = "autovisionai/wandb-api-key" ]; then
    echo "Warning: WANDB secret still exists"
else
    echo "WANDB secret destroyed"
fi

echo ""
echo "------------------------------------------"
echo "Destruction process completed!"
echo "------------------------------------------"
echo ""
echo "Please verify in the AWS console that all resources have been removed."
echo "Some resources might take a few minutes to fully terminate."
echo ""
echo "If any resources still exist, you may need to manually delete them"
echo "or wait for them to be automatically cleaned up by AWS."
