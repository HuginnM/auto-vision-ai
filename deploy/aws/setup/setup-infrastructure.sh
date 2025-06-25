#!/bin/bash

# This script sets up all required AWS infrastructure for AutoVision AI
set -e

REGION="us-west-1"
CLUSTER_NAME="autovisionai"
S3_BUCKET="autovisionai-mlflow-artifacts"
REPO_NAME="autovisionai"


echo "Setting up AWS infrastructure for AutoVision AI..."

# 1. Create S3 bucket for MLflow artifacts
echo "Creating S3 bucket for MLflow artifacts..."
if aws s3api head-bucket --bucket $S3_BUCKET --region $REGION 2>/dev/null; then
    echo "S3 bucket $S3_BUCKET already exists"
else
    aws s3api create-bucket --bucket $S3_BUCKET --region $REGION --create-bucket-configuration LocationConstraint="$REGION"
    echo "Created S3 bucket: $S3_BUCKET"
fi

# 2. Create CloudWatch log groups
echo "Creating CloudWatch log groups..."
LOG_GROUPS=(
    "/ecs/autovision-api"
    "/ecs/autovision-ui"
    "/ecs/autovision-mlflow"
    "/ecs/autovision-tensorboard"
)

for LOG_GROUP in "${LOG_GROUPS[@]}"; do
    if aws logs describe-log-groups --log-group-name-prefix "$LOG_GROUP" --region $REGION --query 'logGroups[0].logGroupName' --output text | grep -q "$LOG_GROUP" 2>/dev/null; then
        echo "Log group $LOG_GROUP already exists"
    else
        aws logs create-log-group --log-group-name "$LOG_GROUP" --region $REGION
        echo "Created log group: $LOG_GROUP"
    fi
done

# 3. Create ECS cluster
echo "Creating ECS cluster..."
if aws ecs describe-clusters --clusters $CLUSTER_NAME --region $REGION --query 'clusters[0].clusterName' --output text | grep -q "$CLUSTER_NAME" 2>/dev/null; then
    echo "ECS cluster $CLUSTER_NAME already exists"
else
    aws ecs create-cluster --cluster-name $CLUSTER_NAME --region $REGION
    echo "Created ECS cluster: $CLUSTER_NAME"
fi

# 4. Create ECR repository
echo "Creating ECR repository..."

if aws ecr describe-repositories --repository-names $REPO_NAME --region $REGION 2>/dev/null; then
    echo "ECR repository $REPO_NAME already exists"
else
    aws ecr create-repository --repository-name $REPO_NAME --region $REGION
    echo "Created ECR repository: $REPO_NAME"
fi

echo ""
echo "==========================================="
echo "Infrastructure Setup Complete!"
echo "==========================================="
echo "S3 Bucket: $S3_BUCKET"
echo "ECS Cluster: $CLUSTER_NAME"
echo "ECR Repository: $REPO_NAME"
echo "Log Groups: Created 4 log groups"
echo ""
echo "Next steps:"
echo "1. Create ECS services using the task definitions"
echo "2. Set up Application Load Balancer"
echo "3. Configure security groups"
echo "4. Run the GitHub Actions workflow to deploy"
echo "==========================================="
