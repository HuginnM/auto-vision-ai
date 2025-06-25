#!/bin/bash

set -euo pipefail

AWS_REGION="us-west-1"
CLUSTER_NAME="autovisionai"

# 1. Register Task Definitions
echo "Registering task definitions..."

aws ecs register-task-definition --cli-input-json file://deploy/aws/api_task_definition.json
aws ecs register-task-definition --cli-input-json file://deploy/aws/ui_task_definition.json
aws ecs register-task-definition --cli-input-json file://deploy/aws/mlflow_task_definition.json
aws ecs register-task-definition --cli-input-json file://deploy/aws/tensorboard_task_definition.json

# 2. Create or Update ECS Service
create_or_update_service() {
  local service_name=$1
  local task_definition=$2

  echo "Checking if service $service_name exists..."
  if aws ecs describe-services --cluster "$CLUSTER_NAME" --services "$service_name" --region "$AWS_REGION" \
      --query 'services[0].status' --output text 2>/dev/null | grep -q "ACTIVE"; then
    echo "Service $service_name exists. Updating..."
    aws ecs update-service \
      --cluster "$CLUSTER_NAME" \
      --service "$service_name" \
      --task-definition "$task_definition" \
      --region "$AWS_REGION"
  else
    echo "Service $service_name does not exist. Creating..."
    aws ecs create-service \
      --cluster "$CLUSTER_NAME" \
      --service-name "$service_name" \
      --task-definition "$task_definition" \
      --desired-count 1 \
      --launch-type FARGATE \
      --network-configuration "awsvpcConfiguration={subnets=[subnet-0e522b325a73bdf85,subnet-085cfe6f7532f5f14],securityGroups=[sg-0028aa6bb51011b34],assignPublicIp=ENABLED}" \
      --region "$AWS_REGION"
  fi
}

# 3. Update all services
create_or_update_service "autovision-api" "autovision-api"
create_or_update_service "autovision-ui" "autovision-ui"
create_or_update_service "autovision-mlflow" "autovision-mlflow"
create_or_update_service "autovision-tensorboard" "autovision-tensorboard"
