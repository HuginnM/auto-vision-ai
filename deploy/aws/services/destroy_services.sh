#!/bin/bash

# Set AWS region and ECS cluster name
AWS_REGION="us-west-1"
CLUSTER_NAME="autovisionai"

# List of ECS services to stop
SERVICES=("autovision-api" "autovision-ui" "autovision-mlflow" "autovision-tensorboard")

echo "Setting desiredCount=0 for ECS services..."
for SERVICE in "${SERVICES[@]}"; do
    echo "Stopping $SERVICE..."
    aws ecs update-service \
        --cluster "$CLUSTER_NAME" \
        --service "$SERVICE" \
        --desired-count 0 \
        --region "$AWS_REGION"
done

echo "ECS services stopped."

# # Delete services completely
# echo "Deleting ECS services..."
# for SERVICE in "${SERVICES[@]}"; do
#     echo "Deleting $SERVICE..."
#     aws ecs delete-service \
#         --cluster "$CLUSTER_NAME" \
#         --service "$SERVICE" \
#         --force \
#         --region "$AWS_REGION"
# done
# echo "ECS services deleted."

# Stop any remaining tasks
echo "Stopping any remaining ECS tasks..."
TASKS=$(aws ecs list-tasks --cluster "$CLUSTER_NAME" --region "$AWS_REGION" --output text --query 'taskArns[]')
for TASK in $TASKS; do
    echo "Stopping task $TASK..."
    aws ecs stop-task --cluster "$CLUSTER_NAME" --task "$TASK" --region "$AWS_REGION"
done

# Delete NAT Gateway
echo "Checking for NAT Gateways..."
NAT_IDS=$(aws ec2 describe-nat-gateways \
    --filter Name=state,Values=available \
    --region "$AWS_REGION" \
    --query "NatGateways[].NatGatewayId" --output text)

for NAT_ID in $NAT_IDS; do
    echo "Deleting NAT Gateway: $NAT_ID"
    aws ec2 delete-nat-gateway --nat-gateway-id "$NAT_ID" --region "$AWS_REGION"
done

# # Delete EFS if no longer needed
# echo "Checking for EFS file systems..."
# EFS_IDS=$(aws efs describe-file-systems --region "$AWS_REGION" --query "FileSystems[].FileSystemId" --output text)

# for EFS_ID in $EFS_IDS; do
#     echo "Deleting EFS: $EFS_ID"
#     MOUNT_TARGETS=$(aws efs describe-mount-targets --file-system-id "$EFS_ID" --region "$AWS_REGION" --query "MountTargets[].MountTargetId" --output text)
#     for TARGET in $MOUNT_TARGETS; do
#         aws efs delete-mount-target --mount-target-id "$TARGET" --region "$AWS_REGION"
#     done
#     sleep 5  # wait for targets to be deleted
#     aws efs delete-file-system --file-system-id "$EFS_ID" --region "$AWS_REGION"
# done

# echo "Shutdown complete. No further charges should apply."
