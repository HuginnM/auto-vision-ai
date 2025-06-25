#!/bin/bash

# This script sets up EFS file system for AutoVision AI experiments
set -e

REGION="us-west-1"
VPC_ID="vpc-0ad5901eb9d7873e9"
SUBNET_IDS=("subnet-01d736e623358f5d0" "subnet-0e522b325a73bdf85")
SECURITY_GROUP_ID="sg-07dd88fd32fa065a6"

echo "Setting up EFS file system for AutoVision AI..."

# Create EFS file system
echo "Creating EFS file system..."
EFS_RESPONSE=$(aws efs create-file-system \
    --creation-token "autovision-experiments-$(date +%s)" \
    --tags Key=Name,Value=autovision-experiments \
    --region $REGION \
    --output json)

FILE_SYSTEM_ID=$(echo $EFS_RESPONSE | jq -r '.FileSystemId')
echo "Created EFS file system: $FILE_SYSTEM_ID"

# Wait for file system to become available
echo "Waiting for file system to become available..."
aws efs wait file-system-available --file-system-id $FILE_SYSTEM_ID --region $REGION

# Create mount targets in each subnet
echo "Creating mount targets..."
for SUBNET_ID in "${SUBNET_IDS[@]}"; do
    echo "Creating mount target in subnet: $SUBNET_ID"
    aws efs create-mount-target \
        --file-system-id $FILE_SYSTEM_ID \
        --subnet-id $SUBNET_ID \
        --security-groups $SECURITY_GROUP_ID \
        --region $REGION
done

# Create access point for experiments directory
echo "Creating access point for experiments directory..."
ACCESS_POINT_RESPONSE=$(aws efs create-access-point \
    --file-system-id $FILE_SYSTEM_ID \
    --posix-user "Uid=1000,Gid=1000" \
    --root-directory "Path=/experiments,CreationInfo={OwnerUid=1000,OwnerGid=1000,Permissions=755}" \
    --tags Key=Name,Value=autovision-experiments-access-point \
    --region $REGION \
    --output json)

ACCESS_POINT_ID=$(echo $ACCESS_POINT_RESPONSE | jq -r '.AccessPointId')
echo "Created access point: $ACCESS_POINT_ID"
echo "=========================================="
echo "EFS Setup Complete!"
echo "=========================================="
echo "File System ID: $FILE_SYSTEM_ID"
echo "Access Point ID: $ACCESS_POINT_ID"
echo "=========================================="
