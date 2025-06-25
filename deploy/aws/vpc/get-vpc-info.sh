#!/bin/bash

# Script to get VPC information for ECS service configuration
set -e

REGION="us-west-1"
VPC_ID="vpc-0ad5901eb9d7873e9"  # Your VPC ID

echo "Getting VPC information for ECS service configuration..."
echo "VPC ID: $VPC_ID"
echo "Region: $REGION"
echo ""

# Get subnets in the VPC
echo "=== SUBNETS ==="
echo "Getting subnets in VPC $VPC_ID..."
aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query 'Subnets[*].[SubnetId,AvailabilityZone,MapPublicIpOnLaunch,Tags[?Key==`Name`].Value|[0]]' \
    --output table \
    --region $REGION

echo ""
echo "=== SECURITY GROUPS ==="
echo "Getting security groups in VPC $VPC_ID..."
aws ec2 describe-security-groups \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query 'SecurityGroups[*].[GroupId,GroupName,Description]' \
    --output table \
    --region $REGION

echo ""
echo "=== RECOMMENDED CONFIGURATION ==="
echo "For ECS services, you typically want:"
echo "- Private subnets (MapPublicIpOnLaunch=false) for better security"
echo "- Or public subnets (MapPublicIpOnLaunch=true) if you need direct internet access"
echo "- Security group that allows HTTP/HTTPS traffic and inter-service communication"
echo ""
echo "Copy the subnet IDs and security group ID to update your GitHub workflow!"
