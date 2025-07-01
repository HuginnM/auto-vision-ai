#!/bin/bash

# Setup Terraform S3 Backend
# This script creates an S3 bucket for storing Terraform state with proper security settings

set -e

# Configuration
PROJECT_NAME="autovisionai"
REGION="us-west-1"
BUCKET_NAME="${PROJECT_NAME}-terraform-state"

echo "Setting up Terraform S3 Backend..."
echo "Project: $PROJECT_NAME"
echo "Region: $REGION"
echo "Bucket: $BUCKET_NAME"

# Create S3 bucket
echo "Creating S3 bucket..."
aws s3api create-bucket \
    --bucket "$BUCKET_NAME" \
    --region "$REGION" \
    --create-bucket-configuration LocationConstraint="$REGION"

# Enable versioning for state file protection
echo "Enabling versioning..."
aws s3api put-bucket-versioning \
    --bucket "$BUCKET_NAME" \
    --versioning-configuration Status=Enabled

# Enable server-side encryption
echo "Enabling encryption..."
aws s3api put-bucket-encryption \
    --bucket "$BUCKET_NAME" \
    --server-side-encryption-configuration '{
        "Rules": [
            {
                "ApplyServerSideEncryptionByDefault": {
                    "SSEAlgorithm": "AES256"
                },
                "BucketKeyEnabled": true
            }
        ]
    }'

# Block public access
echo "Blocking public access..."
aws s3api put-public-access-block \
    --bucket "$BUCKET_NAME" \
    --public-access-block-configuration \
        BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

# Add lifecycle policy to clean up old versions
echo "Setting up lifecycle policy..."
aws s3api put-bucket-lifecycle-configuration \
    --bucket "$BUCKET_NAME" \
    --lifecycle-configuration '{
        "Rules": [
            {
                "ID": "DeleteOldVersions",
                "Status": "Enabled",
                "NoncurrentVersionExpiration": {
                    "NoncurrentDays": 90
                }
            },
            {
                "ID": "DeleteIncompleteMultipartUploads",
                "Status": "Enabled",
                "AbortIncompleteMultipartUpload": {
                    "DaysAfterInitiation": 7
                }
            }
        ]
    }'

echo "S3 Backend setup complete!"
echo "Bucket name: $BUCKET_NAME"
