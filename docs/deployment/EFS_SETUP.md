# EFS Setup for AutoVisionAI

## Overview

The EFS (Elastic File System) is configured to provide shared storage for experiment data across all AutoVisionAI services. This ensures that your cloud deployment has the same file structure as your local development environment.

## Local vs Cloud Structure

### Local Structure
```
AutoVisionAI/
├── experiments/          # Your local experiments folder
├── src/
├── tests/
├── docker/
└── ...
```

### Cloud Structure
```
/app/experiments/         # EFS mount point in containers
├── [same structure as local experiments/]
└── [shared across all services]
```

## EFS Configuration Details

### 1. EFS File System
- **Name**: `autovisionai-experiments-efs`
- **Performance Mode**: General Purpose
- **Throughput Mode**: Provisioned (100 MiB/s)
- **Encryption**: Enabled at rest
- **Lifecycle Policy**: Transitions to IA after 30 days

### 2. EFS Access Point
- **Name**: `autovisionai-experiments-access-point`
- **Root Directory**: `/app/experiments`
- **POSIX User**: UID/GID 1000
- **Permissions**: 755

### 3. Mount Configuration

#### API Service
- **Mount Point**: `/app/experiments`
- **Permissions**: Read/Write
- **Purpose**: Store and retrieve experiment data, model files, etc.

#### UI Service
- **Mount Point**: `/app/experiments`
- **Permissions**: Read-Only
- **Purpose**: Display experiment results, visualizations, etc.

#### TensorBoard Service
- **Mount Point**: `/app/experiments`
- **Permissions**: Read/Write
- **Purpose**: Read TensorBoard logs and write new logs

## Security Features

### 1. Transit Encryption
- **Status**: Enabled
- **Port**: 2049 (NFS)
- **Protocol**: TLS 1.2

### 2. IAM Authorization
- **Status**: Enabled
- **Access Control**: EFS Access Point with IAM policies
- **User Isolation**: Each service has specific permissions

### 3. Network Security
- **Security Group**: `autovisionai-efs-security-group`
- **Allowed Traffic**: NFS (port 2049) from ECS tasks only
- **Subnet**: Private subnets only

## Service Access Matrix

| Service | Mount Path | Permissions | Purpose |
|---------|------------|-------------|---------|
| API | `/app/experiments` | Read/Write | Store experiment data, models |
| UI | `/app/experiments` | Read-Only | Display results, visualizations |
| TensorBoard | `/app/experiments` | Read/Write | TensorBoard logs |
| MLflow | N/A | N/A | Uses S3 for artifacts |

## Code Compatibility

### No Code Changes Required

Your existing code will work without any modifications because:

1. **Same Path Structure**: EFS mounts to `/app/experiments` which matches your local `experiments/` folder structure
2. **Same File Operations**: All file I/O operations work identically
3. **Same Permissions**: Read/write operations work as expected

### Example Code That Works Unchanged

```python
# This code works the same locally and in the cloud
import os
from pathlib import Path

# Local development
experiments_dir = Path("experiments")

# Cloud deployment (same code!)
experiments_dir = Path("/app/experiments")

# Both work identically
experiment_file = experiments_dir / "my_experiment" / "results.json"
experiment_file.write_text("experiment data")
```

## Data Persistence

### What's Stored in EFS
- Experiment configurations
- Model checkpoints
- Training logs
- Evaluation results
- Generated visualizations
- Custom experiment data

### What's NOT Stored in EFS
- MLflow artifacts (stored in S3)
- Application logs (stored in CloudWatch)
- Container images (stored in ECR)

## Backup and Recovery

### Automatic Backups
- EFS has built-in redundancy across AZs
- Data is automatically replicated within the region

### Manual Backups
```bash
# Create a backup of experiments data
aws efs create-backup \
  --file-system-id fs-xxxxxxxxx \
  --region us-west-1

# List backups
aws efs describe-backups \
  --region us-west-1
```

## Monitoring and Troubleshooting

### Check EFS Status
```bash
# Check file system status
aws efs describe-file-systems \
  --file-system-id $(terraform -chdir=deploy/terraform output -raw efs_file_system_id) \
  --region us-west-1

# Check mount targets
aws efs describe-mount-targets \
  --file-system-id $(terraform -chdir=deploy/terraform output -raw efs_file_system_id) \
  --region us-west-1
```

### Common Issues

1. **Mount Failures**
   - Check security group rules
   - Verify subnet routing
   - Check ECS task IAM permissions

2. **Permission Issues**
   - Verify EFS access point configuration
   - Check POSIX user settings
   - Review IAM policies

3. **Performance Issues**
   - Monitor EFS metrics in CloudWatch
   - Consider increasing provisioned throughput
   - Check for high I/O operations

## Cost Optimization

### Current Configuration
- **Throughput**: 100 MiB/s provisioned
- **Storage**: Pay per GB used
- **Lifecycle**: IA transition after 30 days

### Optimization Tips
1. **Monitor Usage**: Use CloudWatch to track actual usage
2. **Adjust Throughput**: Scale down if usage is low
3. **Lifecycle Policies**: Use IA for infrequently accessed data
4. **Cleanup**: Regularly remove old experiment data

## Migration from Local to Cloud

### Step 1: Deploy Infrastructure
```bash
cd deploy/terraform
terraform apply
```

### Step 2: Upload Local Data (Optional)
```bash
# If you want to migrate existing experiments
aws s3 sync experiments/ s3://autovision-mlflow-artifacts/experiments/
```

### Step 3: Verify Mount
```bash
# Check if services can access EFS
aws ecs describe-tasks \
  --cluster autovisionai \
  --tasks $(aws ecs list-tasks --cluster autovisionai --query 'taskArns[0]' --output text)
```

## Best Practices

1. **Organize Data**: Use consistent folder structures
2. **Clean Up**: Remove old experiments regularly
3. **Monitor Usage**: Track storage and throughput usage
4. **Backup Important Data**: Create backups for critical experiments
5. **Use Versioning**: Implement versioning for important files

## Support

For EFS-related issues:
1. Check CloudWatch logs for mount errors
2. Verify security group and IAM configurations
3. Test connectivity from ECS tasks
4. Review EFS metrics in AWS console
