# AutoVisionAI Terraform Deployment

This directory contains Terraform configuration files for deploying the AutoVisionAI application to AWS using ECS Fargate.

## Architecture Overview

The Terraform configuration provisions the following AWS resources:

### Core Infrastructure
- **VPC** with public and private subnets across 2 availability zones
- **Internet Gateway** and **NAT Gateways** for internet connectivity
- **Security Groups** for ALB, ECS tasks, and EFS
- **Application Load Balancer** with target groups for all services

### Container Services
- **ECS Cluster** with Fargate launch type
- **ECR Repository** for container images
- **ECS Services** for:
  - API service (port 8000)
  - UI service (port 8501)
  - MLflow service (port 8080)
  - TensorBoard service (port 6006)

### Storage & Logging
- **EFS File System** for shared experiment data
- **S3 Bucket** for MLflow artifacts with versioning and lifecycle policies
- **CloudWatch Log Groups** for all services
- **Secrets Manager** for WANDB API key

### Auto Scaling & Monitoring
- **Auto Scaling** targets and policies for API and UI services
- **CloudWatch Alarms** for CPU-based scaling

## Prerequisites

1. **AWS CLI** configured with appropriate credentials
2. **Terraform** >= 1.6.0
3. **Docker** for building container images
4. **AWS Account** with permissions for ECS, ECR, VPC, S3, EFS, Secrets Manager

## Required AWS Permissions

Your AWS user/role needs the following permissions:
- EC2 (VPC, Security Groups, Load Balancer)
- ECS (Cluster, Services, Task Definitions)
- ECR (Repository management)
- IAM (Roles and Policies)
- S3 (Bucket management)
- EFS (File System management)
- CloudWatch (Logs and Metrics)
- Secrets Manager
- Application Auto Scaling

## Configuration

### Variables

Key variables in `variables.tf`:

| Variable | Description | Default |
|----------|-------------|---------|
| `aws_region` | AWS region | `us-west-1` |
| `project_name` | Project name prefix | `autovisionai` |
| `environment` | Environment (prod/staging/dev) | `prod` |
| `cluster_name` | ECS cluster name | `autovisionai` |
| `wandb_entity` | WANDB entity name | `arthur-sobol-private` |
| `mlflow_artifacts_bucket` | S3 bucket for MLflow | `autovision-mlflow-artifacts` |

### Customization

Update `terraform.tfvars` to customize your deployment:

```hcl
aws_region   = "us-west-1"
project_name = "autovisionai"
environment  = "prod"

# Optionally override other variables
api_cpu    = 1024
api_memory = 2048
ui_cpu     = 512
ui_memory  = 1024
```

## Deployment

### Manual Deployment

1. **Initialize Terraform:**
   ```bash
   cd deploy/terraform
   terraform init
   ```

2. **Plan the deployment:**
   ```bash
   terraform plan
   ```

3. **Apply the configuration:**
   ```bash
   terraform apply
   ```

4. **Build and push Docker images:**
   ```bash
   # Get ECR repository URL
   ECR_URI=$(terraform output -raw ecr_repository_url)

   # Login to ECR
   aws ecr get-login-password --region us-west-1 | docker login --username AWS --password-stdin $ECR_URI

   # Build and push images
   docker build -f ../../docker/Dockerfile -t $ECR_URI:latest .
   docker push $ECR_URI:latest

   docker build -f ../../docker/Dockerfile.mlflow -t $ECR_URI:mlflow .
   docker push $ECR_URI:mlflow

   docker build -f ../../docker/Dockerfile.tensorboard -t $ECR_URI:tensorboard .
   docker push $ECR_URI:tensorboard
   ```

5. **Update ECS services:**
   ```bash
   # Force new deployment to use updated images
   aws ecs update-service --cluster autovisionai --service autovision-api --force-new-deployment
   aws ecs update-service --cluster autovisionai --service autovision-ui --force-new-deployment
   aws ecs update-service --cluster autovisionai --service autovision-mlflow --force-new-deployment
   aws ecs update-service --cluster autovisionai --service autovision-tensorboard --force-new-deployment
   ```

### CI/CD Deployment

The repository includes a GitHub Actions workflow (`.github/workflows/deploy-terraform.yml`) that automatically:

1. Provisions infrastructure with Terraform
2. Builds and pushes Docker images to ECR
3. Updates ECS services with new images
4. Waits for services to stabilize

#### Required GitHub Secrets:

- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `WANDB_API_KEY`: (Optional) WANDB API key

## Post-Deployment

### Setting Up Secrets

If you didn't provide the WANDB API key during deployment, set it manually:

```bash
aws secretsmanager update-secret \
  --secret-id "autovision/wandb-api-key" \
  --secret-string "your-wandb-api-key" \
  --region us-west-1
```

### Accessing Services

After deployment, get the service URLs:

```bash
terraform output api_url
terraform output ui_url
terraform output mlflow_url
terraform output tensorboard_url
```

### Monitoring

- **CloudWatch Logs**: Check `/ecs/autovision-*` log groups
- **ECS Console**: Monitor service health and task status
- **ALB**: Check target group health in EC2 console

## Scaling

### Manual Scaling

Update desired count for any service:

```bash
aws ecs update-service \
  --cluster autovisionai \
  --service autovision-api \
  --desired-count 3
```

### Auto Scaling

Auto scaling is configured for API and UI services based on CPU utilization:
- Scale up when CPU > 80% for 2 consecutive minutes
- Scale down when CPU < 10% for 2 consecutive minutes

## Troubleshooting

### Common Issues

1. **Service fails to start:**
   - Check CloudWatch logs for container errors
   - Verify ECR image exists and is accessible
   - Check security group rules

2. **EFS mount fails:**
   - Verify EFS security group allows NFS traffic (port 2049)
   - Check subnet routing to EFS mount targets

3. **Load balancer health checks fail:**
   - Verify application is listening on correct port
   - Check health check path configuration
   - Review security group rules

### Useful Commands

```bash
# Check service status
aws ecs describe-services --cluster autovisionai --services autovision-api

# View logs
aws logs tail /ecs/autovision-api --follow

# Check task definitions
aws ecs list-task-definitions --family-prefix autovision

# View ALB target health
aws elbv2 describe-target-health --target-group-arn <target-group-arn>
```

## Cleanup

To destroy all resources:

```bash
terraform destroy
```

**Warning:** This will delete all data in S3 and EFS. Make sure to backup any important data first.

## Security Considerations

- All ECS tasks run in private subnets
- Internet access through NAT gateways
- Security groups follow principle of least privilege
- S3 bucket has public access blocked
- EFS encrypted at rest
- Secrets stored in AWS Secrets Manager

## Cost Optimization

- EFS uses provisioned throughput for predictable costs
- S3 lifecycle policies transition old artifacts to cheaper storage
- Auto scaling minimizes running tasks during low usage
- Spot instances not used for production stability

## Support

For issues or questions:
1. Check CloudWatch logs first
2. Review AWS console for resource status
3. Verify configuration in `terraform.tfvars`
4. Check GitHub Actions workflow logs for CI/CD issues
