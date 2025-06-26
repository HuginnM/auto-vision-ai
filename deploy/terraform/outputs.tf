# outputs.tf

# Load Balancer DNS Name
output "alb_hostname" {
  description = "Load balancer DNS name"
  value       = aws_lb.main.dns_name
}

# Service URLs
output "api_url" {
  description = "API service URL"
  value       = "http://${aws_lb.main.dns_name}:${var.api_port}"
}

output "ui_url" {
  description = "UI service URL"
  value       = "http://${aws_lb.main.dns_name}:${var.ui_port}"
}

output "mlflow_url" {
  description = "MLflow service URL"
  value       = "http://${aws_lb.main.dns_name}:${var.mlflow_port}"
}

output "tensorboard_url" {
  description = "TensorBoard service URL"
  value       = "http://${aws_lb.main.dns_name}:${var.tensorboard_port}"
}

# ECR Repository URL
output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.autovisionai.repository_url
}

# EFS File System
output "efs_file_system_id" {
  description = "EFS file system ID"
  value       = aws_efs_file_system.experiments.id
}

# S3 Bucket
output "s3_bucket_name" {
  description = "S3 bucket name for MLflow artifacts"
  value       = aws_s3_bucket.mlflow_artifacts.bucket
}

# ECS Cluster
output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

# VPC Information
output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = aws_subnet.private[*].id
}
