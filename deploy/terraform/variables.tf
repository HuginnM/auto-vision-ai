# variables.tf

variable "aws_region" {
    description = "The AWS region where resources will be created"
    type        = string
    default     = "us-west-1"
}

variable "project_name" {
    description = "Name of the project"
    type        = string
    default     = "autovisionai"
}

variable "environment" {
    description = "Environment (prod, staging, dev)"
    type        = string
    default     = "prod"
}

variable "cluster_name" {
    description = "ECS cluster name"
    type        = string
    default     = "autovisionai"
}

variable "az_count" {
    description = "Number of AZs to cover in a given region"
    type        = number
    default     = 2
}

# ECR Repository
variable "ecr_repository_name" {
    description = "ECR repository name"
    type        = string
    default     = "autovisionai"
}

# ECS Service Configuration
variable "api_service_name" {
    description = "API service name"
    type        = string
    default     = "autovision-api"
}

variable "ui_service_name" {
    description = "UI service name"
    type        = string
    default     = "autovision-ui"
}

variable "mlflow_service_name" {
    description = "MLflow service name"
    type        = string
    default     = "autovision-mlflow"
}

variable "tensorboard_service_name" {
    description = "TensorBoard service name"
    type        = string
    default     = "autovision-tensorboard"
}

# Task Definition CPU and Memory
variable "api_cpu" {
    description = "API Fargate instance CPU units to provision (1 vCPU = 1024 CPU units)"
    type        = number
    default     = 1024
}

variable "api_memory" {
    description = "API Fargate instance memory to provision (in MiB)"
    type        = number
    default     = 2048
}

variable "ui_cpu" {
    description = "UI Fargate instance CPU units to provision"
    type        = number
    default     = 1024
}

variable "ui_memory" {
    description = "UI Fargate instance memory to provision (in MiB)"
    type        = number
    default     = 2048
}

variable "mlflow_cpu" {
    description = "MLflow Fargate instance CPU units to provision"
    type        = number
    default     = 512
}

variable "mlflow_memory" {
    description = "MLflow Fargate instance memory to provision (in MiB)"
    type        = number
    default     = 1024
}

variable "tensorboard_cpu" {
    description = "TensorBoard Fargate instance CPU units to provision"
    type        = number
    default     = 512
}

variable "tensorboard_memory" {
    description = "TensorBoard Fargate instance memory to provision (in MiB)"
    type        = number
    default     = 1024
}

# Ports
variable "api_port" {
    description = "Port for API service"
    type        = number
    default     = 8000
}

variable "ui_port" {
    description = "Port for UI service"
    type        = number
    default     = 8501
}

variable "mlflow_port" {
    description = "Port for MLflow service"
    type        = number
    default     = 8080
}

variable "tensorboard_port" {
    description = "Port for TensorBoard service"
    type        = number
    default     = 6006
}

# Health Check Paths
variable "api_health_check_path" {
    description = "Health check path for API"
    type        = string
    default     = "/health"
}

variable "ui_health_check_path" {
    description = "Health check path for UI"
    type        = string
    default     = "/_stcore/health"
}

variable "mlflow_health_check_path" {
    description = "Health check path for MLflow"
    type        = string
    default     = "/health"
}

# WANDB Configuration
variable "wandb_entity" {
    description = "WANDB entity name"
    type        = string
    default     = "arthur-sobol-private"
}

# S3 Bucket for MLflow artifacts
variable "mlflow_artifacts_bucket" {
    description = "S3 bucket for MLflow artifacts"
    type        = string
    default     = "autovision-mlflow-artifacts"
}

# Environment Mode
variable "env_mode" {
    description = "Environment mode"
    type        = string
    default     = "prod"
}

# Secrets Manager
variable "wandb_secret_name" {
    description = "The name of the Secrets Manager secret for the WANDB API key"
    type        = string
    default     = "autovisionai/wandb-api-key"
}

variable "create_ecs_services" {
    description = "Controls whether the ECS services and related resources are created. Set to false to create only the base infrastructure."
    type        = bool
    default     = true
}

# Desired Count (Initial number of tasks)
variable "api_desired_count" {
    description = "Initial desired number of tasks for the API service"
    type        = number
    default     = 1
}

variable "ui_desired_count" {
    description = "Initial desired number of tasks for the UI service"
    type        = number
    default     = 1
}

variable "mlflow_desired_count" {
    description = "Initial desired number of tasks for the MLflow service"
    type        = number
    default     = 1
}

variable "tensorboard_desired_count" {
    description = "Initial desired number of tasks for the TensorBoard service"
    type        = number
    default     = 1
}

# Auto Scaling
variable "api_min_capacity" {
    description = "Minimum number of tasks for the API service"
    type        = number
    default     = 1
}

variable "api_max_capacity" {
    description = "Maximum number of tasks for the API service"
    type        = number
    default     = 1
}

variable "ui_min_capacity" {
    description = "Minimum number of tasks for the UI service"
    type        = number
    default     = 1
}

variable "ui_max_capacity" {
    description = "Maximum number of tasks for the UI service"
    type        = number
    default     = 1
}

variable "mlflow_min_capacity" {
    description = "Minimum number of tasks for the MLflow service"
    type        = number
    default     = 1
}

variable "mlflow_max_capacity" {
    description = "Maximum number of tasks for the MLflow service"
    type        = number
    default     = 1
}

variable "tensorboard_min_capacity" {
    description = "Minimum number of tasks for the TensorBoard service"
    type        = number
    default     = 1
}

variable "tensorboard_max_capacity" {
    description = "Maximum number of tasks for the TensorBoard service"
    type        = number
    default     = 1
}

variable "scaling_cpu_target_percentage" {
    description = "The target CPU utilization percentage for ECS service auto-scaling."
    type        = number
    default     = 75
}

variable "scaling_cooldown_seconds_in" {
    description = "The scale-in cooldown period, in seconds."
    type        = number
    default     = 300
}

variable "scaling_cooldown_seconds_out" {
    description = "The scale-out cooldown period, in seconds."
    type        = number
    default     = 60
}
