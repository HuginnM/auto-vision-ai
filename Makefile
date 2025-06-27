# Makefile for AutoVisionAI
# Provides convenient commands for common development and deployment tasks.

# --- Variables ---
# Use SHELL to ensure that the Makefile uses bash, not the default shell.
SHELL := /bin/bash
# Default ECR URL - can be overridden from the command line.
# Use the short git commit hash as the default image tag.
IMAGE_TAG ?= $(shell git rev-parse --short HEAD)
AWS_REGION ?= us-west-1
ACCOUNT_ID ?= 869935094020
ECR_REPOSITORY_NAME ?= autovisionai
ECR_REPOSITORY_URL ?= $(ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/$(ECR_REPOSITORY_NAME)

TF_WORKING_DIR = deploy/terraform
WANDB_SECRET_NAME ?= autovisionai/wandb-api-key

# --- Phony Targets ---
# .PHONY ensures that these targets run even if files with the same name exist.
.PHONY: help build push build-push build-local deploy-infra deploy-services deploy-full deploy-with-secrets push-secrets destroy

# --- Targets ---
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Local Development Targets:"
	@echo "  help         Show this help message."
	@echo "  build        Build all Docker images and tag them locally."
	@echo "  build-local  Build all Docker images with local repository name."
	@echo "  push         Push the already-built images to ECR."
	@echo "  build-push   Run the build and push targets sequentially."
	@echo "  push-secrets Push secrets from .env to AWS Secrets Manager."
	@echo "  deploy-infra Deploy base infrastructure (VPC, ECR, ECS cluster, etc.)."
	@echo "  deploy-services Deploy application services (requires infrastructure)."
	@echo "  deploy-full  Complete deployment from local: infrastructure, secrets, build, push, services."
	@echo "  destroy      Destroy all infrastructure and resources (DANGEROUS!)."

build:
	@echo "--- Building Docker Images for ECR ---"
	@echo "ECR Repository: $(ECR_REPOSITORY_NAME)"
	@echo "Image Tag: $(IMAGE_TAG)"

	@chmod +x scripts/build-images.sh
	@./scripts/build-images.sh $(IMAGE_TAG) $(ECR_REPOSITORY_URL)

push:
	@echo "--- Pushing Docker Images to ECR ---"
	@echo "ECR Repository: $(ECR_REPOSITORY_NAME)"
	@echo "Image Tag: $(IMAGE_TAG)"

	@chmod +x scripts/push-images.sh
	@./scripts/push-images.sh $(IMAGE_TAG) $(ECR_REPOSITORY_URL)

build-push: build push
	@echo "--- Build and push complete ---"

deploy-infra:
	@echo "--- Deploying Base Infrastructure ---"
	@echo "AWS Region: $(AWS_REGION)"
	@echo "Working Directory: $(TF_WORKING_DIR)"

	@terraform -chdir=$(TF_WORKING_DIR) init
	@terraform -chdir=$(TF_WORKING_DIR) plan -var="create_ecs_services=false" -out=tfplan
	@terraform -chdir=$(TF_WORKING_DIR) apply -auto-approve tfplan

	@echo "--- Infrastructure deployment complete ---"

deploy-services:
	@echo "--- Deploying Application Services ---"
	@echo "AWS Region: $(AWS_REGION)"
	@echo "Working Directory: $(TF_WORKING_DIR)"

	@terraform -chdir=$(TF_WORKING_DIR) init
	@terraform -chdir=$(TF_WORKING_DIR) plan -var="create_ecs_services=true" -out=tfplan
	@terraform -chdir=$(TF_WORKING_DIR) apply -auto-approve tfplan

	@echo "--- Services deployment complete ---"
	@echo "Service URLs:"
	@echo "  API: $(shell cd $(TF_WORKING_DIR) && terraform output -raw api_url)"
	@echo "  UI: $(shell cd $(TF_WORKING_DIR) && terraform output -raw ui_url)"
	@echo "  MLflow: $(shell cd $(TF_WORKING_DIR) && terraform output -raw mlflow_url)"
	@echo "  TensorBoard: $(shell cd $(TF_WORKING_DIR) && terraform output -raw tensorboard_url)"

push-secrets:
	@echo "--- Pushing Secrets from .env to AWS Secrets Manager ---"
	@chmod +x scripts/push-secrets.sh
	@./scripts/push-secrets.sh $(AWS_REGION) $(WANDB_SECRET_NAME)

deploy-full: deploy-infra push-secrets build-push deploy-services
	@echo "--- Complete deployment with secrets finished ---"
	@echo "All services are now running!"

destroy:
	@echo "--- Destroying AutoVisionAI Infrastructure ---"
	@echo "WARNING: This will permanently delete all resources!"
	@echo "This includes:"
	@echo "  - ECS cluster and services"
	@echo "  - ECR repository and images"
	@echo "  - S3 bucket and data"
	@echo "  - VPC, subnets, and security groups"
	@echo "  - Load balancer and target groups"
	@echo "  - IAM roles and policies"
	@echo "  - Secrets Manager secrets"
	@echo "  - All other AWS resources"
	@echo ""

	@chmod +x scripts/destroy-infrastructure.sh
	@./scripts/destroy-infrastructure.sh $(AWS_REGION)
