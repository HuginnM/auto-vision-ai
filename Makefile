# Makefile for AutoVisionAI
# Provides convenient commands for common development and deployment tasks.

# --- Variables ---
# Use SHELL to ensure that the Makefile uses bash, not the default shell.
SHELL := /bin/bash
# Default ECR URL - can be overridden from the command line.
ECR_REPOSITORY_URL ?= $(shell terraform -chdir=deploy/terraform output -raw ecr_repository_url)
# Use the short git commit hash as the default image tag.
IMAGE_TAG ?= $(shell git rev-parse --short HEAD)
LOCAL_REPOSITORY ?= autovisionai
AWS_REGION ?= us-west-1
TF_WORKING_DIR = deploy/terraform
# Secrets configuration
WANDB_SECRET_NAME ?= autovisionai/wandb-api-key

# --- Phony Targets ---
# .PHONY ensures that these targets run even if files with the same name exist.
.PHONY: help build push build-push build-local deploy-infra deploy-services deploy-full deploy-with-secrets push-secrets

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

build:
	@echo "--- Building Docker Images for ECR ---"
	@echo "ECR Repository: $(ECR_REPOSITORY_URL)"
	@echo "Image Tag: $(IMAGE_TAG)"

	@chmod +x scripts/build-images.sh
	@./scripts/build-images.sh $(IMAGE_TAG) $(ECR_REPOSITORY_URL)

build-local:
	@echo "--- Building Docker Images Locally ---"
	@echo "Local Repository: $(LOCAL_REPOSITORY)"
	@echo "Image Tag: $(IMAGE_TAG)"

	@chmod +x scripts/build-images.sh
	@./scripts/build-images.sh $(IMAGE_TAG) $(LOCAL_REPOSITORY)

push:
	@echo "--- Pushing Docker Images to ECR ---"
	@echo "ECR Repository: $(ECR_REPOSITORY_URL)"
	@echo "Image Tag: $(IMAGE_TAG)"

	@chmod +x scripts/push-images.sh
	@./scripts/push-images.sh $(ECR_REPOSITORY_URL) $(IMAGE_TAG)

build-push: build push
	@echo "--- Build and push complete ---"

deploy-infra:
	@echo "--- Deploying Base Infrastructure ---"
	@echo "AWS Region: $(AWS_REGION)"
	@echo "Working Directory: $(TF_WORKING_DIR)"

	@cd $(TF_WORKING_DIR) && terraform init
	@cd $(TF_WORKING_DIR) && terraform plan -var="create_ecs_services=false" -out=tfplan
	@cd $(TF_WORKING_DIR) && terraform apply -auto-approve tfplan

	@echo "--- Infrastructure deployment complete ---"
	@echo "ECR Repository URL: $(shell cd $(TF_WORKING_DIR) && terraform output -raw ecr_repository_url)"

deploy-services:
	@echo "--- Deploying Application Services ---"
	@echo "AWS Region: $(AWS_REGION)"
	@echo "Working Directory: $(TF_WORKING_DIR)"

	@cd $(TF_WORKING_DIR) && terraform init
	@cd $(TF_WORKING_DIR) && terraform plan -var="create_ecs_services=true" -out=tfplan
	@cd $(TF_WORKING_DIR) && terraform apply -auto-approve tfplan

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
