# Makefile for AutoVisionAI
# Provides convenient commands for common development and deployment tasks.

# --- Variables ---
# Use SHELL to ensure that the Makefile uses bash, not the default shell.
SHELL := /bin/bash
# Default ECR URL - can be overridden from the command line.
ECR_REPOSITORY_URL ?= $(shell terraform -chdir=deploy/terraform output -raw ecr_repository_url)
# Use the short git commit hash as the default image tag.
IMAGE_TAG ?= $(shell git rev-parse --short HEAD)
# Default local repository name
LOCAL_REPOSITORY ?= autovisionai

# --- Phony Targets ---
# .PHONY ensures that these targets run even if files with the same name exist.
.PHONY: help build push build-push build-local

# --- Targets ---
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  help         Show this help message."
	@echo "  build        Build all Docker images and tag them locally."
	@echo "  build-local  Build all Docker images with local repository name."
	@echo "  push         Push the already-built images to ECR."
	@echo "  build-push   Run the build and push targets sequentially."

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
