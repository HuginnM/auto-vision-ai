# provider.tf

terraform {
  required_version = ">= 1.11.0"  # Required for stable S3 native locking

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # S3 Backend with native locking
  backend "s3" {
    bucket = "autovisionai-terraform-state"  # Set by init or via -backend-config
    key          = "terraform.tfstate"
    region       = "us-west-1"
    encrypt      = true
    use_lockfile = true  # S3 native locking - GA since Terraform 1.11.0
  }
}

# Configure the AWS Provider
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}
