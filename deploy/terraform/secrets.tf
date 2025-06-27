# secrets.tf

# WANDB API Key Secret
resource "aws_secretsmanager_secret" "wandb_api_key" {
  name                    = var.wandb_secret_name
  description             = "WANDB API Key for AutoVisionAI"
  recovery_window_in_days = 7

  tags = {
    Name        = "${var.project_name}-wandb-api-key"
    Environment = var.environment
  }
}

# Secret Version (placeholder - actual value should be set manually or via CI/CD)
resource "aws_secretsmanager_secret_version" "wandb_api_key" {
  secret_id     = aws_secretsmanager_secret.wandb_api_key.id
  secret_string = "placeholder-value-set-this-manually"

  lifecycle {
    ignore_changes = [secret_string]
  }
}
