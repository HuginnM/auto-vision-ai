# logs.tf

# CloudWatch Log Groups for all services
resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${var.api_service_name}"
  retention_in_days = 30

  tags = {
    Name    = "${var.project_name}-api-logs"
    Service = "API"
  }
}

resource "aws_cloudwatch_log_group" "ui" {
  name              = "/ecs/${var.ui_service_name}"
  retention_in_days = 30

  tags = {
    Name    = "${var.project_name}-ui-logs"
    Service = "UI"
  }
}

resource "aws_cloudwatch_log_group" "mlflow" {
  name              = "/ecs/${var.mlflow_service_name}"
  retention_in_days = 30

  tags = {
    Name    = "${var.project_name}-mlflow-logs"
    Service = "MLflow"
  }
}

resource "aws_cloudwatch_log_group" "tensorboard" {
  name              = "/ecs/${var.tensorboard_service_name}"
  retention_in_days = 30

  tags = {
    Name    = "${var.project_name}-tensorboard-logs"
    Service = "TensorBoard"
  }
}
