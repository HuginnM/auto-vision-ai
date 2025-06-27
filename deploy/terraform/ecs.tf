# ecs.tf

# Data source to get current AWS account ID
data "aws_caller_identity" "current" {}

# --- Service Connect Namespace ---
resource "aws_service_discovery_private_dns_namespace" "service_connect" {
  name = "${var.project_name}.local"
  vpc  = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-sc-namespace"
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = var.cluster_name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  # Enable Service Connect for the cluster
  service_connect_defaults {
    namespace = aws_service_discovery_private_dns_namespace.service_connect.arn
  }

  tags = {
    Name = "${var.project_name}-cluster"
  }
}

# API Service
resource "aws_ecs_task_definition" "api" {
  count                    = var.create_ecs_services ? 1 : 0
  family                   = var.api_service_name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.api_cpu
  memory                   = var.api_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn           = aws_iam_role.ecs_task_role.arn

  volume {
    name = "experiments-volume"
    efs_volume_configuration {
      file_system_id          = aws_efs_file_system.experiments.id
      root_directory          = "/"
      transit_encryption      = "ENABLED"
      transit_encryption_port = 2049
      authorization_config {
        access_point_id = aws_efs_access_point.experiments.id
        iam             = "ENABLED"
      }
    }
  }

  container_definitions = jsonencode([
    {
      name      = "api"
      image     = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/${var.ecr_repository_name}:app"
      essential = true
      command = [
        "uv", "run",
        "uvicorn", "autovisionai.api.main:app",
        "--host", "0.0.0.0",
        "--port", tostring(var.api_port)
      ]
      portMappings = [
        {
          containerPort = var.api_port
          name          = "api"
        }
      ]
      environment = [
        { name = "ENV_MODE", value = var.env_mode },
        { name = "PYTHONUNBUFFERED", value = "1" },
        { name = "WANDB_ENTITY", value = var.wandb_entity }
      ]
      secrets = [
        {
          name      = "WANDB_API_KEY"
          valueFrom = "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:${var.wandb_secret_name}"
        }
      ]
      mountPoints = [
        {
          sourceVolume  = "experiments-volume"
          containerPath = "/app/experiments"
          readOnly      = false
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.api.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
      healthCheck = {
        command = ["CMD-SHELL", "curl -f http://localhost:${var.api_port}/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 15
      }
    }
  ])

  tags = {
    Name = "${var.project_name}-api-task"
  }
}

resource "aws_ecs_service" "api" {
  count           = var.create_ecs_services ? 1 : 0
  name            = var.api_service_name
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api[0].arn
  desired_count   = var.api_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks.id]
    subnets          = aws_subnet.private[*].id
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api[0].arn
    container_name   = "api"
    container_port   = var.api_port
  }

  # --- Service Connect ---
  service_connect_configuration {
    enabled   = true
    namespace = aws_service_discovery_private_dns_namespace.service_connect.arn

    service {
      port_name      = "api"
      discovery_name = "api"

      client_alias {
        dns_name = "api"
        port     = var.api_port
      }
    }
  }

  depends_on = [
    aws_lb_listener.api,
    aws_efs_mount_target.experiments
  ]

  tags = {
    Name = "${var.project_name}-api-service"
  }
}

# UI Service
resource "aws_ecs_task_definition" "ui" {
  count                    = var.create_ecs_services ? 1 : 0
  family                   = var.ui_service_name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.ui_cpu
  memory                   = var.ui_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn           = aws_iam_role.ecs_task_role.arn

  volume {
    name = "experiments-volume"
    efs_volume_configuration {
      file_system_id          = aws_efs_file_system.experiments.id
      root_directory          = "/"
      transit_encryption      = "ENABLED"
      transit_encryption_port = 2049
      authorization_config {
        access_point_id = aws_efs_access_point.experiments.id
        iam             = "ENABLED"
      }
    }
  }

  container_definitions = jsonencode([
    {
      name      = "ui"
      image     = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/${var.ecr_repository_name}:app"
      essential = true
      command = [
        "uv", "run",
        "streamlit", "run",
        "src/autovisionai/ui/app.py",
        "--server.port", tostring(var.ui_port),
        "--server.address", "0.0.0.0"
      ]
      portMappings = [
        {
          containerPort = var.ui_port
          name          = "ui"
        }
      ]
      environment = [
        { name = "ENV_MODE", value = var.env_mode },
        { name = "PYTHONUNBUFFERED", value = "1" },
        { name = "WANDB_ENTITY", value = var.wandb_entity }
      ]
      secrets = [
        {
          name      = "WANDB_API_KEY"
          valueFrom = "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:${var.wandb_secret_name}"
        }
      ]
      mountPoints = [
        {
          sourceVolume  = "experiments-volume"
          containerPath = "/app/experiments"
          readOnly      = true
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ui.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
      healthCheck = {
        command = ["CMD-SHELL", "curl -f http://localhost:${var.ui_port}/_stcore/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 30
      }
    }
  ])

  tags = {
    Name = "${var.project_name}-ui-task"
  }
}

resource "aws_ecs_service" "ui" {
  count           = var.create_ecs_services ? 1 : 0
  name            = var.ui_service_name
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.ui[0].arn
  desired_count   = var.ui_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks.id]
    subnets          = aws_subnet.private[*].id
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.ui[0].arn
    container_name   = "ui"
    container_port   = var.ui_port
  }

  # --- Service Connect ---
  service_connect_configuration {
    enabled   = true
    namespace = aws_service_discovery_private_dns_namespace.service_connect.arn

    service {
      port_name      = "ui"
      discovery_name = "ui"

      client_alias {
        dns_name = "ui"
        port     = var.ui_port
      }
    }
  }

  depends_on = [
    aws_lb_listener.ui,
    aws_efs_mount_target.experiments
  ]

  tags = {
    Name = "${var.project_name}-ui-service"
  }
}

# MLflow Service
resource "aws_ecs_task_definition" "mlflow" {
  count                    = var.create_ecs_services ? 1 : 0
  family                   = var.mlflow_service_name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.mlflow_cpu
  memory                   = var.mlflow_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn           = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name      = "mlflow"
      image     = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/${var.ecr_repository_name}:mlflow"
      essential = true
      command = [
        "mlflow", "server",
        "--host", "0.0.0.0",
        "--port", tostring(var.mlflow_port),
        "--backend-store-uri", "/app/mlruns",
        "--default-artifact-root", "s3://${var.mlflow_artifacts_bucket}/"
      ]
      portMappings = [
        {
          containerPort = var.mlflow_port
          name          = "mlflow"
        }
      ]
      environment = [
        { name = "PYTHONUNBUFFERED", value = "1" },
        { name = "AWS_DEFAULT_REGION", value = var.aws_region }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.mlflow.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
      healthCheck = {
        command = ["CMD-SHELL", "curl -f http://localhost:${var.mlflow_port}/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 15
      }
    }
  ])

  tags = {
    Name = "${var.project_name}-mlflow-task"
  }
}

resource "aws_ecs_service" "mlflow" {
  count           = var.create_ecs_services ? 1 : 0
  name            = var.mlflow_service_name
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.mlflow[0].arn
  desired_count   = var.mlflow_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks.id]
    subnets          = aws_subnet.private[*].id
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.mlflow[0].arn
    container_name   = "mlflow"
    container_port   = var.mlflow_port
  }

  # --- Service Connect ---
  service_connect_configuration {
    enabled   = true
    namespace = aws_service_discovery_private_dns_namespace.service_connect.arn

    service {
      port_name      = "mlflow"
      discovery_name = "mlflow"

      client_alias {
        dns_name = "mlflow"
        port     = var.mlflow_port
      }
    }
  }

  depends_on = [aws_lb_listener.mlflow]

  tags = {
    Name = "${var.project_name}-mlflow-service"
  }
}

# TensorBoard Service
resource "aws_ecs_task_definition" "tensorboard" {
  count                    = var.create_ecs_services ? 1 : 0
  family                   = var.tensorboard_service_name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.tensorboard_cpu
  memory                   = var.tensorboard_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn           = aws_iam_role.ecs_task_role.arn

  volume {
    name = "experiments-volume"
    efs_volume_configuration {
      file_system_id          = aws_efs_file_system.experiments.id
      root_directory          = "/"
      transit_encryption      = "ENABLED"
      transit_encryption_port = 2049
      authorization_config {
        access_point_id = aws_efs_access_point.experiments.id
        iam             = "ENABLED"
      }
    }
  }

  container_definitions = jsonencode([
    {
      name      = "tensorboard"
      image     = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/${var.ecr_repository_name}:tensorboard"
      essential = true
      command = [
        "tensorboard",
        "--logdir", "/app/experiments",
        "--host", "0.0.0.0",
        "--port", tostring(var.tensorboard_port)
      ]
      portMappings = [
        {
          containerPort = var.tensorboard_port
          name          = "tensorboard"
        }
      ]
      environment = [
        { name = "PYTHONUNBUFFERED", value = "1" }
      ]
      mountPoints = [
        {
          sourceVolume  = "experiments-volume"
          containerPath = "/app/experiments"
          readOnly      = false
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.tensorboard.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
      healthCheck = {
        command = ["CMD-SHELL", "curl -f http://localhost:${var.tensorboard_port} || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 10
      }
    }
  ])

  tags = {
    Name = "${var.project_name}-tensorboard-task"
  }
}

resource "aws_ecs_service" "tensorboard" {
  count           = var.create_ecs_services ? 1 : 0
  name            = var.tensorboard_service_name
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.tensorboard[0].arn
  desired_count   = var.tensorboard_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    security_groups  = [aws_security_group.ecs_tasks.id]
    subnets          = aws_subnet.private[*].id
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.tensorboard[0].arn
    container_name   = "tensorboard"
    container_port   = var.tensorboard_port
  }

  # --- Service Connect ---
  service_connect_configuration {
    enabled   = true
    namespace = aws_service_discovery_private_dns_namespace.service_connect.arn

    service {
      port_name      = "tensorboard"
      discovery_name = "tensorboard"

      client_alias {
        dns_name = "tensorboard"
        port     = var.tensorboard_port
      }
    }
  }

  depends_on = [
    aws_lb_listener.tensorboard,
    aws_efs_mount_target.experiments
  ]

  tags = {
    Name = "${var.project_name}-tensorboard-service"
  }
}
