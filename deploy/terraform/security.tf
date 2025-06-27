# security.tf

# ALB Security Group
resource "aws_security_group" "alb" {
  name        = "${var.project_name}-alb-security-group"
  description = "Controls access to the Application Load Balancer"
  vpc_id      = aws_vpc.main.id

  # HTTP access from anywhere
  ingress {
    protocol    = "tcp"
    from_port   = 80
    to_port     = 80
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTPS access from anywhere
  ingress {
    protocol    = "tcp"
    from_port   = 443
    to_port     = 443
    cidr_blocks = ["0.0.0.0/0"]
  }

  # API port access
  ingress {
    protocol    = "tcp"
    from_port   = var.api_port
    to_port     = var.api_port
    cidr_blocks = ["0.0.0.0/0"]
  }

  # UI port access
  ingress {
    protocol    = "tcp"
    from_port   = var.ui_port
    to_port     = var.ui_port
    cidr_blocks = ["0.0.0.0/0"]
  }

  # MLflow port access
  ingress {
    protocol    = "tcp"
    from_port   = var.mlflow_port
    to_port     = var.mlflow_port
    cidr_blocks = ["0.0.0.0/0"]
  }

  # TensorBoard port access
  ingress {
    protocol    = "tcp"
    from_port   = var.tensorboard_port
    to_port     = var.tensorboard_port
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-alb-sg"
  }
}

# ECS Tasks Security Group
resource "aws_security_group" "ecs_tasks" {
  name        = "${var.project_name}-ecs-tasks-security-group"
  description = "Controls access to and from ECS tasks"
  vpc_id      = aws_vpc.main.id

  # ALB → Tasks (API/UI/etc)
  ingress {
    protocol        = "tcp"
    from_port       = 0
    to_port         = 65535
    security_groups = [aws_security_group.alb.id]
    description     = "Allow ALB access to services"
  }


  # API port from ALB
  # ingress {
  #   protocol        = "tcp"
  #   from_port       = var.api_port
  #   to_port         = var.api_port
  #   security_groups = [aws_security_group.alb.id]
  # }

  # # UI port from ALB
  # ingress {
  #   protocol        = "tcp"
  #   from_port       = var.ui_port
  #   to_port         = var.ui_port
  #   security_groups = [aws_security_group.alb.id]
  # }

  # # MLflow port from ALB
  # ingress {
  #   protocol        = "tcp"
  #   from_port       = var.mlflow_port
  #   to_port         = var.mlflow_port
  #   security_groups = [aws_security_group.alb.id]
  # }

  # # TensorBoard port from ALB
  # ingress {
  #   protocol        = "tcp"
  #   from_port       = var.tensorboard_port
  #   to_port         = var.tensorboard_port
  #   security_groups = [aws_security_group.alb.id]
  # }

  # Allow traffic between ECS tasks (Service Connect & direct)
  ingress {
    protocol    = "tcp"
    from_port   = 0
    to_port     = 65535
    self        = true
    description = "Inter-service traffic within tasks"
  }

  # DNS egress
  egress {
    protocol    = "udp"
    from_port   = 53
    to_port     = 53
    cidr_blocks = ["0.0.0.0/0"]
    description = "DNS (UDP) for Cloud Map"
  }

  egress {
    protocol    = "tcp"
    from_port   = 53
    to_port     = 53
    cidr_blocks = ["0.0.0.0/0"]
    description = "DNS (TCP) for Cloud Map"
  }

  # Full outbound within VPC
  egress {
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = [aws_vpc.main.cidr_block]
    description = "Allow all outbound traffic within the VPC"
  }

  # Restricted egress - only allow necessary outbound traffic
  egress {
    protocol    = "tcp"
    from_port   = 443
    to_port     = 443
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTPS outbound for API calls"
  }

  egress {
    protocol    = "tcp"
    from_port   = 80
    to_port     = 80
    cidr_blocks = ["0.0.0.0/0"]
    description = "HTTP outbound for package downloads"
  }

  egress {
    protocol    = "tcp"
    from_port   = 2049
    to_port     = 2049
    cidr_blocks = [aws_vpc.main.cidr_block]
    description = "NFS access to EFS within VPC"
  }

  tags = {
    Name = "${var.project_name}-ecs-tasks-sg"
  }
}

# EFS Security Group
resource "aws_security_group" "efs" {
  name        = "${var.project_name}-efs-security-group"
  description = "Allow EFS access from ECS tasks"
  vpc_id      = aws_vpc.main.id

  ingress {
    protocol        = "tcp"
    from_port       = 2049
    to_port         = 2049
    security_groups = [aws_security_group.ecs_tasks.id]
  }

  egress {
    protocol    = "-1"
    from_port   = 0
    to_port     = 0
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-efs-sg"
  }
}
