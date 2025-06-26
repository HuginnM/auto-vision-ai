# alb.tf

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  enable_deletion_protection = false

  tags = {
    Name = "${var.project_name}-alb"
  }
}

# Target Groups
resource "aws_lb_target_group" "api" {
  name        = "${var.project_name}-api-tg"
  port        = var.api_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 3
    interval            = 30
    matcher             = "200"
    path                = var.api_health_check_path
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = {
    Name = "${var.project_name}-api-tg"
  }
}

resource "aws_lb_target_group" "ui" {
  name        = "${var.project_name}-ui-tg"
  port        = var.ui_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 3
    interval            = 30
    matcher             = "200"
    path                = var.ui_health_check_path
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = {
    Name = "${var.project_name}-ui-tg"
  }
}

resource "aws_lb_target_group" "mlflow" {
  name        = "${var.project_name}-mlflow-tg"
  port        = var.mlflow_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 3
    interval            = 30
    matcher             = "200"
    path                = var.mlflow_health_check_path
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = {
    Name = "${var.project_name}-mlflow-tg"
  }
}

resource "aws_lb_target_group" "tensorboard" {
  name        = "${var.project_name}-tensorboard-tg"
  port        = var.tensorboard_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 3
    interval            = 30
    matcher             = "200"
    path                = "/"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = {
    Name = "${var.project_name}-tensorboard-tg"
  }
}

# Listeners
resource "aws_lb_listener" "api" {
  load_balancer_arn = aws_lb.main.arn
  port              = var.api_port
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

resource "aws_lb_listener" "ui" {
  load_balancer_arn = aws_lb.main.arn
  port              = var.ui_port
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ui.arn
  }
}

resource "aws_lb_listener" "mlflow" {
  load_balancer_arn = aws_lb.main.arn
  port              = var.mlflow_port
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.mlflow.arn
  }
}

resource "aws_lb_listener" "tensorboard" {
  load_balancer_arn = aws_lb.main.arn
  port              = var.tensorboard_port
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.tensorboard.arn
  }
}
