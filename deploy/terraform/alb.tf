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
  count       = var.create_ecs_services ? 1 : 0
  name        = "${var.api_service_name}-tg"
  port        = var.api_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 3
    interval            = 30
    matcher             = "200-299"
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
  count       = var.create_ecs_services ? 1 : 0
  name        = "${var.ui_service_name}-tg"
  port        = var.ui_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 3
    interval            = 30
    matcher             = "200-299"
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
  count       = var.create_ecs_services ? 1 : 0
  name        = "${var.mlflow_service_name}-tg"
  port        = var.mlflow_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 3
    interval            = 30
    matcher             = "200-299"
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
  count       = var.create_ecs_services ? 1 : 0
  name        = "${var.tensorboard_service_name}-tg"
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
  count             = var.create_ecs_services ? 1 : 0
  load_balancer_arn = aws_lb.main.arn
  port              = var.api_port
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api[0].arn
  }
}

resource "aws_lb_listener" "ui" {
  count             = var.create_ecs_services ? 1 : 0
  load_balancer_arn = aws_lb.main.arn
  port              = var.ui_port
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ui[0].arn
  }
}

resource "aws_lb_listener" "mlflow" {
  count             = var.create_ecs_services ? 1 : 0
  load_balancer_arn = aws_lb.main.arn
  port              = var.mlflow_port
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.mlflow[0].arn
  }
}

resource "aws_lb_listener" "tensorboard" {
  count             = var.create_ecs_services ? 1 : 0
  load_balancer_arn = aws_lb.main.arn
  port              = var.tensorboard_port
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.tensorboard[0].arn
  }
}
