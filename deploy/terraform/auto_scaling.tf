# auto_scaling.tf

# Auto Scaling Target for API Service
resource "aws_appautoscaling_target" "api" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  role_arn          = aws_iam_role.ecs_auto_scale_role.arn
  min_capacity      = 1
  max_capacity      = 4

  tags = {
    Name = "${var.project_name}-api-autoscaling-target"
  }
}

# Scale Up Policy for API
resource "aws_appautoscaling_policy" "api_up" {
  name               = "${var.project_name}-api-scale-up"
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.api.name}"
  scalable_dimension = "ecs:service:DesiredCount"

  step_scaling_policy_configuration {
    adjustment_type         = "ChangeInCapacity"
    cooldown               = 60
    metric_aggregation_type = "Maximum"

    step_adjustment {
      metric_interval_lower_bound = 0
      scaling_adjustment          = 1
    }
  }

  depends_on = [aws_appautoscaling_target.api]
}

# Scale Down Policy for API
resource "aws_appautoscaling_policy" "api_down" {
  name               = "${var.project_name}-api-scale-down"
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.api.name}"
  scalable_dimension = "ecs:service:DesiredCount"

  step_scaling_policy_configuration {
    adjustment_type         = "ChangeInCapacity"
    cooldown               = 60
    metric_aggregation_type = "Maximum"

    step_adjustment {
      metric_interval_upper_bound = 0
      scaling_adjustment          = -1
    }
  }

  depends_on = [aws_appautoscaling_target.api]
}

# CloudWatch Alarm - API High CPU
resource "aws_cloudwatch_metric_alarm" "api_cpu_high" {
  alarm_name          = "${var.project_name}-api-cpu-high"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ECS"
  period              = "60"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors api cpu utilization"

  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
    ServiceName = aws_ecs_service.api.name
  }

  alarm_actions = [aws_appautoscaling_policy.api_up.arn]

  tags = {
    Name = "${var.project_name}-api-cpu-high-alarm"
  }
}

# CloudWatch Alarm - API Low CPU
resource "aws_cloudwatch_metric_alarm" "api_cpu_low" {
  alarm_name          = "${var.project_name}-api-cpu-low"
  comparison_operator = "LessThanOrEqualToThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ECS"
  period              = "60"
  statistic           = "Average"
  threshold           = "10"
  alarm_description   = "This metric monitors api cpu utilization"

  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
    ServiceName = aws_ecs_service.api.name
  }

  alarm_actions = [aws_appautoscaling_policy.api_down.arn]

  tags = {
    Name = "${var.project_name}-api-cpu-low-alarm"
  }
}

# Auto Scaling Target for UI Service
resource "aws_appautoscaling_target" "ui" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.ui.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  role_arn          = aws_iam_role.ecs_auto_scale_role.arn
  min_capacity      = 1
  max_capacity      = 3

  tags = {
    Name = "${var.project_name}-ui-autoscaling-target"
  }
}

# Scale Up Policy for UI
resource "aws_appautoscaling_policy" "ui_up" {
  name               = "${var.project_name}-ui-scale-up"
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.ui.name}"
  scalable_dimension = "ecs:service:DesiredCount"

  step_scaling_policy_configuration {
    adjustment_type         = "ChangeInCapacity"
    cooldown               = 60
    metric_aggregation_type = "Maximum"

    step_adjustment {
      metric_interval_lower_bound = 0
      scaling_adjustment          = 1
    }
  }

  depends_on = [aws_appautoscaling_target.ui]
}

# Scale Down Policy for UI
resource "aws_appautoscaling_policy" "ui_down" {
  name               = "${var.project_name}-ui-scale-down"
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.ui.name}"
  scalable_dimension = "ecs:service:DesiredCount"

  step_scaling_policy_configuration {
    adjustment_type         = "ChangeInCapacity"
    cooldown               = 60
    metric_aggregation_type = "Maximum"

    step_adjustment {
      metric_interval_upper_bound = 0
      scaling_adjustment          = -1
    }
  }

  depends_on = [aws_appautoscaling_target.ui]
}

# CloudWatch Alarm - UI High CPU
resource "aws_cloudwatch_metric_alarm" "ui_cpu_high" {
  alarm_name          = "${var.project_name}-ui-cpu-high"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ECS"
  period              = "60"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors ui cpu utilization"

  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
    ServiceName = aws_ecs_service.ui.name
  }

  alarm_actions = [aws_appautoscaling_policy.ui_up.arn]

  tags = {
    Name = "${var.project_name}-ui-cpu-high-alarm"
  }
}

# CloudWatch Alarm - UI Low CPU
resource "aws_cloudwatch_metric_alarm" "ui_cpu_low" {
  alarm_name          = "${var.project_name}-ui-cpu-low"
  comparison_operator = "LessThanOrEqualToThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ECS"
  period              = "60"
  statistic           = "Average"
  threshold           = "10"
  alarm_description   = "This metric monitors ui cpu utilization"

  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
    ServiceName = aws_ecs_service.ui.name
  }

  alarm_actions = [aws_appautoscaling_policy.ui_down.arn]

  tags = {
    Name = "${var.project_name}-ui-cpu-low-alarm"
  }
}
