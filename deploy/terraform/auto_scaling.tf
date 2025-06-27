# auto_scaling.tf

# This local map defines all the services that should be configured for auto-scaling.
# The for_each meta-argument in the resources below will iterate over this map.
locals {
  autoscaled_services = var.create_ecs_services ? {
    api         = { service = aws_ecs_service.api[0], min = var.api_min_capacity, max = var.api_max_capacity },
    ui          = { service = aws_ecs_service.ui[0], min = var.ui_min_capacity, max = var.ui_max_capacity },
    mlflow      = { service = aws_ecs_service.mlflow[0], min = var.mlflow_min_capacity, max = var.mlflow_max_capacity },
    tensorboard = { service = aws_ecs_service.tensorboard[0], min = var.tensorboard_min_capacity, max = var.tensorboard_max_capacity }
  } : {}
}

# Create a single scaling target resource that applies to each service defined in the locals map.
resource "aws_appautoscaling_target" "ecs_target" {
  # The for_each loop ensures this resource is created for each service, but only if ecs services are being deployed.
  for_each = local.autoscaled_services

  service_namespace  = "ecs"
  scalable_dimension = "ecs:service:DesiredCount"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${each.value.service.name}"
  min_capacity       = each.value.min
  max_capacity       = each.value.max
  role_arn          = aws_iam_role.ecs_auto_scale_role.arn
}

# Create a single CPU utilization tracking policy that applies to each service.
# This is much simpler than managing separate scale-up/scale-down policies and CloudWatch alarms.
resource "aws_appautoscaling_policy" "ecs_cpu_scaling_policy" {
  for_each = local.autoscaled_services

  name               = "${each.value.service.name}-cpu-target-tracking"
  service_namespace  = aws_appautoscaling_target.ecs_target[each.key].service_namespace
  scalable_dimension = aws_appautoscaling_target.ecs_target[each.key].scalable_dimension
  resource_id        = aws_appautoscaling_target.ecs_target[each.key].resource_id
  policy_type        = "TargetTrackingScaling"

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value       = var.scaling_cpu_target_percentage
    scale_in_cooldown  = var.scaling_cooldown_seconds_in
    scale_out_cooldown = var.scaling_cooldown_seconds_out
  }

  depends_on = [aws_appautoscaling_target.ecs_target]
}
