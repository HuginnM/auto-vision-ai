# efs.tf

# EFS File System for shared experiments data
resource "aws_efs_file_system" "experiments" {
  creation_token                  = "${var.project_name}-experiments"
  performance_mode                = "generalPurpose"
  throughput_mode                 = "provisioned"
  provisioned_throughput_in_mibps = 10
  encrypted                       = true

  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }

  tags = {
    Name = "${var.project_name}-experiments-efs"
  }
}

# EFS Mount Targets in each availability zone
resource "aws_efs_mount_target" "experiments" {
  count           = var.az_count
  file_system_id  = aws_efs_file_system.experiments.id
  subnet_id       = aws_subnet.private[count.index].id
  security_groups = [aws_security_group.efs.id]
}

# EFS Access Point for controlled access
resource "aws_efs_access_point" "experiments" {
  file_system_id = aws_efs_file_system.experiments.id

  posix_user {
    gid = 1000
    uid = 1000
  }

  root_directory {
    path = "/app/experiments"
    creation_info {
      owner_gid   = 1000
      owner_uid   = 1000
      permissions = "755"
    }
  }

  tags = {
    Name = "${var.project_name}-experiments-access-point"
  }
}
