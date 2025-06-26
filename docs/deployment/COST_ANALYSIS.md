# AutoVisionAI Infrastructure Cost Analysis

*Last Updated: {{ CURRENT_DATE }}*
*Based on infrastructure defined in `deploy/terraform`.*

---

## 💰 Estimated Monthly Cost: ~$171.55

### **Annual Cost Estimate: ~$2,058.60**

This estimate is based on the `us-west-1` (N. California) region and assumes continuous operation (730 hours/month). Costs are variable and depend on actual usage.

---

## 📊 Monthly Cost Breakdown

| Category | Service | Configuration | Monthly Cost |
|----------|---------|---------------|--------------|
| **Compute** | **ECS Fargate** | | **$87.60** |
| | API Service | 1 vCPU, 2 GB RAM | $29.20 |
| | UI Service | 1 vCPU, 2 GB RAM | $29.20 |
| | MLflow Service | 0.5 vCPU, 1 GB RAM | $14.60 |
| | TensorBoard | 0.5 vCPU, 1 GB RAM | $14.60 |
| **Networking** | **ALB & NAT** | | **$67.50** |
| | App. Load Balancer | 1 instance, 0.5 LCU | $21.90 |
| | NAT Gateway | 1 Gateway, 5GB processed | $45.60 |
| **Storage** | **EFS & S3** | | **~$10.32** |
| | EFS Provisioned | 10 MiB/s throughput | $6.00 |
| | EFS Storage | 10 GB Standard-IA | ~$3.00 |
| | S3 Storage | 5 GB Standard | $0.12 |
| | ECR Storage | 5 GB of images | $0.50 |
| | CloudWatch Logs | 5 GB ingested/stored | $0.70 |
| **Databases/Other**| **Secrets Manager** | | **$0.40** |
| | Secrets Manager | 1 secret | $0.40 |
| | **TOTAL** | | **~$171.55** |

---

## ✅ Key Cost Optimizations Implemented

### 1. Single NAT Gateway
- **Strategy**: Consolidated all private subnet egress traffic through a single NAT Gateway instead of one per Availability Zone.
- **Impact**: **Reduced NAT Gateway costs by 50%**, saving approximately **$45.60/month**. This is the most significant cost-saving measure in the architecture.

### 2. Provisioned EFS Throughput
- **Strategy**: Switched EFS from the default "Bursting" mode to "Provisioned" throughput, setting it to a baseline of **10 MiB/s**.
- **Impact**: Provides predictable performance at a fixed, lower cost (~$6.00/month) compared to the variable and potentially high costs of bursting credits. Ideal for consistent workloads.

### 3. Fargate Right-Sizing
- **Strategy**: Selected specific vCPU and Memory configurations for each service based on expected load (e.g., smaller instances for MLflow/TensorBoard).
- **Impact**: Prevents over-provisioning and ensures we only pay for the compute capacity we need.

### 4. EFS Intelligent-Tiering
- **Strategy**: Implemented a lifecycle policy to automatically move EFS files to the cost-effective Infrequent Access (IA) storage class after 30 days.
- **Impact**: Reduces storage costs for older, less-frequently-accessed experiment data by up to 92%.

---

## ⚠️ Cost Monitoring & Alerts

It is critical to monitor costs to avoid unexpected charges.

- **AWS Budgets**: Set up a monthly cost budget for the entire account to receive alerts when spending exceeds predefined thresholds.
- **CloudWatch Alarms**: Create alarms on specific high-cost metrics:
  - `NATGatewayAddressProcessedBytes`: To monitor data processing, which is the primary driver of NAT Gateway costs.
  - `ProvisionedThroughput`: For EFS, to ensure usage doesn't exceed the provisioned amount.
  - `CPUUtilization`: For ECS services, to help with right-sizing decisions.

---
