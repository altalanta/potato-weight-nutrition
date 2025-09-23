# Infrastructure as Code (Terraform)

This directory contains Terraform configuration for provisioning AWS resources for the potato-weight-nutrition pipeline.

## Resources Provisioned

### S3 Buckets
- **Data Lake Bucket**: Stores bronze/silver/gold data layers
- **MLflow Artifacts Bucket**: Stores MLflow model artifacts and experiment data

### DynamoDB
- **MLflow Backend Table**: Backend store for MLflow experiment tracking

### IAM
- **GitHub OIDC Provider**: Enables GitHub Actions to assume AWS roles
- **GitHub OIDC Role**: IAM role with least-privilege access to S3 and DynamoDB
- **IAM Policies**: Granular permissions for S3, DynamoDB, and CloudWatch Logs

## Quick Start

### Prerequisites
- AWS CLI configured with appropriate credentials
- Terraform >= 1.5 installed
- AWS account with permissions to create IAM roles, S3 buckets, and DynamoDB tables

### Deploy Infrastructure

1. **Initialize Terraform**:
   ```bash
   cd infra/terraform
   terraform init
   ```

2. **Plan the deployment**:
   ```bash
   terraform plan -var="project_prefix=cdp-dev" -var="aws_region=us-east-1"
   ```

3. **Apply the configuration**:
   ```bash
   terraform apply -var="project_prefix=cdp-dev" -var="aws_region=us-east-1"
   ```

4. **Get outputs**:
   ```bash
   terraform output
   ```

### Configuration Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `aws_region` | AWS region for resources | `us-east-1` | No |
| `project_prefix` | Prefix for resource names | `cdp-dev` | No |
| `github_repo` | GitHub repository (`owner/repo`) | `*` | No |
| `environment` | Environment name (dev/prod) | `dev` | No |

### Example with custom variables

```bash
terraform apply \
  -var="project_prefix=potato-prod" \
  -var="aws_region=us-west-2" \
  -var="github_repo=yourusername/potato-weight-nutrition" \
  -var="environment=prod"
```

## GitHub OIDC Configuration

The Terraform creates a GitHub OIDC provider and IAM role that allows GitHub Actions to securely access AWS resources without storing long-lived credentials.

### Trust Policy
The IAM role trusts GitHub Actions from the specified repository with these conditions:
- Audience must be `sts.amazonaws.com`
- Subject must match `repo:OWNER/REPO:*`

### Required GitHub Repository Settings

After applying Terraform, configure these secrets in your GitHub repository:

1. Go to **Settings** > **Secrets and variables** > **Actions**
2. Add these secrets:
   - `AWS_ROLE_ARN`: Use the `github_oidc_role_arn` output
   - `AWS_REGION`: Use the `aws_region` variable value

### GitHub Actions Workflow Example

```yaml
- name: Configure AWS credentials
  uses: aws-actions/configure-aws-credentials@v4
  with:
    role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
    aws-region: ${{ secrets.AWS_REGION }}
    role-session-name: GitHubActions
```

## Outputs

The Terraform configuration provides these outputs for use in CI/CD and application configuration:

- `data_lake_bucket_name`: S3 bucket for data storage
- `mlflow_artifacts_bucket_name`: S3 bucket for MLflow artifacts
- `mlflow_backend_table_name`: DynamoDB table for MLflow backend
- `github_oidc_role_arn`: IAM role ARN for GitHub Actions
- `env_vars`: Complete environment variable configuration

## Environment Variables for Pipeline

Use the `env_vars` output to configure your pipeline:

```bash
# Get environment variables
terraform output -json env_vars

# Example usage in GitHub Actions
echo "DATA_LAKE_BUCKET=$(terraform output -raw data_lake_bucket_name)" >> $GITHUB_ENV
echo "MLFLOW_ARTIFACTS_DESTINATION=$(terraform output -raw mlflow_artifacts_bucket_name)" >> $GITHUB_ENV
```

## Security Features

- **Encryption**: All S3 buckets use AES-256 server-side encryption
- **Public Access**: S3 buckets block all public access
- **Least Privilege**: IAM role has minimal required permissions
- **OIDC**: Short-lived tokens instead of long-lived AWS keys
- **Resource Isolation**: Unique resource names prevent conflicts

## Cost Optimization

- **DynamoDB**: Pay-per-request billing mode
- **S3**: Standard storage class (can be optimized with lifecycle policies)
- **No Always-On Resources**: All resources scale to zero when not in use

## Cleanup

To destroy all resources:

```bash
terraform destroy -var="project_prefix=cdp-dev"
```

**Warning**: This will permanently delete all data in S3 buckets and DynamoDB tables.

## Switching to Remote State

For production use, configure S3 backend for Terraform state:

1. Create an S3 bucket for state storage
2. Update `backend.tf`:
   ```hcl
   terraform {
     backend "s3" {
       bucket = "your-terraform-state-bucket"
       key    = "potato-weight-nutrition/terraform.tfstate"
       region = "us-east-1"
     }
   }
   ```
3. Run `terraform init -migrate-state`

## Troubleshooting

### Common Issues

1. **Bucket name conflicts**: S3 bucket names are globally unique. The configuration uses random suffixes to avoid conflicts.

2. **IAM permissions**: Ensure your AWS credentials have permissions to create IAM roles and policies.

3. **OIDC thumbprint**: The GitHub OIDC thumbprint is pre-configured but may need updates if GitHub changes their certificates.

4. **Region-specific resources**: Some AWS services have region-specific behavior. Test in your target region.

### Validation

After deployment, validate the configuration:

```bash
# Test S3 access
aws s3 ls s3://$(terraform output -raw data_lake_bucket_name)

# Test DynamoDB access
aws dynamodb describe-table --table-name $(terraform output -raw mlflow_backend_table_name)

# Test IAM role
aws sts get-caller-identity
```