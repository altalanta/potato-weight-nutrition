output "data_lake_bucket_name" {
  description = "Name of the S3 bucket for data lake storage"
  value       = aws_s3_bucket.data_lake.bucket
}

output "data_lake_bucket_arn" {
  description = "ARN of the S3 bucket for data lake storage"
  value       = aws_s3_bucket.data_lake.arn
}

output "mlflow_artifacts_bucket_name" {
  description = "Name of the S3 bucket for MLflow artifacts"
  value       = aws_s3_bucket.mlflow_artifacts.bucket
}

output "mlflow_artifacts_bucket_arn" {
  description = "ARN of the S3 bucket for MLflow artifacts"
  value       = aws_s3_bucket.mlflow_artifacts.arn
}

output "mlflow_backend_table_name" {
  description = "Name of the DynamoDB table for MLflow backend store"
  value       = aws_dynamodb_table.mlflow_backend.name
}

output "mlflow_backend_table_arn" {
  description = "ARN of the DynamoDB table for MLflow backend store"
  value       = aws_dynamodb_table.mlflow_backend.arn
}

output "github_oidc_role_arn" {
  description = "ARN of the IAM role for GitHub Actions OIDC"
  value       = aws_iam_role.github_oidc.arn
}

output "github_oidc_provider_arn" {
  description = "ARN of the GitHub OIDC provider"
  value       = aws_iam_openid_connect_provider.github.arn
}

# Environment variables for application configuration
output "env_vars" {
  description = "Environment variables for pipeline configuration"
  value = {
    AWS_REGION                    = var.aws_region
    DATA_LAKE_BUCKET             = aws_s3_bucket.data_lake.bucket
    MLFLOW_S3_ENDPOINT_URL       = "https://s3.${var.aws_region}.amazonaws.com"
    MLFLOW_ARTIFACTS_DESTINATION = "s3://${aws_s3_bucket.mlflow_artifacts.bucket}"
    MLFLOW_BACKEND_STORE_URI     = "dynamodb://${aws_dynamodb_table.mlflow_backend.name}"
    PROJECT_PREFIX               = var.project_prefix
    ENVIRONMENT                  = var.environment
  }
  sensitive = false
}