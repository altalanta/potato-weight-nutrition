# Data Lake S3 Bucket
resource "aws_s3_bucket" "data_lake" {
  bucket = "${var.project_prefix}-potato-data-lake-${random_id.suffix.hex}"
}

resource "aws_s3_bucket_versioning" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# MLflow Artifacts S3 Bucket
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "${var.project_prefix}-mlflow-artifacts-${random_id.suffix.hex}"
}

resource "aws_s3_bucket_versioning" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Random suffix for unique bucket names
resource "random_id" "suffix" {
  byte_length = 4
}

# DynamoDB table for MLflow backend store
resource "aws_dynamodb_table" "mlflow_backend" {
  name           = "${var.project_prefix}-mlflow-backend"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "experiment_id"
  range_key      = "run_uuid"

  attribute {
    name = "experiment_id"
    type = "S"
  }

  attribute {
    name = "run_uuid"
    type = "S"
  }

  tags = {
    Name = "${var.project_prefix}-mlflow-backend"
  }
}

# GitHub OIDC Provider
resource "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"

  client_id_list = [
    "sts.amazonaws.com",
  ]

  thumbprint_list = [
    "6938fd4d98bab03faadb97b34396831e3780aea1",  # GitHub Actions OIDC thumbprint
    "1c58a3a8518e8759bf075b76b750d4f2df264fcd"   # Backup thumbprint
  ]

  tags = {
    Name = "${var.project_prefix}-github-oidc"
  }
}

# IAM Role for GitHub Actions OIDC
resource "aws_iam_role" "github_oidc" {
  name = "${var.project_prefix}-github-oidc-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = aws_iam_openid_connect_provider.github.arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          }
          StringLike = {
            "token.actions.githubusercontent.com:sub" = "repo:${var.github_repo}:*"
          }
        }
      }
    ]
  })

  tags = {
    Name = "${var.project_prefix}-github-oidc-role"
  }
}

# IAM Policy for GitHub Actions - S3 access
resource "aws_iam_policy" "github_s3_access" {
  name        = "${var.project_prefix}-github-s3-access"
  description = "S3 access policy for GitHub Actions"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.data_lake.arn,
          "${aws_s3_bucket.data_lake.arn}/*",
          aws_s3_bucket.mlflow_artifacts.arn,
          "${aws_s3_bucket.mlflow_artifacts.arn}/*"
        ]
      }
    ]
  })
}

# IAM Policy for GitHub Actions - DynamoDB access
resource "aws_iam_policy" "github_dynamodb_access" {
  name        = "${var.project_prefix}-github-dynamodb-access"
  description = "DynamoDB access policy for GitHub Actions"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem",
          "dynamodb:DeleteItem",
          "dynamodb:Query",
          "dynamodb:Scan"
        ]
        Resource = aws_dynamodb_table.mlflow_backend.arn
      }
    ]
  })
}

# IAM Policy for CloudWatch Logs (optional observability)
resource "aws_iam_policy" "github_cloudwatch_logs" {
  name        = "${var.project_prefix}-github-cloudwatch-logs"
  description = "CloudWatch Logs access policy for GitHub Actions"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:${var.aws_region}:*:log-group:/aws/potato-pipeline/*"
      }
    ]
  })
}

# Attach policies to GitHub OIDC role
resource "aws_iam_role_policy_attachment" "github_s3_access" {
  role       = aws_iam_role.github_oidc.name
  policy_arn = aws_iam_policy.github_s3_access.arn
}

resource "aws_iam_role_policy_attachment" "github_dynamodb_access" {
  role       = aws_iam_role.github_oidc.name
  policy_arn = aws_iam_policy.github_dynamodb_access.arn
}

resource "aws_iam_role_policy_attachment" "github_cloudwatch_logs" {
  role       = aws_iam_role.github_oidc.name
  policy_arn = aws_iam_policy.github_cloudwatch_logs.arn
}