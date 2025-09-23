# Local backend for simplicity - can be switched to S3 later
# To use S3 backend, uncomment and configure:
# terraform {
#   backend "s3" {
#     bucket = "your-terraform-state-bucket"
#     key    = "potato-weight-nutrition/terraform.tfstate"
#     region = "us-east-1"
#   }
# }

# For now using local backend - state stored in terraform.tfstate