locals {
  bucket_final_name = length(trimspace(var.bucket_suffix)) > 0 ? "${var.bucket_name}-${var.bucket_suffix}" : var.bucket_name
}


resource "aws_s3_bucket" "bi_bucket" {
  bucket        = local.bucket_final_name
  force_destroy = true
}

# Ownership controls - required before public access block
resource "aws_s3_bucket_ownership_controls" "ownership" {
  bucket = aws_s3_bucket.bi_bucket.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "block_public" {
  bucket = aws_s3_bucket.bi_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "versioning" {
  bucket = aws_s3_bucket.bi_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "sse" {
  bucket = aws_s3_bucket.bi_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "lifecycle" {
  bucket = aws_s3_bucket.bi_bucket.id

  rule {
    id     = "expire-old-versions"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = 365
    }
  }
}



# IAM role for EC2 (or other AWS service) to assume — change Principal service if using ECS/Lambda, etc.
resource "aws_iam_role" "bi_role" {
  name = "bi-ml-app-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Service = "ec2.amazonaws.com"
        },
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Project = "bi-sales-forecasting"
  }
}

# IAM policy that limits actions to the bucket
resource "aws_iam_policy" "bi_policy" {
  name        = "bi-app-s3-policy"
  description = "S3 access policy limited to the BI bucket"
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Sid    = "ListBucket",
        Effect = "Allow",
        Action = [
          "s3:ListBucket"
        ],
        Resource = [
          aws_s3_bucket.bi_bucket.arn
        ]
      },
      {
        Sid    = "ObjectActions",
        Effect = "Allow",
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ],
        Resource = [
          "${aws_s3_bucket.bi_bucket.arn}/*"
        ]
      }
    ]
  })
}

# Attach policy to role
resource "aws_iam_role_policy_attachment" "attach_policy" {
  role       = aws_iam_role.bi_role.name
  policy_arn = aws_iam_policy.bi_policy.arn
}

# (Optional) Instance profile so EC2 can assume the role
resource "aws_iam_instance_profile" "bi_instance_profile" {
  name = "bi-ml-app-instance-profile"
  role = aws_iam_role.bi_role.name
}
