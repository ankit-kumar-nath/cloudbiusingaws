output "bucket_name" {
  description = "S3 bucket name for BI uploads"
  value       = aws_s3_bucket.bi_bucket.bucket
}

output "bucket_arn" {
  description = "S3 bucket ARN"
  value       = aws_s3_bucket.bi_bucket.arn
}

output "role_arn" {
  description = "IAM role ARN for the BI app"
  value       = aws_iam_role.bi_role.arn
}

output "instance_profile" {
  description = "IAM instance profile name (for EC2 deployments)"
  value       = aws_iam_instance_profile.bi_instance_profile.name
}
