import boto3
import os
import time
import uuid
from botocore.exceptions import ClientError

S3_BUCKET = os.getenv("S3_BUCKET_NAME", "bi-sales-forecasting-uploads-your-unique-suffix")
S3_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")

# boto3 picks up credentials from environment, shared config, or instance profile
_s3 = boto3.client("s3", region_name=S3_REGION)

def upload_fileobj_to_s3(fileobj, filename, user_id="guest"):
    timestamp = int(time.time())
    key = f"uploads/{user_id}/{timestamp}_{uuid.uuid4().hex}_{filename}"
    try:
        fileobj.seek(0)
        _s3.upload_fileobj(fileobj, S3_BUCKET, key)
        # Pre-signed URL could be created if needed; return key and object URL
        url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"
        return key, url
    except ClientError as e:
        raise RuntimeError(f"S3 upload failed: {e}")
