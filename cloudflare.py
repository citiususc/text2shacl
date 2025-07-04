import boto3
import os
from dotenv import load_dotenv
from botocore.exceptions import ClientError
import base64
import tempfile
import uuid
import json

def start_cloudflare():
    """
    Initializes a Cloudflare R2 connection using environment variables
    loaded from a .env file. It sets up a global S3 client
    with the Cloudflare R2 endpoint, configured for automatic region detection.
    
    Raises:
        ValueError: If the configured bucket does not exist in R2.
    """
    global s3
    global bucket_name
    global account_id

    load_dotenv()

    account_id = os.getenv('ACCOUNT_ID')
    access_key = os.getenv('R2_ACCESS_KEY')
    secret_key = os.getenv('R2_SECRET_KEY')
    bucket_name = os.getenv('R2_BUCKET')

    endpoint_url = f'https://{account_id}.r2.cloudflarestorage.com'

    # Initialize boto3 S3 client for Cloudflare R2
    s3 = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint_url,
        region_name='auto'
    )

    if not _bucket_exists():
        raise ValueError(f"Bucket {bucket_name} does not exist.")


def get_cloudflare_data():
    """
    Returns the currently loaded Cloudflare R2 account and bucket
    information for external use.
    
    Returns:
        tuple: (account_id, bucket_name)
    """
    return account_id, bucket_name


def _bucket_exists():
    """
    Checks whether the configured R2 bucket exists by performing
    a head_bucket operation.

    Returns:
        bool: True if the bucket exists, False if a 404 error is returned.
    """
    try:
        s3.head_bucket(Bucket=bucket_name)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise


def upload_base64_image(base64_image):
    """
    Uploads a base64-encoded PNG image to Cloudflare R2
    and returns its public URL.

    Steps:
    - Decodes the base64 string
    - Temporarily saves it to a file
    - Uploads it to the configured R2 bucket with a random UUID name
    - Returns the public access URL

    Args:
        base64_image (str): Base64-encoded PNG image.

    Returns:
        str: Public URL of the uploaded file, or None if an error occurs.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
            load_dotenv()
            pub_url = os.getenv('PUB_URL')

            # Write decoded image to temp file
            tmp_file.write(base64.b64decode(base64_image))
            tmp_file.flush()

            # Generate a random UUID name for the object
            object_name = f"{uuid.uuid4()}.png"

            # Upload to Cloudflare R2
            s3.upload_file(tmp_file.name, bucket_name, object_name)

        return f"{pub_url}/{object_name}"
    except Exception as e:
        print("Error uploading the file:", e)
        return None
