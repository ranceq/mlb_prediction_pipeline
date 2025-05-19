import argparse
import os
from google.cloud import storage

def download_bucket(bucket_name: str, prefix: str, dest_dir: str):
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    for blob in blobs:
        if not blob.name.lower().endswith('.csv'):
            continue
        # Determine local path by stripping prefix
        rel_path = blob.name[len(prefix):].lstrip('/')
        # If file is under a year folder (e.g. '2015/gameinfo.csv')
        parts = rel_path.split('/', 1)
        if parts[0].isdigit() and len(parts[0]) == 4:
            out_path = os.path.join(dest_dir, 'retrosheet', parts[0], parts[1])
        else:
            out_path = os.path.join(dest_dir, 'external', rel_path.replace('/', '_'))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        blob.download_to_filename(out_path)
        print(f"Downloaded {blob.name} to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download and organize raw CSVs from GCS.")
    parser.add_argument('--bucket', required=True, help="GCS bucket name")
    parser.add_argument('--prefix', required=True, help="GCS prefix (e.g., 'retrosheet/')")
    parser.add_argument('--dest', default='data', help="Local destination base dir")
    args = parser.parse_args()
    download_bucket(args.bucket, args.prefix, args.dest)