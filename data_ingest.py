import argparse
import os
import pandas as pd
from pybaseball import statcast
from google.cloud import storage
import warnings

# Suppress specific FutureWarnings
warnings.filterwarnings(
    'ignore',
    category=FutureWarning,
    message=".*errors='ignore' is deprecated.*",
    module="pybaseball.datahelpers.postprocessing"
)

def upload_to_gcs(local_path: str, bucket_name: str, prefix: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(os.path.join(prefix, os.path.basename(local_path)))
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{bucket_name}/{prefix}{os.path.basename(local_path)}")
    os.remove(local_path)  # free local space

def fetch_statcast(start: str, end: str, out_dir: str, gcs_bucket: str=None, gcs_prefix: str=None) -> None:
    """
    Download Statcast data in monthly chunks from start to end,
    save one CSV per year under out_dir, and optionally upload to GCS.
    """
    os.makedirs(out_dir, exist_ok=True)
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    current = start_date
    all_frames = []

    while current <= end_date:
        period_end = current + pd.offsets.MonthEnd(0)
        period_end = min(period_end, end_date)
        s = current.strftime('%Y-%m-%d')
        e = period_end.strftime('%Y-%m-%d')
        print(f"Fetching Statcast: {s} → {e}")
        df_chunk = statcast(s, e)
        if df_chunk is not None and not df_chunk.empty:
            all_frames.append(df_chunk)
        current = period_end + pd.Timedelta(days=1)

    if not all_frames:
        print(f"No data for {start} to {end}")
        return

    df = pd.concat(all_frames, ignore_index=True)
    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
    df = df.dropna(subset=['game_date'])
    df['year'] = df['game_date'].dt.year

    saved_files = []
    for year, group in df.groupby('year'):
        filename = f"statcast_{year}.csv"
        local_path = os.path.join(out_dir, filename)
        group.to_csv(local_path, index=False)
        print(f"Saved {local_path} ({len(group)} rows)")
        saved_files.append(local_path)
        if gcs_bucket and gcs_prefix:
            upload_to_gcs(local_path, gcs_bucket, gcs_prefix)

    print("Statcast ingestion complete. Files saved:")
    for f in saved_files:
        print("  ", f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ingest Statcast data monthly and upload to GCS.")
    parser.add_argument('--start', required=True, help="Start date YYYY-MM-DD")
    parser.add_argument('--end', required=True, help="End date YYYY-MM-DD")
    parser.add_argument('--out-dir', required=True, help="Directory to write CSVs")
    parser.add_argument('--gcs-bucket', help="(Optional) GCS bucket to upload")
    parser.add_argument('--gcs-prefix', help="(Optional) GCS prefix for uploaded files")
    args = parser.parse_args()
    fetch_statcast(args.start, args.end, args.out_dir, args.gcs_bucket, args.gcs_prefix)