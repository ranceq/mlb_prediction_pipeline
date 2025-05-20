#!/usr/bin/env python3
# daily_pipeline.py

"""
Automate daily data ingestion (Statcast & schedule), feature engineering, and next-day predictions.
"""
import argparse
from datetime import date, timedelta
import pandas as pd
import requests
from pybaseball import statcast
from google.cloud import storage, bigquery
from joblib import load


def ingest_statcast(bucket_name: str, prefix: str, start_dt: str, end_dt: str):
    # Fetch Statcast data for the given date range
    df = statcast(start_dt, end_dt)
    local_csv = f"/tmp/statcast_{start_dt}_{end_dt}.csv"
    df.to_csv(local_csv, index=False)
    # Upload to GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{prefix}/statcast_{start_dt}_{end_dt}.csv")
    blob.upload_from_filename(local_csv)
    print(f"Uploaded Statcast {start_dt}->{end_dt} to gs://{bucket_name}/{prefix}/")
    # Append into BigQuery mlb_raw.statcast
    bq = bigquery.Client()
    table_ref = bq.dataset('mlb_raw').table('statcast')
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND
    )
    uri = f"gs://{bucket_name}/{prefix}/statcast_{start_dt}_{end_dt}.csv"
    load_job = bq.load_table_from_uri(uri, table_ref, job_config=job_config)
    load_job.result()
    print("Statcast loaded into BigQuery mlb_raw.statcast")


def ingest_schedule(bucket_name: str):
    """
    Fetch tomorrow's MLB schedule via the official Stats API and load into BigQuery mlb_raw.schedule.
    """
    tomorrow = date.today() + timedelta(days=1)
    target_date = tomorrow.isoformat()
    url = f"https://statsapi.mlb.com/api/v1/schedule?date={target_date}&sportId=1"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json().get('dates', [])
    rows = []
    for date_block in data:
        for game in date_block.get('games', []):
            rows.append({
                'game_id':       int(game['gamePk']),
                'game_datetime': game['gameDate'],
                'home_team':     game['teams']['home']['team']['abbreviation'],
                'away_team':     game['teams']['away']['team']['abbreviation']
            })
    if not rows:
        print(f"No games scheduled on {target_date}")
        return
    df = pd.DataFrame(rows)
    # Load into BigQuery mlb_raw.schedule (overwrite)
    bq = bigquery.Client()
    table_ref = bq.dataset('mlb_raw').table('schedule')
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        schema=[
          bigquery.SchemaField('game_id', 'INT64'),
          bigquery.SchemaField('game_datetime', 'DATETIME'),
          bigquery.SchemaField('home_team', 'STRING'),
          bigquery.SchemaField('away_team', 'STRING'),
        ]
    )
    load_job = bq.load_table_from_dataframe(df, table_ref, job_config=job_config)
    load_job.result()
    print(f"Schedule for {target_date} loaded into mlb_raw.schedule")


def feature_engineer_for_date(target_date: str):
    # Run feature-engineering SQL for a single date
    sql = f"""
    INSERT INTO mlb_features.game_features
    WITH gi AS (
      SELECT *, CASE WHEN wteam = hometeam THEN 1 ELSE 0 END AS label
      FROM mlb_raw.gameinfo
      WHERE PARSE_DATE('%Y%m%d', CAST(date AS STRING)) = DATE('{target_date}')
    ), sc AS (
      SELECT CAST(game_date AS DATE) AS game_date, home_team, away_team,
             AVG(launch_speed) AS avg_launch_speed,
             AVG(launch_angle) AS avg_launch_angle,
             AVG(estimated_woba_using_speedangle) AS avg_xwoba
      FROM mlb_raw.statcast
      WHERE CAST(game_date AS DATE) = DATE('{target_date}')
      GROUP BY game_date, home_team, away_team
    ), pf AS (
      SELECT year, park_id, basic_pf AS park_factor FROM mlb_raw.park_factors
    )
    SELECT gi.*, sc.avg_launch_speed, sc.avg_launch_angle, sc.avg_xwoba, pf.park_factor
    FROM gi
    LEFT JOIN sc ON sc.game_date = gi.game_date AND sc.home_team = gi.home_team AND sc.away_team = gi.away_team
    LEFT JOIN pf ON pf.park_id = gi.site AND pf.year = EXTRACT(YEAR FROM gi.game_date)
    """
    bq = bigquery.Client()
    bq.query(sql).result()
    print(f"Features appended for {target_date}")


def predict_next_day(model_gcs_path: str):
    tomorrow = date.today() + timedelta(days=1)
    dstr = tomorrow.isoformat()
    bq = bigquery.Client()
    query = f"""
    SELECT gf.* EXCEPT(label)
    FROM mlb_raw.schedule s
    JOIN mlb_features.game_features gf
      ON s.game_id = gf.game_id
    WHERE DATE(s.game_datetime) = DATE('{dstr}')
    """
    df = bq.query(query).to_dataframe()
    # Load model
    storage_client = storage.Client()
    bucket_name, prefix = model_gcs_path.replace('gs://','').split('/',1)
    blob = storage_client.bucket(bucket_name).blob(f"{prefix}/model.joblib")
    blob.download_to_filename('/tmp/model.joblib')
    model = load('/tmp/model.joblib')
    # Prepare X and predict
    X = df.drop(columns=['game_id','home_team','away_team','game_date'])
    probs = model.predict_proba(X)[:,1]
    df_out = df[['game_id','home_team','away_team']].copy()
    df_out['win_prob'] = probs
    print(df_out)
    return df_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--stat-prefix', default='data/statcast')
    parser.add_argument('--model-path', required=True,
                        help='gs://bucket/models/mlb_xgb/latest')
    parser.add_argument('--mode', choices=['ingest','schedule','features','predict'], required=True)
    args = parser.parse_args()

    # Compute dates
    today = date.today()
    yesterday = (today - timedelta(days=1)).isoformat()

    if args.mode == 'ingest':
        ingest_statcast(args.bucket, args.stat_prefix, yesterday, yesterday)
    elif args.mode == 'schedule':
        ingest_schedule(args.bucket)
    elif args.mode == 'features':
        feature_engineer_for_date(yesterday)
    elif args.mode == 'predict':
        predict_next_day(args.model_path)

if __name__ == '__main__':
    main()
