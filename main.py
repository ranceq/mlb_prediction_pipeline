#!/usr/bin/env python3
# main.py
"""
Cloud Run entrypoint: daily MLB pipeline with data ingestion, feature engineering, prediction, and weekly retraining.
"""
import os
from datetime import date, timedelta
import pandas as pd
import requests
from google.cloud import storage, bigquery
from joblib import load, dump
from flask import Flask, request, jsonify

app = Flask(__name__)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return ('OK', 200)

# --- Core pipeline functions ---

def ingest_statcast(bucket_name: str, prefix: str, start_dt: str, end_dt: str):
    # defer heavy import
    from pybaseball import statcast
    df = statcast(start_dt, end_dt)
    local_csv = f"/tmp/statcast_{start_dt}_{end_dt}.csv"
    df.to_csv(local_csv, index=False)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{prefix}/statcast_{start_dt}_{end_dt}.csv")
    blob.upload_from_filename(local_csv)
    bq = bigquery.Client()
    table_ref = bq.dataset('mlb_raw').table('statcast')
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND
    )
    uri = f"gs://{bucket_name}/{prefix}/statcast_{start_dt}_{end_dt}.csv"
    bq.load_table_from_uri(uri, table_ref, job_config=job_config).result()
    print(f"Statcast {start_dt}->{end_dt} loaded into BigQuery")


def ingest_schedule(bucket_name: str):
    tomorrow = date.today() + timedelta(days=1)
    target_date = tomorrow.isoformat()
    url = f"https://statsapi.mlb.com/api/v1/schedule?date={target_date}&sportId=1"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json().get('dates', [])
    rows = []
    for block in data:
        for game in block.get('games', []):
            rows.append({
                'game_id': int(game['gamePk']),
                'game_datetime': game['gameDate'],
                'home_team': game['teams']['home']['team']['abbreviation'],
                'away_team': game['teams']['away']['team']['abbreviation']
            })
    if not rows:
        print(f"No games scheduled on {target_date}")
        return
    df = pd.DataFrame(rows)
    bq = bigquery.Client()
    table_ref = bq.dataset('mlb_raw').table('schedule')
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        schema=[
            bigquery.SchemaField('game_id','INT64'),
            bigquery.SchemaField('game_datetime','DATETIME'),
            bigquery.SchemaField('home_team','STRING'),
            bigquery.SchemaField('away_team','STRING')
        ]
    )
    bq.load_table_from_dataframe(df, table_ref, job_config=job_config).result()
    print(f"Schedule for {target_date} loaded into BigQuery")


def feature_engineer_for_date(target_date: str):
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
      SELECT year, park_id, basic_pf AS park_factor
      FROM mlb_raw.park_factors
    )
    SELECT gi.*, sc.avg_launch_speed, sc.avg_launch_angle, sc.avg_xwoba, pf.park_factor
    FROM gi
    LEFT JOIN sc ON sc.game_date=gi.game_date AND sc.home_team=gi.home_team AND sc.away_team=gi.away_team
    LEFT JOIN pf ON pf.park_id=gi.site AND pf.year=EXTRACT(YEAR FROM gi.game_date)
    """
    bigquery.Client().query(sql).result()
    print(f"Features for {target_date} appended to BigQuery")


def predict_next_day(model_gcs_path: str, bucket_name: str):
    # defer heavy import
    from xgboost import XGBClassifier
    tomorrow = date.today() + timedelta(days=1)
    dstr = tomorrow.isoformat()
    bq = bigquery.Client()
    query = f"""
    SELECT gf.* EXCEPT(label)
    FROM mlb_raw.schedule s
    JOIN mlb_features.game_features gf ON s.game_id=gf.game_id
    WHERE DATE(s.game_datetime)=DATE('{dstr}')
    """
    df = bq.query(query).to_dataframe()
    storage_client = storage.Client()
    bucket, prefix = model_gcs_path.replace('gs://','').split('/',1)
    storage_client.bucket(bucket).blob(f"{prefix}/model.joblib").download_to_filename('/tmp/model.joblib')
    model = load('/tmp/model.joblib')
    X = df.drop(columns=['game_id','home_team','away_team','game_date'])
    df_out = df[['game_id','home_team','away_team']].copy()
    df_out['win_prob'] = model.predict_proba(X)[:,1]
    local_csv = f"/tmp/predictions_{dstr}.csv"
    df_out.to_csv(local_csv, index=False)
    storage_client.bucket(bucket_name).blob(f"predictions/{dstr}.csv").upload_from_filename(local_csv)
    print(f"Saved predictions to gs://{bucket_name}/predictions/{dstr}.csv")
    return df_out


def retrain_weekly(model_gcs_path: str, bucket_name: str):
    # defer heavy import
    import optuna
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score
    bq = bigquery.Client()
    df = bq.query('SELECT * FROM mlb_features.game_features').to_dataframe()
    X = df.drop(columns=['game_id','home_team','away_team','game_date','label'])
    y = df['label']
    for col in X.select_dtypes(include=['object']):
        X[col] = X[col].astype('category').cat.codes
    X = X.apply(pd.to_numeric, errors='coerce')
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth',3,10),
            'learning_rate': trial.suggest_float('learning_rate',0.01,0.3,log=True),
            'n_estimators': trial.suggest_int('n_estimators',100,500),
            'subsample': trial.suggest_float('subsample',0.7,1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree',0.7,1.0)
        }
        clf = XGBClassifier(**params, eval_metric='logloss')
        return cross_val_score(clf, X, y, cv=3, scoring='accuracy', n_jobs=-1).mean()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    model = XGBClassifier(**best_params, eval_metric='logloss')
    model.fit(X, y)
    local_model = '/tmp/model.joblib'
    dump(model, local_model)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    prefix = model_gcs_path.replace(f'gs://{bucket_name}/','')
    bucket.blob(f"{prefix}/model.joblib").upload_from_filename(local_model)
    print(f"Retrained model saved to {model_gcs_path}")


@app.route('/', methods=['POST'])
def handler():
    data = request.get_json(silent=True) or {}
    mode = data.get('mode')
    bucket = data.get('bucket')
    prefix = data.get('stat_prefix','data/statcast')
    model_path = data.get('model_path')
    if mode == 'ingest':
        yesterday = (date.today()-timedelta(days=1)).isoformat()
        ingest_statcast(bucket, prefix, yesterday, yesterday)
        return ('OK',200)
    if mode == 'schedule':
        ingest_schedule(bucket)
        return ('OK',200)
    if mode == 'features':
        yesterday = (date.today()-timedelta(days=1)).isoformat()
        feature_engineer_for_date(yesterday)
        return ('OK',200)
    if mode == 'predict':
        df = predict_next_day(model_path, bucket)
        return jsonify(df.to_dict(orient='records'))
    if mode == 'retrain':
        retrain_weekly(model_path, bucket)
        return ('OK',200)
    return ('Invalid mode',400)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
