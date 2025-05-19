#!/usr/bin/env python3
# cross_validate.py

import argparse
import json
from datetime import datetime
import pandas as pd
from google.cloud import bigquery, storage
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from joblib import dump
import warnings

warnings.filterwarnings("ignore", category=UserWarning)



def load_train(bq_client, season):
    sql = f"""
    SELECT *
    FROM mlb_features.game_features
    WHERE EXTRACT(YEAR FROM game_date) < {season}
    """
    return bq_client.query(sql).to_dataframe()


def load_test(bq_client, season):
    sql = f"""
    SELECT
      * EXCEPT(label),
      label
    FROM mlb_features.game_features
    WHERE EXTRACT(YEAR FROM game_date) = {season}
    ORDER BY game_date
    """
    return bq_client.query(sql).to_dataframe()


def save_model(local_path, gcs_dir, season):
    storage_client = storage.Client()
    bucket_name, prefix = gcs_dir.replace("gs://", "").split("/", 1)
    dst = f"{prefix}/{season}/model.joblib"
    bucket = storage_client.bucket(bucket_name)
    bucket.blob(dst).upload_from_filename(local_path)
    print(f"  • Saved model to gs://{bucket_name}/{dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=datetime.today().year)
    parser.add_argument(
        "--model-dir", required=True,
        help="GCS folder for models, e.g. gs://algo-bot-v3/models/mlb_xgb"
    )
    parser.add_argument("--grid", action="store_true", help="Enable grid search per season")
    args = parser.parse_args()

    bq = bigquery.Client()
    results = {}

    for season in range(args.start_year, args.end_year + 1):
        print(f"\n=== Season {season} ===")
        # Load data
        df_train = load_train(bq, season)
        df_test  = load_test(bq, season)

        # Separate features and label
        y_train = df_train["label"]
        X_train = df_train.drop(columns=["game_id","home_team","away_team","game_date","label"])
        y_test  = df_test["label"]
        X_test  = df_test.drop(columns=["game_id","home_team","away_team","game_date","label"])

        # 1) Encode categorical columns
        cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            X_train[col] = X_train[col].astype("category").cat.codes
            X_test[col]  = X_test[col].astype("category").cat.codes
        print(f"  • Encoded: {cat_cols}")

        # 2) Cast all remaining columns to numeric
        num_cols = [c for c in X_train.columns if c not in cat_cols]
        for col in num_cols:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
            X_test[col]  = pd.to_numeric(X_test[col], errors="coerce")
        print(f"  • Numeric cast: {num_cols}")

        # 3) Hyperparameter tuning or default training
        if args.grid:
            print("  • Running grid search...")
            grid = GridSearchCV(
                XGBClassifier(eval_metric="logloss"),
                {"max_depth":[3,5], "n_estimators":[50,100], "learning_rate":[0.01,0.1]},
                cv=3, n_jobs=-1
            )
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            print(f"  • Best params: {grid.best_params_}")
        else:
            model = XGBClassifier(eval_metric="logloss")
            model.fit(X_train, y_train)

        # 4) Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"  • Season {season} accuracy = {acc:.4f}")
        results[season] = float(acc)

        # 5) Save model
        local_path = "/tmp/model.joblib"
        dump(model, local_path)
        save_model(local_path, args.model_dir, season)

    # 6) Upload backtest results
    bucket_name, prefix = args.model_dir.replace("gs://","").split("/",1)
    storage.Client().bucket(bucket_name) \
        .blob(f"{prefix}/backtest_results.json") \
        .upload_from_string(json.dumps(results))
    print("\nBacktest complete. Results by season:")
    print(json.dumps(results, indent=2))
