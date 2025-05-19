import argparse
import pandas as pd
from google.cloud import bigquery, storage
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
import warnings

warnings.filterwarnings("ignore", category=UserWarning)



def load_features(query: str) -> pd.DataFrame:
    """
    Run a BigQuery SQL query and return the resulting DataFrame.
    """
    client = bigquery.Client()
    df = client.query(query).to_dataframe()
    return df


def save_model_to_gcs(local_path: str, gcs_path: str):
    """
    Upload a local file to GCS at the specified gcs_path (gs://bucket/dir/file).
    """
    client = storage.Client()
    uri = gcs_path.replace("gs://", "").split("/", 1)
    bucket_name, blob_path = uri[0], uri[1]
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    print(f"Model uploaded to {gcs_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train MLB win-prediction model from BigQuery features."
    )
    parser.add_argument(
        '--query', required=True,
        help='BigQuery SQL query to load features and labels'
    )
    parser.add_argument(
        '--model-dir', required=True,
        help='GCS directory to save the trained model (e.g. gs://bucket/models/mlb_xgb)'
    )
    parser.add_argument(
        '--test-size', type=float, default=0.2,
        help='Fraction of data to hold out for validation'
    )
    parser.add_argument(
        '--random-state', type=int, default=42,
        help='Random seed for train-test split'
    )
    args = parser.parse_args()

    print("Loading features from BigQuery...")
    df = load_features(args.query)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # Expect a 'label' column indicating home win = 1, loss = 0
    if 'label' in df.columns:
        y = df['label'].astype(int)
        X = df.drop(columns=['game_id', 'home_team', 'away_team', 'game_date', 'label'])
    else:
        raise KeyError("Expected a 'label' column in feature DataFrame.")

    # Split into train/validation
    print("Splitting data into train/validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state
    )
    print(f"Training on {X_train.shape[0]} rows, validating on {X_val.shape[0]} rows")

    # Encode categorical/object features to numeric codes
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        # Use categorical codes; unseen categories in validation become -1
        X_train[col] = X_train[col].astype('category').cat.codes
        X_val[col] = X_val[col].astype('category').cat.codes
    print(f"Encoded categorical columns: {cat_cols}")

    # Train the XGBoost model
    print("Training XGBoost classifier...")
    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)

    # Validate
    preds_val = model.predict(X_val)
    val_acc = accuracy_score(y_val, preds_val)
    print(f"Validation accuracy: {val_acc:.4f}")

    # Save the model locally
    local_model_path = '/tmp/model.joblib'
    dump(model, local_model_path)
    print(f"Model saved locally to {local_model_path}")

    # Upload the model to GCS
    gcs_model_path = f"{args.model_dir}/model.joblib"
    save_model_to_gcs(local_model_path, gcs_model_path)


if __name__ == '__main__':
    main()
