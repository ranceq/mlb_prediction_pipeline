import json
from flask import Request, jsonify
from daily_pipeline import ingest_statcast, ingest_schedule, feature_engineer_for_date, predict_next_day
from datetime import date, timedelta

def main(request: Request):
    """
    Expects a JSON body:
    { "mode":"ingest"|"schedule"|"features"|"predict",
      "bucket": "...", 
      "stat_prefix": "...", 
      "model_path": "gs://..." }
    """
    data = request.get_json(silent=True) or {}
    mode = data.get("mode")
    bucket = data.get("bucket")
    stat_prefix = data.get("stat_prefix", "data/statcast")
    model_path = data.get("model_path")

    today = date.today()
    yesterday = (today - timedelta(days=1)).isoformat()

    try:
        if mode == "ingest":
            ingest_statcast(bucket, stat_prefix, yesterday, yesterday)
        elif mode == "schedule":
            ingest_schedule(bucket)
        elif mode == "features":
            feature_engineer_for_date(yesterday)
        elif mode == "predict":
            df = predict_next_day(model_path)
            return jsonify(df.to_dict(orient="records"))
        else:
            return ("Unknown mode", 400)
    except Exception as e:
        return (str(e), 500)

    return ("OK", 200)
