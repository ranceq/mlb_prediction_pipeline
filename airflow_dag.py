from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner':'airflow', 'depends_on_past':False,
    'start_date':datetime(2025,5,13),
    'retries':1, 'retry_delay':timedelta(minutes=5)
}

with DAG('mlb_pipeline', default_args=default_args, schedule_interval='@daily') as dag:
    ingest = BashOperator(
        task_id='ingest',
        bash_command='python data_ingest.py --start 2015-04-05 --end=$(date +"%Y-%m-%d") --out-dir data'
    )
    clean  = BashOperator(task_id='clean', bash_command='python clean_merge.py --retrosheet-dir data/retrosheet --statcast-dir data --out-dir data/merged')
    elo    = BashOperator(task_id='elo',   bash_command='python compute_elo.py --merged-dir data/merged --out-dir data/elo')
    feat   = BashOperator(task_id='features', bash_command='python feature_engineering.py --merged-dir data/merged --elo-dir data/elo --out-dir data/features')
    train  = BashOperator(task_id='train', bash_command='python train_model.py --features-dir data/features --run-dir runs/$(date +"%Y%m%d")')
    ingest >> clean >> elo >> feat >> train