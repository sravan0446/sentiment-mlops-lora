from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime, timedelta
import sys
import os

# Add the source folder to path so we can import train.py
sys.path.append("/opt/airflow")
from src.model.train import train_model

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'sentiment_retraining_pipeline',
    default_args=default_args,
    description='Retrain model when new data arrives',
    schedule_interval='@daily',
    catchup=False,
) as dag:

    # Step 1: Check if 'new_data.csv' exists
    wait_for_data = FileSensor(
        task_id='wait_for_new_data',
        filepath='/opt/airflow/data/raw/new_data.csv',
        poke_interval=10, # Check every 10 seconds
        timeout=600,      # Stop checking after 10 mins
        mode='poke'
    )

    # Step 2: Run the LoRA Training Function
    retrain_model = PythonOperator(
        task_id='retrain_lora_model',
        python_callable=train_model,
        # We can pass arguments if needed
        op_kwargs={'epochs': 1, 'batch_size': 4}
    )

    wait_for_data >> retrain_model