
# End-to-End MLOps Pipeline for Sentiment Analysis 🚀

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![Airflow](https://img.shields.io/badge/Orchestration-Airflow-green)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-orange)

An end-to-end MLOps system that fine-tunes **DistilBERT** using **LoRA (Low-Rank Adaptation)** for real-time sentiment analysis. The pipeline features automated retraining via **Apache Airflow**, experiment tracking with **MLflow**, and production-ready inference using **FastAPI**.

## 🏗 Architecture
1.  **Orchestration:** Apache Airflow watches for new data in `data/raw` and triggers the retraining DAG.
2.  **Training:** Uses Hugging Face PEFT (LoRA) to fine-tune DistilBERT efficiently on consumer hardware (reduced VRAM usage by ~60%).
3.  **Tracking:** MLflow logs metrics (accuracy, loss), parameters, and stores versioned model adapters.
4.  **Inference:** FastAPI automatically detects and loads the latest "best" model adapter from the registry.
5.  **Infrastructure:** Fully Dockerized stack (Airflow, Postgres, MLflow, API).

## 🛠 Tech Stack
*   **Model:** DistilBERT + LoRA (PEFT library)
*   **Orchestration:** Apache Airflow (Dockerized)
*   **Experiment Tracking:** MLflow
*   **API Framework:** FastAPI
*   **Containerization:** Docker & Docker Compose
*   **Database:** PostgreSQL

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/sravan0446/sentiment-mlops-lora.git
cd sentiment-mlops-lora
```

### 2. Start the Infrastructure
Make sure Docker Desktop is running (allocate 8GB+ RAM in `.wslconfig`).
```bash
docker-compose up --build -d
```

### 3. Access Dashboards
*   **Airflow:** [http://localhost:8080](http://localhost:8080) (User: `admin`, Pass: `admin`)
*   **MLflow:** [http://localhost:5000](http://localhost:5000)
*   **FastAPI Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Trigger the Pipeline
The Airflow DAG waits for a file named `new_data.csv`. Run this to simulate new data arrival:

```bash
# PowerShell
echo "text,label" > data/raw/new_data.csv
echo "The interface is amazing!,1" >> data/raw/new_data.csv
```

Watch the Airflow DAG turn green and a new run appear in MLflow!
