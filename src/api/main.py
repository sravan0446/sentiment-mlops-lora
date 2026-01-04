import mlflow
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

app = FastAPI(title="Sentiment Analysis LoRA API")

# --- Global Variables for Model ---
model = None
tokenizer = None
labels = ["NEGATIVE", "POSITIVE"]


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    model_version: str


def load_latest_model():
    """
    Auto-detects the latest run from MLflow and loads the LoRA adapter.
    """
    global model, tokenizer

    print("🔍 Searching for latest model in MLflow...")

    # 1. Get the latest run from your experiment
    try:
        current_experiment = mlflow.get_experiment_by_name("sentiment-analysis-lora")
        if current_experiment is None:
            raise Exception("Experiment 'sentiment-analysis-lora' not found.")

        runs = mlflow.search_runs(
            experiment_ids=[current_experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )

        if runs.empty:
            raise Exception("No runs found!")

        latest_run_id = runs.iloc[0]["run_id"]
        artifact_uri = runs.iloc[0]["artifact_uri"]
        print(f"✅ Found latest Run ID: {latest_run_id}")

    except Exception as e:
        print(f"❌ MLflow Error: {e}")
        print("⚠️  Falling back to manual path (update this if needed)")
        # If MLflow search fails, you can hardcode a path for debugging
        return

    # 2. Construct the local path to the artifacts
    # MLflow URI format: file:///C:/Users/.../mlruns/...
    # We need to convert it to a standard path if local
    if artifact_uri.startswith("file:///"):
        artifact_path = artifact_uri.replace("file:///", "")
    else:
        artifact_path = artifact_uri

    adapter_path = f"{artifact_path}/lora_adapter"
    print(f"📂 Loading adapter from: {adapter_path}")

    # 3. Load Base Model (DistilBERT)
    base_model_id = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        num_labels=2
    )

    # 4. Load & Merge LoRA Adapter
    # We merge the adapter into the base model for faster inference speed
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # Optional: Merge weights for speed (Good for CPU inference)
    model = model.merge_and_unload()
    model.eval()  # Set to evaluation mode
    print("🚀 Model loaded successfully!")
    return latest_run_id


# Load model on startup
latest_run_id = load_latest_model()


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Preprocess
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Postprocess
    prediction_idx = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][prediction_idx].item()

    return SentimentResponse(
        sentiment=labels[prediction_idx],
        confidence=round(confidence, 4),
        model_version=latest_run_id
    )