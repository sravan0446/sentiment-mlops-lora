import argparse
import os
import mlflow
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType
)
import evaluate

# # 1. Setup MLflow Experiment
# tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
# mlflow.set_tracking_uri(tracking_uri)
#
# print(f"📡 Connecting to MLflow at: {tracking_uri}")
# mlflow.set_experiment("sentiment-analysis-lora")


def compute_metrics(eval_pred):
    load_accuracy = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}


def train_model(epochs=1, batch_size=16, learning_rate=2e-4):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"📡 Connecting to MLflow at: {tracking_uri}")
    mlflow.set_experiment("sentiment-analysis-lora")
    print("Loading Data...")
    # Using IMDB dataset as a proxy for customer reviews
    dataset = load_dataset("imdb")

    # Small subset for quick debugging/iteration (Remove .select() for full training)
    train_ds = dataset["train"].shuffle(seed=42).select(range(1000))
    test_ds = dataset["test"].shuffle(seed=42).select(range(200))

    model_id = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=False)

    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_test = test_ds.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 2. Load Base Model
    print("Loading Base Model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2, id2label={0: "NEGATIVE", 1: "POSITIVE"}, label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )

    # 3. Apply LoRA (The "Secret Sauce")
    print("Applying LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,  # Rank
        lora_alpha=32,  # Alpha
        lora_dropout=0.1,
        # Target specific modules for DistilBERT
        target_modules=["q_lin", "v_lin", "k_lin", "out_lin"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",  # <--- CHANGED THIS (was evaluation_strategy)
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="mlflow",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 5. Execute Training
    print("Starting Training...")
    with mlflow.start_run() as run:
        # Log LoRA params specifically
        mlflow.log_params(peft_config.to_dict())

        trainer.train()

        # 6. Save Adapter to MLflow
        # We save the PEFT model locally first
        adapter_path = "model_adapter"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        # Log the artifacts
        mlflow.log_artifacts(adapter_path, artifact_path="lora_adapter")

        print(f"Run Complete. Run ID: {run.info.run_id}")


if __name__ == "__main__":
    train_model()