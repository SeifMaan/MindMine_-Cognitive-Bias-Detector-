"""
train.py
--------
Fine-tunes DistilBERT on biases_dataset.csv for cognitive bias classification.
Optimized for RTX 3060 6GB VRAM.

Output:
    ./bias_model/         ← saved HuggingFace model + tokenizer
    ./bias_model/label_map.json  ← label string ↔ int mapping

Usage:
    pip install transformers datasets scikit-learn torch accelerate
    python train.py
"""

import json
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

os.environ["TRANSFORMERS_NO_TF"] = "1"

from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
DATA_FILE = "biases_dataset.csv"
MODEL_DIR = "./bias_model"
BASE_MODEL = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
SEED = 42

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Using device: {device}")
if device == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── Load & Clean Data ────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return text


df = pd.read_csv(DATA_FILE)
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].apply(clean_text)

# remove duplicates (important for LLM-generated data)
df = df.drop_duplicates(subset=["text"])

# ── Encode Labels ────────────────────────────────────────────────────────────
labels = sorted(df["label"].unique().tolist())
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
df["label_id"] = df["label"].map(label2id)

print(f"\n📊 Dataset: {len(df)} rows, {len(labels)} classes")
print(f"   Classes: {labels}\n")

# Save label map
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
with open(f"{MODEL_DIR}/label_map.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

# ── Class Weights (for imbalance) ────────────────────────────────────────────
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(df["label_id"]), y=df["label_id"]
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# ── Train / Validation Split ─────────────────────────────────────────────────
train_df, val_df = train_test_split(
    df, test_size=0.15, random_state=SEED, stratify=df["label_id"]
)
print(f"Train: {len(train_df)} | Val: {len(val_df)}")

# ── Tokenize ─────────────────────────────────────────────────────────────────
tokenizer = DistilBertTokenizerFast.from_pretrained(BASE_MODEL)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding=True,  # dynamic padding (better)
        max_length=MAX_LEN,
    )


def df_to_dataset(df: pd.DataFrame) -> Dataset:
    ds = Dataset.from_dict(
        {
            "text": df["text"].tolist(),
            "label": df["label_id"].tolist(),
        }
    )
    return ds.map(tokenize, batched=True, remove_columns=["text"])


train_ds = df_to_dataset(train_df)
val_ds = df_to_dataset(val_df)

# ── Model ────────────────────────────────────────────────────────────────────
model = DistilBertForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)
model.to(device)


# ── Custom Loss (weighted) ───────────────────────────────────────────────────
def compute_loss(model, inputs, return_outputs=False, **kwargs):
    labels = inputs.get("labels")
    outputs = model(**inputs)
    logits = outputs.get("logits")

    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
    loss = loss_fct(logits, labels)

    return (loss, outputs) if return_outputs else loss


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    return {
        "accuracy": acc,
        "f1": f1,
    }


# ── Training Args ─────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    logging_steps=20,
    fp16=(device == "cuda"),
    dataloader_num_workers=0,
    seed=SEED,
    report_to="none",
    # improvements
    lr_scheduler_type="linear",
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
)

# ── Trainer ──────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)


# ── Train ─────────────────────────────────────────────────────────────────────
print("\n🚀 Starting training...\n")
trainer.train()

# ── Final Evaluation ─────────────────────────────────────────────────────────
print("\n📈 Final evaluation on validation set:")
results = trainer.evaluate()
print(f"   Validation Accuracy: {results['eval_accuracy']*100:.2f}%")

# Detailed per-class report
raw_preds = trainer.predict(val_ds)
y_pred = np.argmax(raw_preds.predictions, axis=-1)
y_true = val_df["label_id"].tolist()

print("\nPer-class report:")
print(classification_report(y_true, y_pred, target_names=labels))

# ── Save ──────────────────────────────────────────────────────────────────────
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print(f"\n✅ Model saved to '{MODEL_DIR}/'")
print(f"   Label map saved to '{MODEL_DIR}/label_map.json'")
print(f"\n📌 CV bullet: Fine-tuned DistilBERT classifier on {len(train_df)} examples")
print(
    f"   achieving {results['eval_accuracy']*100:.1f}% validation accuracy across {len(labels)} cognitive bias classes."
)
