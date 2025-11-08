# scripts/train_bert_models.py
"""
Train two BERT models:
1️⃣ Topic classification (Academic / Instructor / Infrastructure)
2️⃣ Sentiment classification (Positive / Neutral / Negative)
Optimized for Colab GPU (Tesla T4 / A100)
"""

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.preprocessing import LabelEncoder
import torch
import os

# ==============================
# ✅ 1. LOAD DATA
# ==============================
DATA_PATH = "data/labeled_feedback.csv"
print(f"📂 Loading dataset from: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["feedbacktext", "categorylabel", "sentimentlabel"])
print("✅ Columns:", df.columns.tolist())
print("✅ Sample:\n", df.head(2))

# ==============================
# ✅ 2. ENCODE LABELS
# ==============================
topic_encoder = LabelEncoder()
sentiment_encoder = LabelEncoder()

df["topic_label"] = topic_encoder.fit_transform(df["categorylabel"])
df["sentiment_label"] = sentiment_encoder.fit_transform(df["sentimentlabel"])

# Save encoders for backend decoding later
os.makedirs("models", exist_ok=True)
pd.Series(topic_encoder.classes_).to_csv("models/topic_labels.csv", index=False)
pd.Series(sentiment_encoder.classes_).to_csv("models/sentiment_labels.csv", index=False)

# ==============================
# ✅ 3. TOKENIZER
# ==============================
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
    return tokenizer(
        examples["feedbacktext"],
        truncation=True,
        padding=False,
        max_length=128,
    )

# Create Hugging Face Datasets
topic_ds = Dataset.from_pandas(df[["feedbacktext", "topic_label"]])
sentiment_ds = Dataset.from_pandas(df[["feedbacktext", "sentiment_label"]])

topic_ds = topic_ds.map(tokenize_fn, batched=True)
sentiment_ds = sentiment_ds.map(tokenize_fn, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ==============================
# ✅ 4. GPU CHECK
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️ Training on device: {device}")

# ==============================
# ✅ 5. TRAINING CONFIG
# ==============================
from transformers import TrainingArguments
import torch

def get_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",              # ✅ Updated name
        save_strategy="epoch",              # ✅ Keep consistent with eval
        learning_rate=2e-5,
        per_device_train_batch_size=8,      # ✅ Safer for Colab GPU (T4)
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),     # ✅ Enable mixed precision if GPU
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",                   # Disable WandB etc.
    )


# ==============================
# ✅ 6. TRAIN TOPIC MODEL
# ==============================
print("\n🚀 Training BERT for TOPIC classification...")
topic_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(topic_encoder.classes_)
).to(device)

trainer_topic = Trainer(
    model=topic_model,
    args=get_training_args("models/bert_topic"),
    train_dataset=topic_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer_topic.train()
trainer_topic.save_model("models/bert_topic")
print("✅ Saved topic model -> models/bert_topic")

# ==============================
# ✅ 7. TRAIN SENTIMENT MODEL
# ==============================
print("\n🚀 Training BERT for SENTIMENT classification...")
sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(sentiment_encoder.classes_)
).to(device)

trainer_sentiment = Trainer(
    model=sentiment_model,
    args=get_training_args("models/bert_sentiment"),
    train_dataset=sentiment_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer_sentiment.train()
trainer_sentiment.save_model("models/bert_sentiment")
print("✅ Saved sentiment model -> models/bert_sentiment")

print("\n🎯 Training completed successfully!")
