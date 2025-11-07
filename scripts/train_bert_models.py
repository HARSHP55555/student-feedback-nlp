# scripts/train_bert_models.py

import pandas as pd
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ------------------------
# 1️⃣ Load your cleaned data
# ------------------------
df = pd.read_csv("data/labeled_feedback.csv")

# For both topic classification & sentiment classification
topic_labels = sorted(df["topic"].unique())
sentiment_labels = sorted(df["sentiment"].unique())

topic_label2id = {label: i for i, label in enumerate(topic_labels)}
sentiment_label2id = {label: i for i, label in enumerate(sentiment_labels)}

# ------------------------
# 2️⃣ Prepare tokenizer
# ------------------------
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=128
    )

# ------------------------
# 3️⃣ Split and convert to Dataset
# ------------------------
def prepare_dataset(label_column, label_map):
    df_local = df.copy()
    df_local["label"] = df_local[label_column].map(label_map)
    train_df, test_df = train_test_split(df_local, test_size=0.2, random_state=42)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]])
    test_ds = Dataset.from_pandas(test_df[["text", "label"]])

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    return train_ds, test_ds

train_topic, test_topic = prepare_dataset("topic", topic_label2id)
train_sent, test_sent = prepare_dataset("sentiment", sentiment_label2id)

# ------------------------
# 4️⃣ Initialize model
# ------------------------
def train_model(name, num_labels, train_ds, test_ds):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels
    )

    args = TrainingArguments(
        output_dir=f"models/{name}_bert",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        logging_dir=f"logs/{name}",
        save_strategy="epoch",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(f"models/{name}_bert")
    print(f"✅ Saved model: models/{name}_bert")

# ------------------------
# 5️⃣ Train both models
# ------------------------
print("🚀 Training Topic Classification Model...")
train_model("topic", len(topic_labels), train_topic, test_topic)

print("🚀 Training Sentiment Classification Model...")
train_model("sentiment", len(sentiment_labels), train_sent, test_sent)
