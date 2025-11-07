# scripts/train_bert_models.py
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch

# ---------- CONFIG ----------
DATA_PATH = "data/labeled_feedback.csv"
MODEL_DIR = "models/bert_topic"
BERT_MODEL = "bert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ---------- LOAD DATA ----------
df = pd.read_csv(DATA_PATH)
df = df[["text", "topic", "sentiment"]].dropna()

# ---------- LABEL ENCODERS ----------
topic_le = LabelEncoder()
sent_le = LabelEncoder()

df["topic_label"] = topic_le.fit_transform(df["topic"])
df["sent_label"] = sent_le.fit_transform(df["sentiment"])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# ---------- TOKENIZER ----------
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

# ---------- Datasets ----------
train_topic_ds = Dataset.from_pandas(df_train[["text", "topic_label"]])
test_topic_ds  = Dataset.from_pandas(df_test[["text", "topic_label"]])
train_topic_ds = train_topic_ds.map(tokenize, batched=True)
test_topic_ds  = test_topic_ds.map(tokenize, batched=True)

train_sent_ds = Dataset.from_pandas(df_train[["text", "sent_label"]])
test_sent_ds  = Dataset.from_pandas(df_test[["text", "sent_label"]])
train_sent_ds = train_sent_ds.map(tokenize, batched=True)
test_sent_ds  = test_sent_ds.map(tokenize, batched=True)

# ---------- MODEL for TOPIC ----------
topic_model = AutoModelForSequenceClassification.from_pretrained(
    BERT_MODEL, num_labels=len(topic_le.classes_)
).to(DEVICE)

args_topic = TrainingArguments(
    output_dir=MODEL_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs_topic",
)

trainer_topic = Trainer(
    model=topic_model,
    args=args_topic,
    train_dataset=train_topic_ds,
    eval_dataset=test_topic_ds,
    tokenizer=tokenizer,
)

print("🚀 Training Topic Model...")
trainer_topic.train()
topic_model.save_pretrained(MODEL_DIR + "_cls")
tokenizer.save_pretrained(MODEL_DIR + "_cls")

# ---------- MODEL for SENTIMENT ----------
sent_model = AutoModelForSequenceClassification.from_pretrained(
    BERT_MODEL, num_labels=len(sent_le.classes_)
).to(DEVICE)

args_sent = TrainingArguments(
    output_dir="models/bert_sentiment",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs_sent",
)

trainer_sent = Trainer(
    model=sent_model,
    args=args_sent,
    train_dataset=train_sent_ds,
    eval_dataset=test_sent_ds,
    tokenizer=tokenizer,
)

print("🚀 Training Sentiment Model...")
trainer_sent.train()
sent_model.save_pretrained("models/bert_sentiment_cls")
tokenizer.save_pretrained("models/bert_sentiment_cls")

print("✅ Both BERT models trained and saved!")
