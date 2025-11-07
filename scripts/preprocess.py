# scripts/preprocess.py
from pathlib import Path
import pandas as pd
import re
import spacy

# Load spaCy model (make sure you have run: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "data" / "labeled_feedback.csv"
dst = ROOT / "data" / "processed_feedback.csv"

# Verify dataset exists
if not src.exists():
    raise FileNotFoundError(f"❌ Cleaned dataset not found at: {src}\nRun clean_dataset.py first.")

# Load cleaned dataset
df = pd.read_csv(src)
print(f"✅ Loaded cleaned dataset with {len(df)} rows")

# Basic text cleanup
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)        # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)              # remove numbers/punctuation
    text = re.sub(r"\s+", " ", text).strip()           # remove extra spaces
    return text

# Lemmatization & stopword removal
def spacy_process(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

print("🔄 Cleaning and lemmatizing text...")
df["clean_text"] = df["text"].apply(clean_text).apply(spacy_process)

# Drop duplicates / missing rows
df = df.drop_duplicates(subset=["clean_text"]).dropna(subset=["clean_text"])

# Save processed version
df.to_csv(dst, index=False, encoding="utf-8")
print(f"✅ Preprocessed dataset saved -> {dst}")
print("✅ Columns:", list(df.columns))
print(df.head(3))
