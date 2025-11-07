# scripts/clean_dataset.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "data" / "Student_feedback.csv"
dst = ROOT / "data" / "labeled_feedback.csv"

if not src.exists():
    raise FileNotFoundError(f"❌ Dataset not found at: {src}")

# Read dataset
df = pd.read_csv(src, encoding="utf-8")

# ✅ For generated dataset, columns are already clean and readable
# Just rename for consistency with rest of pipeline
df = df.rename(columns={
    "feedback_text": "text",
    "sentiment_label": "sentiment",
    "topic": "topic",
    "emotion_tag": "emotion"
})

# Add ID column
df.insert(0, "id", range(1, len(df) + 1))

# Save cleaned version
df.to_csv(dst, index=False, encoding="utf-8")

print(f"✅ Cleaned dataset saved -> {dst}")
print("✅ Columns:", list(df.columns))
print("✅ Sample:")
print(df.head(5))
