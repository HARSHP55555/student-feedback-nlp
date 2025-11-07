import pandas as pd

# df = pd.read_csv("data/Student_feedback.csv")
# print("\n📊 First 5 rows:\n", df.head())
# print("\n📋 Column names:\n", df.columns)
# print("\nℹ️  Dataset info:\n")
# print(df.info())

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "labeled_feedback.csv")

print("Topic counts:")
print(df['topic'].value_counts())
print("\nSentiment counts:")
print(df['sentiment'].value_counts())