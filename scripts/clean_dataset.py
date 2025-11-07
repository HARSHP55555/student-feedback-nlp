# scripts/clean_dataset.py
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "data" / "labeled_feedback.csv"
dst = ROOT / "data" / "processed_feedback.csv"

print(f"✅ Loading: {src}")
df = pd.read_csv(src)

# Normalize columns
df = df.rename(columns={
    "studentname": "student_name",
    "studentid": "student_id",
    "department": "department",
    "coursename": "course_name",
    "facultyname": "faculty_name",
    "feedbacktext": "text",
    "categorylabel": "topic",
    "sentimentlabel": "sentiment",
    "safetytext": "safety_text",
    "safetyalertneeded": "safety_alert"
})

# Assign ID column
df.insert(0, "id", range(1, len(df) + 1))

# Save cleaned dataset
df.to_csv(dst, index=False)
print(f"✅ Cleaned and saved -> {dst}")
print(f"✅ Columns: {list(df.columns)}")
print(df.sample(5))
