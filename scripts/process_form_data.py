import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "data" / "form_feedback.csv"
dst = ROOT / "data" / "form_feedback_cleaned.csv"

# Load Google Form or website data
df = pd.read_csv(src)

# 🧹 Normalize column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# ✅ Expected columns: student_id, name, department, course_name, faculty_name, feedback_text
# If your feedback question is labeled differently, rename it:
for col in df.columns:
    if "feedback" in col:
        df = df.rename(columns={col: "feedback_text"})

# Keep only relevant columns
columns_to_keep = [
    "student_id", "name", "department", "course_name", "faculty_name",
    "teaching_quality", "course_content", "communication_skills",
    "infrastructure/resources", "engagement", "feedback_text"
]

df = df[[c for c in columns_to_keep if c in df.columns]]

# Clean text
df["feedback_text"] = df["feedback_text"].astype(str).str.strip()

# Save cleaned dataset
df.to_csv(dst, index=False)
print(f"✅ Cleaned data saved -> {dst}")
print(f"🧾 Total records: {len(df)}")
print("📊 Columns:", list(df.columns))
