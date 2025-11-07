from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report
import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "labeled_feedback.csv")

# Balance dataset by upsampling minority classes
max_size = df["topic"].value_counts().max()
balanced_df = pd.concat([
    resample(group, replace=True, n_samples=max_size, random_state=42)
    for _, group in df.groupby("topic")
])

X = balanced_df["text"]
y = balanced_df["topic"]

vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
Xv = vec.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(Xv, y, test_size=0.2, random_state=42)
model = LinearSVC()
model.fit(X_train, y_train)

print("✅ Topic model trained successfully!")
print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, ROOT / "models" / "topic_model.pkl")
joblib.dump(vec, ROOT / "models" / "topic_vectorizer.pkl")
