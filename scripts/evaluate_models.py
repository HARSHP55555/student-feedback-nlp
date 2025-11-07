from pathlib import Path
import pandas as pd, joblib
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT/"data/labeled_feedback.csv")

for name, mfile, vfile, target in [
    ("Topic", "topic_model.pkl", "topic_vectorizer.pkl", "topic"),
    ("Sentiment", "sentiment_model.pkl", "sentiment_vectorizer.pkl", "sentiment")
]:
    model = joblib.load(ROOT/"models"/mfile)
    vec   = joblib.load(ROOT/"models"/vfile)
    X, y = vec.transform(df["text"]), df[target]
    pred = model.predict(X)
    print(f"\n=== {name} ===")
    print(classification_report(y, pred))
