from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import random
import joblib
from pathlib import Path

app = FastAPI(title="Student Feedback NLP")

ROOT = Path(__file__).resolve().parents[1]
topic_model = joblib.load(ROOT / "models" / "topic_model.pkl")
topic_vec = joblib.load(ROOT / "models" / "topic_vectorizer.pkl")
sent_model = joblib.load(ROOT / "models" / "sentiment_model.pkl")
sent_vec = joblib.load(ROOT / "models" / "sentiment_vectorizer.pkl")

# Suggestion logic (example only)
def get_suggestions(topic, sentiment):
    base_suggestions = {
        "Instructor": [
            "Encourage more Q&A sessions and student interaction.",
            "Clarify complex topics with examples or visuals.",
            "Offer short concept summaries after each lecture."
        ],
        "Infrastructure": [
            "Ensure reliable Wi-Fi and projector systems.",
            "Upgrade lab equipment and computers.",
            "Improve classroom ventilation and maintenance."
        ],
        "Academic": [
            "Review and update syllabus with modern topics.",
            "Include more practical sessions and case studies.",
            "Balance theory with real-world applications."
        ]
    }

    # Sentiment tone mapping
    tone_messages = {
        "positive": "Great job — keep maintaining this quality!",
        "neutral": "Good overall, but there’s still room for small improvements.",
        "negative": "Please focus on these areas for immediate improvement."
    }

    suggestions = base_suggestions.get(topic, [])
    tone = tone_messages.get(sentiment, "")

    return [tone] + suggestions


@app.post("/analyze")
def analyze(
    text: str = Form(...),
    emotion_hint: str = Form("neutral"),
    sarcasm_flag: int = Form(0)
):
    headers = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}

    # Force model to reprocess every request (fresh prediction)
    topic = topic_model.predict(topic_vec.transform([text]))[0]
    sentiment = sent_model.predict(sent_vec.transform([text]))[0]

    suggestions = get_suggestions(topic, sentiment)

    alert = {
        "urgent": sentiment == "negative" and emotion_hint in ["anger", "fear"],
        "level": "high" if sentiment == "negative" else "low",
        "matches": []
    }

    response_data = {
        "text": text,
        "topic": topic,
        "sentiment": sentiment,
        "suggestions": suggestions,
        "alert": alert
    }
    print("🧠 DEBUG:", text, "->", topic, sentiment)

    return JSONResponse(content=response_data, headers=headers)
