# 🧠 Student Feedback NLP
AI system to interpret student feedback, detect sentiment, suggest improvements, and show a modern dashboard.

## Features
- Topic & sentiment classification (FastAPI)
- Suggestions + Alerts
- Streamlit dashboard (Dept/Teacher/Suggestions views)
- CSV pipeline from Google Forms

## Run
```bash
# API
uvicorn backend.app:app --reload

# Dashboard
cd dashboard
streamlit run app.py
