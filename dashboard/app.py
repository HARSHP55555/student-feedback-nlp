# dashboard/app.py
from pathlib import Path
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Student Feedback Analytics", page_icon="📊", layout="wide")
API_URL = "http://127.0.0.1:8000/analyze"
  # change in .streamlit/secrets.toml if deploying

# ------------------ STYLES ------------------
st.markdown(
    """
    <style>
      .alert-box {background:#2A0B0B; border-left:6px solid #FF4B4B; padding:10px; border-radius:10px; margin:8px 0;}
      .suggest-box {background:#0F1B2D; border-left:6px solid #58A6FF; padding:10px; border-radius:10px; margin:8px 0;}
      .card {background:#0E1117; border:1px solid #1f232b; border-radius:14px; padding:14px; margin-bottom:12px;}
      .muted {opacity: 0.85;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ HELPERS ------------------
@st.cache_data
def load_default():
    # Try form_feedback_cleaned first; else fallback to labeled_feedback
    base = Path(__file__).resolve().parents[1] / "data"
    for name in ["form_feedback_cleaned.csv", "labeled_feedback.csv"]:
        p = base / name
        if p.exists():
            df = pd.read_csv(p)
            return normalize_cols(df)
    return pd.DataFrame()

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # lower + snake
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # unify canonical names
    rename_map = {
        "feedback_text": "text",
        "sentiment_label": "sentiment",
        "faculty": "faculty_name",
        "course": "course_name",
        "subject": "subject_name",
        "course_title": "subject_name",
        "dept": "department",
        "infrastructure/resources": "infrastructure_resources",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k: v})

    # ensure columns exist (create empties if missing)
    for col in ["student_id", "name", "department", "subject_name", "course_name", "faculty_name", "text", "topic", "sentiment", "emotion"]:
        if col not in df.columns:
            df[col] = ""

    # rating columns — pick any present
    possible_ratings = [
        "teaching_quality", "course_content", "communication_skills",
        "infrastructure_resources", "engagement"
    ]
    for pr in possible_ratings:
        if pr not in df.columns:
            df[pr] = pd.NA

    # drop rows with empty text for analytics
    df["text"] = df["text"].astype(str).str.strip()
    return df

def analyze_api(text: str, emotion: str = "neutral", sarcasm: int = 0):
    try:
        r = requests.post(API_URL, data={"text": text, "emotion_hint": emotion, "sarcasm_flag": sarcasm}, headers={"Cache-Control": "no-cache"})
        if r.status_code == 200:
            return r.json()
        return {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def pie_chart(series: pd.Series, title: str):
    counts = series.value_counts(dropna=False)
    if counts.empty:
        st.info("No data to plot.")
        return
    fig, ax = plt.subplots()
    ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%", startangle=90)
    ax.axis('equal')
    ax.set_title(title)
    st.pyplot(fig)

def bar_chart(series: pd.Series, title: str):
    counts = series.value_counts(dropna=False)
    if counts.empty:
        st.info("No data to plot.")
        return
    fig, ax = plt.subplots()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=15)
    st.pyplot(fig)

def faculty_rating_summary(df: pd.DataFrame, group_cols=("department", "faculty_name")):
    rating_cols = [c for c in ["teaching_quality","course_content","communication_skills","infrastructure_resources","engagement"] if c in df.columns]
    if not rating_cols:
        return pd.DataFrame()
    g = df.groupby(list(group_cols), dropna=False)[rating_cols].mean(numeric_only=True).round(2).reset_index()
    return g

# ------------------ SIDEBAR ------------------
st.sidebar.title("🎛️ Controls")
view = st.sidebar.radio("Choose View", ["Department Dashboard", "Teacher Dashboard", "Suggestions Analytics"], index=0)

uploaded = st.sidebar.file_uploader("Upload CSV (form_feedback_cleaned.csv or labeled_feedback.csv)", type=["csv"])
if uploaded:
    try:
        raw = pd.read_csv(uploaded)
        df = normalize_cols(raw)
        st.sidebar.success(f"Loaded {len(df)} rows from upload.")
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")
        df = load_default()
else:
    df = load_default()

if df.empty:
    st.warning("No data found. Upload a CSV from your form or place one in data/form_feedback_cleaned.csv.")
    st.stop()

st.sidebar.write("—")
st.sidebar.caption("Filters")

dept_options = ["All"] + sorted([d for d in df["department"].dropna().unique() if str(d).strip() != ""])
dept_sel = st.sidebar.selectbox("Department", dept_options)

fac_series = df["faculty_name"].fillna("").astype(str).str.strip()
teacher_options = ["All"] + sorted([t for t in fac_series.unique() if t])
teacher_sel = st.sidebar.selectbox("Faculty", teacher_options)

if dept_sel != "All":
    df = df[df["department"] == dept_sel]
if teacher_sel != "All":
    df = df[df["faculty_name"] == teacher_sel]

st.title("📊 Student Feedback Analytics")
st.caption("Modern dashboard with department & teacher insights, live suggestions, and alerts.")

# ------------------ VIEWS ------------------
if view == "Department Dashboard":
    st.subheader("Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Feedback", len(df))
    with c2:
        st.metric("Departments", df["department"].nunique())
    with c3:
        st.metric("Faculty", df["faculty_name"].nunique())

    st.markdown("### Distributions")
    c1, c2 = st.columns(2)
    with c1:
        pie_chart(df["topic"], "Topic Distribution")
    with c2:
        bar_chart(df["sentiment"], "Sentiment Distribution")

    st.markdown("### Faculty Rating Averages")
    summary = faculty_rating_summary(df, ("department","faculty_name"))
    if not summary.empty:
        st.dataframe(summary, use_container_width=True)
    else:
        st.info("No rating columns found. Ensure your CSV has: teaching_quality, course_content, communication_skills, infrastructure_resources, engagement")

    st.markdown("### Recent Feedback")
    for _, row in df.tail(8).iterrows():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"**{row.get('student_id','')}** — `{row.get('department','')}` | `{row.get('subject_name') or row.get('course_name','')}` | `{row.get('faculty_name','')}`")
        st.markdown(f"<span class='muted'>💬</span> {row['text']}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif view == "Teacher Dashboard":
    st.subheader("Teacher-Wise Analytics")

    # ensure a selected teacher context (else show overall)
    if teacher_sel == "All":
        st.info("Tip: choose a specific Faculty from the sidebar to focus this view.")
    by_teacher = faculty_rating_summary(df, ("faculty_name",))
    if not by_teacher.empty:
        st.dataframe(by_teacher, use_container_width=True)

    st.markdown("### Teacher Feedback")
    if not len(df):
        st.info("No feedback for the current filters.")
    else:
        for _, row in df.tail(15).iterrows():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"**{row.get('faculty_name','')}** — `{row.get('department','')}` | `{row.get('subject_name') or row.get('course_name','')}`")
            st.markdown(f"<span class='muted'>💬</span> {row['text']}", unsafe_allow_html=True)
            st.markdown(f"**Topic:** `{row.get('topic','')}` | **Sentiment:** `{row.get('sentiment','')}` | **Emotion:** `{row.get('emotion','')}`")
            st.markdown("</div>", unsafe_allow_html=True)

else:  # Suggestions Analytics
    st.subheader("Live Suggestions & Alerts (via API)")

    limit = st.slider("How many recent rows to analyze?", min_value=5, max_value=100, value=20, step=5)
    sample = df.tail(limit).copy()

    if st.button("Analyze Now"):
        results = []
        for _, r in sample.iterrows():
            res = analyze_api(r["text"], str(r.get("emotion","neutral") or "neutral"))
            topic = res.get("topic", r.get("topic","-"))
            sentiment = res.get("sentiment", r.get("sentiment","-"))
            suggestions = " • ".join(res.get("suggestions", [])) if isinstance(res.get("suggestions", []), list) else ""
            alert = res.get("alert", {})
            urgent = alert.get("urgent", False)
            level = alert.get("level", "")
            results.append({
                "student_id": r.get("student_id",""),
                "department": r.get("department",""),
                "faculty_name": r.get("faculty_name",""),
                "subject_name": r.get("subject_name") or r.get("course_name",""),
                "text": r["text"],
                "topic": topic,
                "sentiment": sentiment,
                "alert_level": level,
                "urgent": urgent,
                "suggestions": suggestions
            })
        out = pd.DataFrame(results)
        st.dataframe(out, use_container_width=True)

        st.markdown("### Highlighted Cards")
        for _, r in out.iterrows():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"**{r['student_id']}** — `{r['department']}` | `{r['subject_name']}` | `{r['faculty_name']}`")
            st.markdown(f"<span class='muted'>💬</span> {r['text']}", unsafe_allow_html=True)
            st.markdown(f"**🗂️ Topic:** `{r['topic']}` | **🧠 Sentiment:** `{r['sentiment']}`")
            if r["urgent"]:
                st.markdown(f"<div class='alert-box'>🚨 Alert: <b>{r['alert_level'].capitalize()}</b></div>", unsafe_allow_html=True)
            if r["suggestions"]:
                st.markdown(f"<div class='suggest-box'>💡 {r['suggestions'].replace(' • ', '<br>💡 ')}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Click **Analyze Now** to fetch suggestions & alerts from the FastAPI backend.")
