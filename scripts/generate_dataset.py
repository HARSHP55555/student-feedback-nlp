# scripts/generate_dataset.py
from pathlib import Path
import pandas as pd
import random

ROOT = Path(__file__).resolve().parents[1]
data_path = ROOT / "data" / "Student_feedback.csv"

departments = ["CS", "ECE", "ME", "CIVIL", "BIO", "MBA", "ENG"]
subjects = {
    "CS": ["Data Structures", "Cybersecurity", "AI", "Database Systems"],
    "ECE": ["Embedded Systems", "Digital Circuits", "VLSI"],
    "ME": ["Fluid Mechanics", "Thermodynamics", "Machine Design"],
    "CIVIL": ["Surveying", "Concrete Tech", "Structural Analysis"],
    "BIO": ["Genetics", "Microbiology"],
    "MBA": ["Finance", "Marketing", "HR Management"],
    "ENG": ["Poetry", "Modern Fiction", "Communication Skills"]
}

topics = ["Academic", "Instructor", "Infrastructure"]
sentiments = ["positive", "neutral", "negative"]
emotions = ["joy", "sadness", "anger", "fear", "neutral"]

academic_feedback = {
    "positive": [
        "The course content is well structured and easy to understand.",
        "Assignments helped me learn effectively.",
        "Class discussions were very engaging."
    ],
    "neutral": [
        "Course load is average, not too heavy.",
        "Lectures are okay, nothing special.",
        "Sometimes hard to keep up with pace."
    ],
    "negative": [
        "Too much syllabus for short time.",
        "Course content feels outdated.",
        "No practical examples provided."
    ]
}

instructor_feedback = {
    "positive": [
        "The teacher explains clearly and encourages questions.",
        "Faculty are approachable and supportive.",
        "Instructor provides valuable feedback on assignments."
    ],
    "neutral": [
        "Teaching pace is fine but could use more interaction.",
        "Sometimes lectures get monotonous.",
        "The instructor sticks to slides mostly."
    ],
    "negative": [
        "Teacher doesn’t clear doubts properly.",
        "No real-life examples used in lectures.",
        "Feedback from instructor is delayed."
    ]
}

infrastructure_feedback = {
    "positive": [
        "Labs are well equipped and maintained.",
        "Campus infrastructure is modern and clean.",
        "Library is quiet and resourceful."
    ],
    "neutral": [
        "Wi-Fi is decent but can be slow sometimes.",
        "Classrooms are fine but need minor repairs.",
        "Labs function most of the time."
    ],
    "negative": [
        "Lab computers don’t work properly.",
        "Air conditioning often fails in classrooms.",
        "Projectors and mics need replacement."
    ]
}

def generate_feedback(student_id):
    dept = random.choice(departments)
    subject = random.choice(subjects[dept])
    topic = random.choice(topics)
    sentiment = random.choice(sentiments)
    emotion = random.choice(emotions)

    if topic == "Academic":
        feedback = random.choice(academic_feedback[sentiment])
    elif topic == "Instructor":
        feedback = random.choice(instructor_feedback[sentiment])
    else:
        feedback = random.choice(infrastructure_feedback[sentiment])

    response_required = 1 if sentiment == "negative" and emotion in ["anger", "fear", "sadness"] else 0

    return {
        "student_id": f"STU{student_id:04d}",
        "department": dept,
        "subject_name": subject,
        "feedback_text": feedback,
        "sentiment_label": sentiment,
        "topic": topic,
        "emotion_tag": emotion,
        "response_required": response_required
    }

# Generate synthetic dataset
data = [generate_feedback(i) for i in range(1, 501)]
df = pd.DataFrame(data)
df.to_csv(data_path, index=False, encoding="utf-8")

print(f"✅ Generated dataset with {len(df)} feedback entries -> {data_path}")
print(df.head(5))
