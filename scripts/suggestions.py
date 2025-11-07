# scripts/suggestions.py
import re

# Topic-based improvement templates
TOPIC_SUGGESTIONS = {
    "Academic": [
        "Review and update syllabus with modern topics.",
        "Include more practical sessions or examples.",
        "Balance theory with real-world applications."
    ],
    "Instructor": [
        "Encourage more Q&A sessions and student interaction.",
        "Provide quicker feedback on assignments.",
        "Use more examples during lectures for clarity."
    ],
    "Infrastructure": [
        "Ensure reliable Wi-Fi and projector systems.",
        "Upgrade lab equipment and computers.",
        "Improve classroom ventilation and maintenance."
    ],
}

# Keyword-based hints
KEYWORD_HINTS = {
    r"\bpace|fast|rushed\b": "Slow down lecture pace to improve understanding.",
    r"\bconfus|unclear|hard\b": "Clarify complex topics with examples or visuals.",
    r"\blab|computer|wifi|projector\b": "Check and maintain lab equipment regularly.",
    r"\bmark|grade|evaluation\b": "Provide transparent grading and rubrics.",
    r"\bassignment|homework\b": "Give clear assignment guidelines and deadlines."
}

# Positive reinforcements
POSITIVE_FEEDBACK = {
    "Academic": [
        "Great to see engaging and updated course content!",
        "Students seem satisfied with the learning experience."
    ],
    "Instructor": [
        "Excellent rapport between instructor and students.",
        "Teaching style is clear and effective — keep it up!"
    ],
    "Infrastructure": [
        "Campus facilities appear well-maintained and useful.",
        "Labs and classrooms are functioning smoothly."
    ]
}

def generate_suggestions(topic: str, sentiment: str, text: str):
    suggestions = []
    text_lower = text.lower()

    if sentiment == "negative":
        suggestions.extend(TOPIC_SUGGESTIONS.get(topic, []))
        for pattern, hint in KEYWORD_HINTS.items():
            if re.search(pattern, text_lower):
                suggestions.append(hint)

    elif sentiment == "neutral":
        suggestions.append("Consider gathering more feedback for improvements.")
        suggestions.extend(TOPIC_SUGGESTIONS.get(topic, [])[:2])

    elif sentiment == "positive":
        suggestions.extend(POSITIVE_FEEDBACK.get(topic, []))
        suggestions.append("Maintain this level of satisfaction and consistency!")

    # Remove duplicates and limit
    return list(dict.fromkeys(suggestions))[:5]
