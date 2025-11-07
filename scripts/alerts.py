# scripts/alerts.py
import json, re
from pathlib import Path

# Load alert keywords
RULES_PATH = Path(__file__).resolve().parent / "alert_rules.json"
RULES = json.loads(RULES_PATH.read_text(encoding="utf-8"))

def check_alerts(text: str, emotion: str, sarcasm_flag: int = 0):
    text = text.lower()
    matched = [kw for kw in RULES["urgent_keywords"] if re.search(kw, text)]
    urgent = bool(matched) or (emotion in RULES["escalate_if_emotion"] and sarcasm_flag == 1)
    level = (
        "critical" if matched
        else "high" if emotion in RULES["escalate_if_emotion"]
        else "info"
    )
    return {"urgent": urgent, "level": level, "matches": matched}
