import json
import re
import random

# Load intents from KB.json
with open("data/KB.json", "r", encoding="utf-8") as f:
    intent_data = json.load(f)["intents"]

def match_intent(user_input):
    user_input_lower = user_input.lower()
    for intent in intent_data:
        for pattern in intent["patterns"]:
            if re.search(rf"\b{re.escape(pattern.lower())}\b", user_input_lower):
                return intent["tag"], random.choice(intent["responses"])
    return None, None