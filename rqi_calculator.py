# rqi_calculator.py

import re
import numpy as np
import joblib

# Load model & vectorizer
svm_model = joblib.load("models/svm_final_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

AMBIGUOUS_WORDS = [
    "may", "might", "could", "should",
    "fast", "efficient", "adequate",
    "sufficient", "user-friendly", "as needed"
]


def ambiguity_score(text):
    text_lower = text.lower()
    found = [w for w in AMBIGUOUS_WORDS if w in text_lower]
    count = len(found)
    score = max(0, 25 - count * 5)

    if found:
        reason = f"Detected ambiguous terms: {', '.join(found)}"
    else:
        reason = "No ambiguous terms detected — clear and precise language"

    return score, reason


def length_score(text):
    words = text.split()
    length = len(words)

    if 8 <= length <= 25:
        score = 20
        reason = f"{length} words — within the optimal range (8–25)"
    elif 5 <= length < 8 or 25 < length <= 35:
        score = 10
        reason = f"{length} words — slightly outside the ideal range (8–25)"
    else:
        score = 5
        if length < 5:
            reason = f"Only {length} words — too brief to be well-specified"
        else:
            reason = f"{length} words — too lengthy, consider splitting"

    return score, reason


def user_story_score(text):
    pattern = r"as a .* i want .* so that .*"
    if re.search(pattern, text.lower()):
        return 25, "Follows the 'As a… I want… so that…' user story format"
    else:
        return 10, "Does not follow the standard user story format"


def model_confidence_score(text):
    X = vectorizer.transform([text])
    score = svm_model.decision_function(X)[0]

    confidence = 1 / (1 + np.exp(-abs(score)))  # sigmoid
    final = round(confidence * 30, 2)

    pct = round(confidence * 100, 1)
    if confidence >= 0.85:
        reason = f"Model is {pct}% confident — strong, decisive classification"
    elif confidence >= 0.65:
        reason = f"Model is {pct}% confident — moderately certain classification"
    else:
        reason = f"Model is only {pct}% confident — classification is borderline"

    return final, reason


def calculate_rqi(text):
    amb, amb_reason = ambiguity_score(text)
    length, length_reason = length_score(text)
    story, story_reason = user_story_score(text)
    model, model_reason = model_confidence_score(text)

    rqi = amb + length + story + model

    return {
        "scores": {
            "Ambiguity Score": amb,
            "Length Score": length,
            "User Story Score": story,
            "Model Confidence": model,
            "RQI Total": round(rqi, 2),
        },
        "explanations": {
            "Ambiguity Score": amb_reason,
            "Length Score": length_reason,
            "User Story Score": story_reason,
            "Model Confidence": model_reason,
        },
    }
