# xai_explainer.py

import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer

# Load model & vectorizer
svm_model = joblib.load("models/svm_final_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

class_names = ["Functional", "Non-Functional"]

def predict_proba_wrapper(texts):
    """
    LIME expects probability outputs.
    LinearSVC gives decision scores, so we convert them.
    """
    X = vectorizer.transform(texts)
    scores = svm_model.decision_function(X)

    # Convert scores to pseudo-probabilities
    probs = 1 / (1 + np.exp(-scores))
    return np.vstack([1 - probs, probs]).T


def explain_requirement(text):
    explainer = LimeTextExplainer(class_names=class_names)

    explanation = explainer.explain_instance(
        text,
        predict_proba_wrapper,
        num_features=10
    )

    return explanation


def format_explanation_to_text(explanation_list, predicted_label):
    """
    Categorizes LIME explanation terms into Positive, Negative, and Neutral contributors.
    Returns a dictionary with categorised lists and an overall summary.
    """
    categorized = {
        "positive": [],
        "negative": [],
        "neutral": [],
        "summary": ""
    }

    # Sort by absolute score
    sorted_exp = sorted(explanation_list, key=lambda x: abs(x[1]), reverse=True)
    
    top_positive_words = []
    top_negative_words = []

    for word, score in sorted_exp:
        abs_score = abs(score)
        
        # Determine the supported class by this word
        # Assuming Score > 0 => Non-Functional, Score < 0 => Functional
        if score > 0:
            supported_class = "Non-Functional Requirement"
        else:
            supported_class = "Functional Requirement"
            
        # Determine Category
        # Positive Contributor: Supports the PREDICTED label
        # Negative Contributor: Supports the OPPOSITE label
        # Neutral: Very low score (e.g., < 0.05)
        
        is_positive_contributor = (supported_class == predicted_label)
        
        if abs_score < 0.05:
            # Neutral
            cat_list = categorized["neutral"]
            cat_list.append(f"'{word}' (Minimal impact)")
        elif is_positive_contributor:
            # Positive
            categorized["positive"].append(f"'{word}' (Strongly supports decision)")
            top_positive_words.append(word)
        else:
            # Negative
            categorized["negative"].append(f"'{word}' (Opposes decision, suggests {supported_class})")
            top_negative_words.append(word)

    # Generate Summary
    if top_positive_words:
        summary = f"The model classifies this as '{predicted_label}' mainly because of terms like {', '.join(top_positive_words[:3])}."
        if top_negative_words:
            summary += f" However, terms like {', '.join(top_negative_words[:3])} provided some conflicting evidence."
    else:
        summary = f"The model classifies this as '{predicted_label}', though no single term strongly dominated the decision."
        
    categorized["summary"] = summary
    
    return categorized
