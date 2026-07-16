# app.py

import csv
import io
from flask import Flask, render_template, request
import joblib

from xai_explainer import explain_requirement, format_explanation_to_text
from rqi_calculator import calculate_rqi
from user_story_analyzer import analyze_user_story

app = Flask(__name__)

# Load model & vectorizer
svm_model = joblib.load("models/svm_final_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


def _analyze_single(text):
    """Run the full analysis pipeline on a single requirement string."""
    X = vectorizer.transform([text])
    prediction = svm_model.predict(X)[0]
    label = "Non-Functional Requirement" if prediction == 1 else "Functional Requirement"

    lime_exp = explain_requirement(text).as_list()
    explanation = format_explanation_to_text(lime_exp, label)

    rqi_result = calculate_rqi(text)
    agile_result = analyze_user_story(text)

    return {
        "text": text,
        "label": label,
        "explanation": explanation,
        "rqi": rqi_result,
        "agile": agile_result,
    }


def _parse_uploaded_file(file_storage):
    """Extract a list of requirement strings from an uploaded .txt or .csv file."""
    filename = file_storage.filename.lower()
    raw = file_storage.read().decode("utf-8", errors="replace")

    requirements = []

    if filename.endswith(".csv"):
        reader = csv.reader(io.StringIO(raw))
        for row in reader:
            if row and row[0].strip():
                requirements.append(row[0].strip())
    else:
        # .txt — one requirement per line
        for line in raw.splitlines():
            line = line.strip()
            if line:
                requirements.append(line)

    return requirements


@app.route("/", methods=["GET", "POST"])
def index():
    results = None

    if request.method == "POST":
        requirements_to_analyze = []

        # Check for uploaded file first
        uploaded = request.files.get("req_file")
        if uploaded and uploaded.filename:
            requirements_to_analyze = _parse_uploaded_file(uploaded)

        # Fall back to textarea input
        if not requirements_to_analyze:
            text = request.form.get("requirement", "").strip()
            if text:
                requirements_to_analyze = [text]

        if requirements_to_analyze:
            results = [_analyze_single(req) for req in requirements_to_analyze]

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
