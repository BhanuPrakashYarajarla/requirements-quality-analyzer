# AI-Assisted Classification & Quality Assessment of Software Requirements

> An ML-powered web application that classifies software requirements, evaluates their quality using a custom scoring index, and provides explainable AI insights — all through a clean Flask interface.

---

## Overview

This tool takes a natural language software requirement (or user story) as input and runs it through a full analysis pipeline:

1. **Classification** — Predicts whether the requirement is *Functional* or *Non-Functional* using a trained SVM model
2. **Explainability** — Uses LIME to highlight which words drove the classification decision
3. **RQI Scoring** — Computes a *Requirements Quality Index* (0–100) across 4 dimensions
4. **INVEST Analysis** — Checks if the requirement follows Agile's INVEST criteria for well-formed user stories

Supports both single-requirement input (textarea) and bulk analysis via `.txt` / `.csv` file upload.

---

## Features

- 🤖 **SVM Classifier** — TF-IDF vectorized text classified by a trained Linear SVC model
- 🔍 **LIME Explainability** — Word-level positive/negative/neutral contribution analysis
- 📊 **Requirements Quality Index (RQI)** — Composite score out of 100 across:
  - Ambiguity Score (25 pts)
  - Length Score (20 pts)
  - User Story Format Score (25 pts)
  - Model Confidence Score (30 pts)
- ✅ **INVEST Checker** — Validates Independent, Valuable, Estimable, Small, Testable criteria
- 📂 **Bulk File Upload** — Analyze multiple requirements from `.txt` or `.csv` files
- 🌐 **Flask Web Interface** — Simple, clean browser-based UI

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Flask |
| ML Model | Scikit-learn (LinearSVC) |
| Vectorizer | TF-IDF (scikit-learn) |
| Explainability | LIME (`lime`) |
| Data | Pandas, NumPy |
| Frontend | HTML, CSS (Jinja2 templates) |

---

## Project Structure

```
requirements-quality-analyzer/
│
├── app.py                    # Flask app — routes and analysis pipeline
├── rqi_calculator.py         # Requirements Quality Index scoring logic
├── user_story_analyzer.py    # INVEST criteria + user story component extractor
├── xai_explainer.py          # LIME-based explainability module
├── test.py                   # Basic sanity tests
│
├── models/
│   ├── svm_final_model.pkl   # Trained LinearSVC classifier
│   └── tfidf_vectorizer.pkl  # Fitted TF-IDF vectorizer
│
├── Pure Ds/
│   ├── Pure_Annotate_Dataset.csv   # Labelled training dataset
│   └── testData.csv                # Test samples
│
├── User stories/             # 26 real-world user story datasets (g02–g28)
│   └── *.txt
│
├── SoftwareEngPro.ipynb      # Jupyter notebook — EDA, training & evaluation
│
├── templates/
│   └── index.html            # Main web UI template
│
├── static/
│   └── style.css             # Styling
│
└── .gitignore
```

---

## RQI Scoring Breakdown

| Dimension | Max Score | How it's measured |
|---|---|---|
| **Ambiguity Score** | 25 | Penalizes vague words like *may*, *efficient*, *adequate* |
| **Length Score** | 20 | Optimal range: 8–25 words |
| **User Story Format** | 25 | Checks for *"As a… I want… so that…"* pattern |
| **Model Confidence** | 30 | Sigmoid of SVM decision function score |
| **RQI Total** | **100** | Sum of all four dimensions |

---

## INVEST Criteria

| Criterion | Check |
|---|---|
| **I**ndependent | No conjunctions (*and*/*or*) that may bundle multiple concerns |
| **V**aluable | Contains a *"so that"* clause to state business value |
| **E**stimable | At least 8 words for sufficient detail |
| **S**mall | At most 30 words to remain manageable |
| **T**estable | Contains keywords like *shall*, *must*, *when*, *if* |

---

## Getting Started

### Prerequisites

```bash
pip install flask scikit-learn lime joblib numpy pandas
```

### Run the App

```bash
git clone https://github.com/BhanuPrakashYarajarla/requirements-quality-analyzer.git
cd requirements-quality-analyzer
python app.py
```

Then open your browser at: **http://127.0.0.1:5000**

### Usage

**Single Requirement:**
- Paste any requirement or user story into the text box and click **Analyze**

**Bulk Analysis:**
- Upload a `.txt` file (one requirement per line) or a `.csv` file (requirements in the first column)

---

## Dataset

Training data is sourced from the **PURE dataset** — a publicly available annotated corpus of software requirements labelled as Functional (FR) or Non-Functional (NFR).

The `User stories/` directory contains 26 real-world Agile user story sets from open-source projects, used for INVEST validation testing.

---

## Example

**Input:**
```
As a registered user, I want to reset my password so that I can regain access to my account.
```

**Output:**
- 🏷️ **Label:** Functional Requirement
- 📊 **RQI Score:** 87.4 / 100
- ✅ **INVEST:** 4/5 (not Testable — no *shall/must/if/when*)
- 🔍 **LIME:** *"password"*, *"reset"*, *"access"* strongly support Functional classification


