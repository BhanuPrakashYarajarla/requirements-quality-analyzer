# user_story_analyzer.py

import re


def is_user_story(text):
    pattern = r"as a .* i want .*"
    return bool(re.search(pattern, text.lower()))


def extract_components(text):
    text = text.lower()

    role = re.search(r"as a (.*?),", text)
    goal = re.search(r"i want (.*?) so that", text)
    benefit = re.search(r"so that (.*)", text)

    return {
        "Role": role.group(1) if role else None,
        "Goal": goal.group(1) if goal else None,
        "Benefit": benefit.group(1) if benefit else None
    }


def invest_check(text):
    text_lower = text.lower()
    words = text_lower.split()

    invest = {}
    reasons = {}

    # I — Independent
    has_and = "and" in text_lower
    has_or = "or" in text_lower
    invest["Independent"] = not has_and and not has_or
    if invest["Independent"]:
        reasons["Independent"] = "No conjunctions ('and'/'or') detected — requirement appears self-contained"
    else:
        conj = []
        if has_and:
            conj.append("'and'")
        if has_or:
            conj.append("'or'")
        reasons["Independent"] = f"Detected {', '.join(conj)} — may combine multiple concerns"

    # V — Valuable
    invest["Valuable"] = "so that" in text_lower
    if invest["Valuable"]:
        reasons["Valuable"] = "Contains 'so that' clause — clearly states business value"
    else:
        reasons["Valuable"] = "Missing 'so that' clause — business value is unclear"

    # E — Estimable
    invest["Estimable"] = len(words) >= 8
    reasons["Estimable"] = (
        f"{len(words)} words — {'sufficient' if invest['Estimable'] else 'insufficient'} "
        f"detail for estimation (need ≥ 8)"
    )

    # S — Small
    invest["Small"] = len(words) <= 30
    reasons["Small"] = (
        f"{len(words)} words — {'within' if invest['Small'] else 'exceeds'} "
        f"recommended limit (≤ 30)"
    )

    # T — Testable
    testable_keywords = ["shall", "must", "when", "if"]
    found = [w for w in testable_keywords if w in text_lower]
    invest["Testable"] = len(found) > 0
    if found:
        reasons["Testable"] = f"Contains testable keywords: {', '.join(found)}"
    else:
        reasons["Testable"] = "No testable keywords (shall, must, when, if) found"

    return invest, reasons


def analyze_user_story(text):
    analysis = {}

    analysis["Is User Story"] = is_user_story(text)
    analysis["Components"] = extract_components(text)

    invest, invest_reasons = invest_check(text)
    analysis["INVEST"] = invest
    analysis["INVEST Reasons"] = invest_reasons

    passed = sum(invest.values())
    analysis["INVEST Score"] = f"{passed}/5"

    if analysis["Is User Story"]:
        analysis["Story Reason"] = "Matches the 'As a… I want…' user story pattern"
    else:
        analysis["Story Reason"] = "Does not match the standard user story pattern"

    return analysis
