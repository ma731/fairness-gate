"""Plain-English titles and explanations for every check name.

`fairness.RAC1P.max_fpr_gap` is precise and means nothing to most readers. Every check
gets a readable title and one sentence on what it measures, kept in one place so the
site and the PR reviewer explain things the same way.
"""

from __future__ import annotations

# Census column codes. RAC1P and SEX are what the survey calls them; the crossed
# attribute is derived here for auditing and never fed to the model.
ATTRIBUTES = {
    "RAC1P": {
        "who": "racial groups",
        "note": "the census race code, recoded into named groups",
    },
    "SEX": {
        "who": "men and women",
        "note": "the census sex code, the only two values it records",
    },
    "RACExSEX": {
        "who": "race and sex together",
        "note": (
            "race crossed with sex, so each cell is one combination rather than one "
            "attribute averaged over the other"
        ),
    },
}

METRICS = {
    "max_tpr_gap": {
        "title": "Recall gap",
        "what": (
            "Recall is the share of people who genuinely qualify that the model "
            "actually finds. This is the distance between the group it finds most "
            "often and the group it finds least often."
        ),
        "why": (
            "A big number means the model is overlooking one group far more than "
            "another, even though both genuinely qualify."
        ),
    },
    "max_fpr_gap": {
        "title": "False alarm gap",
        "what": (
            "A false positive is somebody flagged who should not have been. This is "
            "the distance between the group it happens to most and the group it "
            "happens to least."
        ),
        "why": (
            "A big number means one group is being wrongly flagged much more often "
            "than another."
        ),
    },
    "max_calibration_gap": {
        "title": "Calibration gap",
        "what": (
            "Calibrated means the probability can be taken at face value: out of a "
            "thousand people scored at 0.7, about seven hundred should qualify. This "
            "is how differently that holds from one group to the next."
        ),
        "why": (
            "A big number means the same score means different things depending on "
            "which group you are in, so a single cut-off is not treating people alike."
        ),
    },
    "max_demographic_parity_difference": {
        "title": "Selection rate gap",
        "what": (
            "How often the model says yes to each group, regardless of whether it was "
            "right. This is the distance between the highest and lowest rate."
        ),
        "why": (
            "A big number means groups are being selected at very different rates. "
            "Some of that gap is in the data itself, which is why this one is judged "
            "alongside the base rate rather than on its own."
        ),
    },
}

# Checks that are not attribute-scoped, keyed by their whole name.
NAMED = {
    "performance.auc": {
        "title": "Ranking quality",
        "what": (
            "AUC measures whether the model puts people in the right order: given one "
            "person who qualifies and one who does not, how often does it score the "
            "first higher. 0.5 is a coin flip, 1.0 is perfect."
        ),
        "why": (
            "The model has to be genuinely useful before its fairness is worth "
            "arguing about, so this is a floor rather than a ceiling."
        ),
    },
    "performance.ece": {
        "title": "Calibration error",
        "what": (
            "How far the model's stated probabilities drift from what actually "
            "happens, averaged across the whole range."
        ),
        "why": (
            "A big number means the probabilities cannot be taken at face value, "
            "which matters for any decision that uses the score rather than the yes "
            "or no."
        ),
    },
    "performance.accuracy_over_majority": {
        "title": "Beating the lazy answer",
        "what": (
            "How much better the model does than always guessing the more common "
            "answer and never looking at anybody."
        ),
        "why": (
            "If this is small the model is not doing much work, and no amount of "
            "fairness analysis would make it worth deploying."
        ),
    },
    "shift.auc": {
        "title": "Ranking quality, unseen states",
        "what": (
            "The same ranking measure, in four states the model was never trained on."
        ),
        "why": (
            "Models are usually scored where they were built. This asks whether it "
            "still works somewhere it has not seen."
        ),
    },
    "shift.tpr_gap_increase_vs_test": {
        "title": "Does the gap get worse elsewhere",
        "what": (
            "How much larger the recall gap becomes in those unseen states, compared "
            "with the test year."
        ),
        "why": (
            "A big number means the disparity is not stable, so measuring it once "
            "where you built the model would have told you only the comfortable half "
            "of the story."
        ),
    },
}


# The regression guard: this run against the recorded baseline.
REGRESSION = {
    "summary": (
        "Nothing got worse since the baseline",
        (
            "Compares this run with a recorded baseline, so a change that stays inside "
            "every limit above but still makes things worse gets caught."
        ),
    ),
    "auc_drop": ("Ranking quality drop since the baseline", ""),
    "ece_increase": ("Calibration error rise since the baseline", ""),
    "baseline": (
        "Regression guard has no baseline",
        (
            "policy.yaml declares the guard, but no baseline has been recorded, so it "
            "can't run."
        ),
    ),
}


def explain(name: str) -> tuple[str, str]:
    """A readable title and an explanation, for any check name in the policy.

    Falls back to the raw name rather than inventing a description, because a check
    nobody wrote an explanation for should look unexplained rather than look fine.
    """
    if name in NAMED:
        entry = NAMED[name]
        return entry["title"], f"{entry['what']} {entry['why']}"

    parts = name.split(".")
    if parts[0] == "regression":
        if len(parts) == 2 and parts[1] in REGRESSION:
            return REGRESSION[parts[1]]
        if len(parts) == 3:
            gap = {"tpr_gap": "Recall gap", "fpr_gap": "False alarm gap"}.get(parts[2])
            who = ATTRIBUTES.get(parts[1], {}).get("who", parts[1])
            if gap:
                return f"{gap} change since the baseline, {who}", ""

    if len(parts) == 3 and parts[0] == "fairness" and parts[2] in METRICS:
        entry = METRICS[parts[2]]
        who = ATTRIBUTES.get(parts[1], {}).get("who", parts[1])
        return f"{entry['title']}, {who}", f"{entry['what']} {entry['why']}"

    return name, ""


def attribute_note(attribute: str) -> str:
    """What a census code actually is, for the places the code itself is shown."""
    return ATTRIBUTES.get(attribute, {}).get("note", "")
