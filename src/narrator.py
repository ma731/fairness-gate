"""A written summary of the audit, and the verifier that decides whether to publish it.

The rest of this project generates its prose from templates, which cannot be wrong and
also cannot say anything a template author did not think of. A language model writes
better summaries than a template and will, given the chance, state a number that is not
in the data. On a page whose entire argument is that you should check what your model is
doing, publishing unverified generated text would be the joke writing itself.

So the model never publishes. It proposes, and `verify()` decides:

1. **Every number has to trace to the audit.** Not approximately, not plausibly. Each
   numeric token in the draft must match a measured value, its percentage form, or a
   rounding of one. Anything else is an invented statistic, which is the failure mode
   that actually matters here.
2. **Superlatives have to be right.** "The group it overlooks most" names a specific
   group, and there is exactly one correct answer in the data.
3. **Suppressed groups stay suppressed.** Six race-by-sex cells are below the reporting
   floor. A draft that quotes a rate for one of them has published a number the rest of
   the project deliberately withheld.
4. **The mitigation was not adopted.** It was measured and refused. A summary saying it
   was applied would invert the finding.
5. **No claim the model is fair.** It is not, and no summary of this audit gets to say
   otherwise.
6. **No findings about attributes this audit never measured.** Hispanic origin, age,
   disability and the rest are not in here, and a confident sentence about one of them
   is fabrication regardless of whether it sounds reasonable.

A draft with any violation is rejected and regenerated, up to a limit. If nothing
verifies, nothing is published and the page falls back to the template prose it already
had. Failing closed is the only option that is consistent with the rest of the project.

The generation half needs an API key and runs by hand. The verifier needs neither, so it
runs in CI against a suite of drafts with known planted errors (`evals/`), which is what
stops the guardrail quietly rotting.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
NARRATION_PATH = REPO_ROOT / "results" / "narration.json"

MAX_ATTEMPTS = 4

# A number matches a measured value within this much. Rates on 600,000 people are
# stable well inside it, and it is tight enough that 0.312 does not pass as 0.412.
TOLERANCE = 5e-4

# Things this audit does not measure. A confident sentence about any of them is
# fabrication however reasonable it sounds, and these are the ones a model reaches for.
ABSENT_TERMS = [
    "hispanic", "latino", "latinx", "age group", "older workers", "younger workers",
    "disability", "disabled", "veteran", "religion", "immigrant", "citizenship",
    "national origin", "marital discrimination", "education discrimination",
]

# No summary of this audit gets to say the model came out fine.
FORBIDDEN = [
    (r"\b(is|are|was|were)\s+(now\s+)?(fair|unbiased|bias[- ]free|equitable)\b",
     "claims_fairness"),
    (r"\bno (evidence of )?(bias|disparity|discrimination)\b", "claims_fairness"),
    (r"\bfairness (issue|problem)s? (were|was|have been) (fixed|resolved|solved)\b",
     "claims_fairness"),
    (r"\b(bias|disparity|gap) (has been|was|were|is) (eliminated|removed|fixed)\b",
     "claims_fairness"),
    ((r"\b(we |I )?(applied|adopted|shipped|deployed|implemented|enabled) "
      r"(the )?(per-group|group-specific|equal opportunity) (thresholds?|mitigation)\b"),
     "claims_mitigation_shipped"),
    (r"\bmitigation (is|was|has been) (applied|adopted|shipped|deployed|in place)\b",
     "claims_mitigation_shipped"),
]

SUPERLATIVE_WORST = re.compile(
    r"\b(worst|least|lowest|most overlooked|missed most|overlooks most|"
    r"hardest hit|worst[- ]treated)\b", re.IGNORECASE
)
SUPERLATIVE_BEST = re.compile(
    r"\b(best|highest|most often|best[- ]treated|found most)\b", re.IGNORECASE
)

@dataclass
class Violation:
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


@dataclass
class Facts:
    """Everything a draft is allowed to assert, pulled straight out of the audit."""

    numbers: set = field(default_factory=set)
    groups: dict = field(default_factory=dict)     # name -> {tpr, fpr, n, reportable}
    suppressed: set = field(default_factory=set)
    worst_tpr: str = ""
    best_tpr: str = ""
    mitigation_adopted: bool = False
    # Specific quantities a sentence can be checked against, not just membership of.
    accuracy_lo: float = 0.0
    accuracy_hi: float = 0.0
    accuracy_cost: float = 0.0


def _add(into: set, value) -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        into.add(float(value))


def facts(result: dict, tables: dict[str, pd.DataFrame]) -> Facts:
    """The allowed universe: the specific quantities a summary may cite.

    The first version of this walked every number in audit.json, which sounded thorough
    and was useless. The audit contains a 41 point threshold sweep and a pile of
    histogram bins, so the allowed set ran to thousands of values and almost any two or
    three decimal number found something to match. Three planted errors sailed through
    the evals because of it.

    So the set is built by hand from the headline quantities: the gaps, the per-group
    rates, the model scores, the policy limits, the mitigation summary and the counts.
    A summary citing a single point off an internal sweep is not summarising, and the
    verifier is allowed to be strict about that.
    """
    numbers: set = set()

    for split in result.get("splits", {}).values():
        _add(numbers, split.get("n"))
        _add(numbers, split.get("positive_rate"))

    for scores in result.get("scores", {}).values():
        for key in ("auc", "average_precision", "brier", "ece", "accuracy",
                    "threshold", "majority_baseline"):
            _add(numbers, scores.get(key))

    for split_rows in result.get("fairness", {}).values():
        for row in split_rows:
            for key, value in row.items():
                if key != "attribute":
                    _add(numbers, value)

    for check in result.get("checks", []):
        for key in ("measured", "warn_at", "fail_at"):
            _add(numbers, check.get(key))

    for attribute in (result.get("uncertainty") or {}).values():
        for interval in attribute.values():
            if isinstance(interval, dict):
                for key in ("lo", "hi", "level"):
                    _add(numbers, interval.get(key))

    # The mitigation summary, but deliberately not its sweep.
    for arm in (result.get("mitigation") or {}).values():
        for key in ("accuracy_cost", "gap_closed"):
            _add(numbers, arm.get(key))
        for block in ("baseline", "per_group", "best_global"):
            for key, value in (arm.get(block) or {}).items():
                if key != "thresholds":
                    _add(numbers, value)

    arm = result.get("unaware") or {}
    for key in ("tpr_gap", "fpr_gap", "accuracy", "threshold", "n"):
        _add(numbers, arm.get(key))
    for value in (arm.get("comparison") or {}).values():
        _add(numbers, value)

    for key in ("threshold", "train_year", "val_year", "test_year"):
        _add(numbers, result.get("design", {}).get(key))

    # How many checks landed where, which a summary legitimately quotes.
    counts = {"pass": 0, "warn": 0, "fail": 0}
    for check in result.get("checks", []):
        counts[check["status"]] = counts.get(check["status"], 0) + 1
    for value in counts.values():
        _add(numbers, value)
    _add(numbers, len(result.get("checks", [])))

    test = tables["test"]
    for column in ("n", "tpr", "fpr", "accuracy", "base_rate", "selection_rate", "ece"):
        if column in test.columns:
            numbers.update(float(v) for v in test[column].dropna())
    _add(numbers, int((~test[test["attribute"] == "RACExSEX"]["reportable"]).sum()))
    _add(numbers, len(test[test["attribute"] == "RACExSEX"]))

    groups, suppressed = {}, set()
    race = test[test["attribute"] == "RAC1P"]
    for _, row in race.iterrows():
        name = str(row["group"])
        groups[name] = {
            "tpr": float(row["tpr"]), "fpr": float(row["fpr"]),
            "accuracy": float(row["accuracy"]),
            "n": int(row["n"]), "reportable": bool(row["reportable"]),
        }
        if not row["reportable"]:
            suppressed.add(name)
    for _, row in test[test["attribute"] == "RACExSEX"].iterrows():
        if not row["reportable"]:
            suppressed.add(str(row["group"]))

    shown = {k: v for k, v in groups.items() if v["reportable"]}
    worst = min(shown, key=lambda k: shown[k]["tpr"]) if shown else ""
    best = max(shown, key=lambda k: shown[k]["tpr"]) if shown else ""

    # Counts the prose legitimately uses: how many people a group overlooks, and the
    # total across all of them. Derived, but derived from measured things.
    total_overlooked = 0.0
    for name, g in shown.items():
        positives = float(race[race["group"] == name]["n"].iloc[0]) * float(
            race[race["group"] == name]["base_rate"].iloc[0]
        )
        missed = positives * (1 - g["tpr"])
        numbers.add(round(missed))
        numbers.add(round((1 - g["tpr"]) * 100))
        numbers.add(round(g["tpr"] * 100))
        total_overlooked += missed
    numbers.add(round(total_overlooked))

    # Ratios the page itself states, so a summary may repeat them.
    if worst and best:
        lo, hi = 1 - shown[best]["tpr"], 1 - shown[worst]["tpr"]
        if lo:
            numbers.add(round(hi / lo, 1))

    accuracies = [g["accuracy"] for g in shown.values()] if shown else [0.0]

    return Facts(
        numbers={round(n, 6) for n in numbers},
        groups=groups,
        suppressed=suppressed,
        worst_tpr=worst,
        best_tpr=best,
        mitigation_adopted=False,
        accuracy_lo=min(accuracies),
        accuracy_hi=max(accuracies),
        accuracy_cost=float(
            ((result.get("mitigation") or {}).get("RAC1P") or {}).get("accuracy_cost", 0)
        ),
    )


def _tokens(text: str) -> list[tuple[str, float, int, bool]]:
    """Numeric tokens as (literal, value, decimals written, was a percentage).

    How many decimals someone wrote matters. "0.31" claims three significant figures and
    "0.3119" claims five, so they are not the same claim and cannot be checked the same
    way.
    """
    out = []
    for match in re.finditer(r"(\d[\d,]*(?:\.(\d+))?)\s*(%?)", text):
        literal, decimals_part, percent = match.group(1), match.group(2), match.group(3)
        try:
            value = float(literal.replace(",", ""))
        except ValueError:
            continue
        out.append((
            match.group(0).strip(),
            value,
            len(decimals_part) if decimals_part else 0,
            percent == "%",
        ))
    return out


def _supported(value: float, decimals: int, percent: bool, known: set) -> bool:
    """Does this number trace to something measured?

    A number is supported when it is a measured value, or a *rounding* of one to the
    precision it was actually written at. That second part is what makes this usable:
    a summary writing 0.312 for 0.311919 is rounding, not inventing, and a verifier that
    rejects it would be switched off within a day.

    The precision has to travel with the number, though. Comparing every literal to
    every fact with one loose tolerance was the first version, and it let 0.447 through
    by matching some unrelated calibration error four decimal places away. Now 0.312
    is only accepted if some measured value actually rounds to 0.312 at three places.

    Two readings are allowed for a bare number: the value itself, and the percentage
    reading, because prose says "54%" and "finds 54 in another" for a recall of 0.5420.
    The percentage reading is only offered when it makes sense, which is when the token
    carried a percent sign or is at least 1. Offering it for every small decimal is how
    the loose version leaked.
    """
    readings = []
    if percent:
        readings.append((value / 100.0, decimals + 2))
    else:
        readings.append((value, decimals))
        if value >= 1:
            readings.append((value / 100.0, decimals + 2))

    for candidate, places in readings:
        for fact in known:
            if abs(candidate - fact) <= TOLERANCE:
                return True
            if abs(round(fact, places) - candidate) <= 1e-9:
                return True
    return False


def check_numbers(text: str, f: Facts) -> list[Violation]:
    out = []
    for raw, value, decimals, percent in _tokens(text):
        if not _supported(value, decimals, percent, f.numbers):
            out.append(Violation(
                "unsupported_number",
                f"{raw!r} does not match anything in the audit",
            ))
    return out


def check_suppressed(text: str, f: Facts) -> list[Violation]:
    """A rate for a group the project withheld is the project contradicting itself."""
    out = []
    lower = text.lower()
    for name in f.suppressed:
        idx = lower.find(name.lower())
        if idx == -1:
            continue
        window = text[max(0, idx - 160): idx + len(name) + 160]
        if _tokens(window):
            out.append(Violation(
                "suppressed_group_rate",
                f"quotes a figure alongside {name!r}, which is below the reporting floor",
            ))
    return out


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def check_superlatives(text: str, f: Facts) -> list[Violation]:
    """There is exactly one correct answer to "which group does it miss most"."""
    out = []
    shown = {k: v for k, v in f.groups.items() if v["reportable"]}
    for sentence in _sentences(text):
        named = [n for n in shown if n.lower() in sentence.lower()]
        if len(named) != 1:
            continue  # a comparison of two groups is not a superlative claim
        name = named[0]
        if SUPERLATIVE_WORST.search(sentence) and name != f.worst_tpr:
            out.append(Violation(
                "wrong_superlative",
                f"calls {name!r} the worst treated; the audit says {f.worst_tpr!r}",
            ))
        elif SUPERLATIVE_BEST.search(sentence) and name != f.best_tpr:
            out.append(Violation(
                "wrong_superlative",
                f"calls {name!r} the best treated; the audit says {f.best_tpr!r}",
            ))
    return out


def check_forbidden(text: str, f: Facts) -> list[Violation]:
    out = []
    for pattern, code in FORBIDDEN:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if code == "claims_mitigation_shipped" and f.mitigation_adopted:
                continue
            out.append(Violation(code, f"says {match.group(0)!r}"))
    return out


def check_absent_attributes(text: str, f: Facts) -> list[Violation]:
    """A confident sentence about something never measured is fabrication."""
    out = []
    lower = text.lower()
    for term in ABSENT_TERMS:
        if term in lower:
            out.append(Violation(
                "absent_attribute",
                f"discusses {term!r}, which this audit does not measure",
            ))
    return out


def check_substance(text: str, f: Facts) -> list[Violation]:
    """A draft that is merely harmless is not worth publishing either."""
    out = []
    if len(text.split()) < 60:
        out.append(Violation("too_thin", "under sixty words, so it says nothing useful"))
    if not _tokens(text):
        out.append(Violation("no_numbers", "states no measured figure at all"))
    if f.worst_tpr and f.worst_tpr.lower() not in text.lower():
        out.append(Violation(
            "missing_finding",
            f"never names {f.worst_tpr!r}, the group the model overlooks most",
        ))
    return out


RANGE = re.compile(
    r"\b(?:between|from)\s+(\d*\.?\d+)\s+(?:and|to)\s+(\d*\.?\d+)", re.IGNORECASE
)
COST = re.compile(
    r"\b(?:cost|costs|for|at)\s+(\d*\.?\d+)\s+(?:of|in)\s+accuracy", re.IGNORECASE
)


def _decimals(literal: str) -> int:
    return len(literal.split(".")[1]) if "." in literal else 0


def check_claims(text: str, f: Facts) -> list[Violation]:
    """Is the number being used for the thing it says it is?

    Membership in the audit is necessary and not sufficient, and this is where I found
    that out. "Accuracy between 0.79 and 0.84" passed the number check because 0.79 is
    a rounding of a shift-split AUC sitting elsewhere in the file. The number existed.
    The claim was still false: the measured range starts at 0.77.

    That was not a hypothetical. It was the figure sitting in my own README, and this
    check is what caught it. So the claims with a single correct answer get compared
    against that answer rather than against the whole haystack.
    """
    out = []
    for sentence in _sentences(text):
        if "accurac" in sentence.lower():
            match = RANGE.search(sentence)
            if match:
                lo_lit, hi_lit = match.group(1), match.group(2)
                lo, hi = float(lo_lit), float(hi_lit)
                if 0 < lo <= 1 and 0 < hi <= 1:
                    want_lo = round(f.accuracy_lo, _decimals(lo_lit))
                    want_hi = round(f.accuracy_hi, _decimals(hi_lit))
                    if abs(lo - want_lo) > 1e-9 or abs(hi - want_hi) > 1e-9:
                        out.append(Violation(
                            "wrong_claim",
                            f"says accuracy runs {lo} to {hi}; measured is "
                            f"{f.accuracy_lo:.4f} to {f.accuracy_hi:.4f}",
                        ))

        match = COST.search(sentence)
        if match:
            literal = match.group(1)
            stated = float(literal)
            want = round(f.accuracy_cost, _decimals(literal))
            if abs(stated - want) > 1e-9:
                out.append(Violation(
                    "wrong_claim",
                    f"puts the accuracy cost at {stated}; measured is "
                    f"{f.accuracy_cost:.4f}",
                ))
    return out


CHECKS = (
    check_numbers,
    check_claims,
    check_suppressed,
    check_superlatives,
    check_forbidden,
    check_absent_attributes,
    check_substance,
)


def verify(text: str, f: Facts) -> list[Violation]:
    """Every check, in one pass. An empty list is the only thing that publishes."""
    out: list[Violation] = []
    for check in CHECKS:
        out.extend(check(text, f))
    return out


# ---------------------------------------------------------------------------- #
# Generation
# ---------------------------------------------------------------------------- #
SYSTEM = (
    "You summarise a fairness audit for a reader who does not work in machine "
    "learning. You may only state figures that appear in the facts given to you. You "
    "never round beyond three decimals, never estimate, and never mention an attribute "
    "the audit did not measure. If a fact is not in front of you, you leave it out."
)

TEMPLATE = """Write a short summary of this fairness audit: four or five sentences,
plain language, no jargon without explaining it, and no em dashes.

It must say what the disparity is, name the group the model overlooks most often, note
that accuracy is nearly flat across groups anyway, and say that the available mitigation
was measured and deliberately not adopted.

Do not claim the model is fair. Do not claim anything was fixed.

Facts you may use, and nothing else:

{facts}
"""


def prompt(result: dict, tables: dict[str, pd.DataFrame]) -> str:
    """The facts the model is allowed to work from, written out."""
    test = tables["test"]
    race = test[(test["attribute"] == "RAC1P") & test["reportable"]]
    rows = "\n".join(
        f"- {r['group']}: recall {r['tpr']:.4f}, false positive rate {r['fpr']:.4f}, "
        f"accuracy {r['accuracy']:.4f}, {int(r['n']):,} people"
        for _, r in race.sort_values("tpr").iterrows()
    )
    summary = next(s for s in result["fairness"]["test"] if s["attribute"] == "RAC1P")
    mit = (result.get("mitigation") or {}).get("RAC1P") or {}
    arm = (result.get("unaware") or {}).get("comparison") or {}

    body = f"""Population: {result['splits']['test']['n']:,} people, survey year {result['design']['test_year']}.
Recall gap across racial groups: {summary['tpr_gap']:.4f}
False positive gap across racial groups: {summary['fpr_gap']:.4f}
Base rate gap (a real difference in the data): {summary['base_rate_gap']:.4f}
Policy verdict: {result['policy_verdict']}

Per group:
{rows}

Mitigation measured and NOT adopted:
- recall gap would fall from {(mit.get('baseline') or {}).get('tpr_gap', 0):.4f} to {(mit.get('per_group') or {}).get('tpr_gap', 0):.4f}
- false positive gap would RISE from {(mit.get('baseline') or {}).get('fpr_gap', 0):.4f} to {(mit.get('per_group') or {}).get('fpr_gap', 0):.4f}
- accuracy cost {mit.get('accuracy_cost', 0):.4f}
- it requires the person's race at the moment of prediction, which is why it was refused

Deleting race and sex from the features:
- recall gap only falls from {arm.get('tpr_gap_aware', 0):.4f} to {arm.get('tpr_gap_unaware', 0):.4f}
- that is {arm.get('share_remaining', 0):.0%} of the gap surviving"""

    return TEMPLATE.format(facts=body)


def generate(result: dict, tables: dict[str, pd.DataFrame], call,
             attempts: int = MAX_ATTEMPTS) -> dict:
    """Ask, verify, and ask again. Publish only a draft with nothing against it.

    `call(system, user)` returns a string. Kept as a parameter so the harness can drive
    this with recorded drafts and no network, which is the only way the retry logic is
    testable at all.
    """
    f = facts(result, tables)
    question = prompt(result, tables)
    history = []

    for attempt in range(1, attempts + 1):
        ask = question
        if history:
            problems = "\n".join(f"- {v}" for v in history[-1]["violations"])
            ask = (
                f"{question}\n\nYour previous draft was rejected by the verifier:\n"
                f"{problems}\n\nWrite it again, fixing every one of those."
            )
        draft = call(SYSTEM, ask)
        violations = verify(draft, f)
        history.append({
            "attempt": attempt,
            "draft": draft,
            "violations": [str(v) for v in violations],
        })
        if not violations:
            return {
                "text": draft,
                "verified": True,
                "attempts": attempt,
                "history": history,
            }

    # Nothing verified. Publishing the least-bad draft would be the exact behaviour
    # this module exists to prevent, so nothing is published.
    return {"text": "", "verified": False, "attempts": attempts, "history": history}


def load() -> dict:
    """The committed narration, if one verified. Absent is a normal state."""
    if not NARRATION_PATH.exists():
        return {}
    record = json.loads(NARRATION_PATH.read_text(encoding="utf-8"))
    return record if record.get("verified") and record.get("text") else {}
