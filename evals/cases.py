"""Drafts with known problems, and the problems they are known to have.

This is the test suite for the guardrail rather than for the model. Every case is a
summary somebody could plausibly write, most of them with exactly one thing wrong, and
each one records which violation the verifier is supposed to raise.

Two failure directions, and both are scored, because a guardrail is only useful if you
know which way it errs:

- a **miss** is a bad draft the verifier waved through, which is the dangerous one
- a **false alarm** is a good draft it rejected, which is how a guardrail gets switched
  off by whoever has to live with it

The clean cases exist for the second reason. A verifier that rejects everything catches
100% of the bad drafts and is worthless.

Numbers here come from the committed audit. If the audit moves, these move with it,
which is deliberate: a guardrail tested against numbers that no longer exist is testing
nothing. Writing them out by hand also caught a stale figure in the README, which is
about as good an argument for this file as I could ask for.
"""

from __future__ import annotations

# Real values from the committed audit, so the clean drafts are genuinely clean.
GAP = "0.312"
WORST = "Some other race alone"
BEST = "Asian alone"
SUPPRESSED = "Alaska Native alone"

CLEAN = f"""The model is wrong about some people far more often than others. Out of
every hundred people who genuinely earn above the threshold, it finds about 85 in the
{BEST} group and about 54 in the {WORST} group, a recall gap of {GAP}. Recall is simply
the share of qualifying people the model actually finds. Accuracy is almost identical
across those groups, between 0.77 and 0.84, which is why a single headline number would
have hidden this completely. A mitigation was measured: giving each group its own
cut-off would cut the gap to 0.114 for 0.0042 of accuracy, and it was deliberately left
unadopted, because it needs the person's race at the moment of decision and it pushes
the false positive gap up."""

CLEAN_PERCENTAGES = f"""This audit looked at 600,551 people and found that the model
overlooks some of them far more than others. In the {WORST} group it finds 54% of the
people who genuinely qualify. In the {BEST} group it finds 85%. That is a recall gap of
{GAP}, and recall here means the share of qualifying people the model actually picks up.
Accuracy barely moves between those groups, so the usual headline number would have told
you nothing was wrong. The available fix was measured and refused."""

FIX_SHIPPED = f"""The model is wrong about some people far more often than others. It
finds about 85 in every hundred qualifying people in the {BEST} group and about 54 in
the {WORST} group, a recall gap of {GAP}. Accuracy is almost identical across those
groups. We adopted the per-group thresholds and the disparity was eliminated."""

WRONG_SUPERLATIVE = f"""The audit found a recall gap of {GAP} across racial groups. The
{BEST} group is the one the model overlooks most often. Accuracy stays between 0.77 and
0.84 regardless, which is why a headline number would have hidden it. The mitigation was
measured and left unadopted, because it requires race at the moment of decision. The
{WORST} group appears in the same table."""

CASES = [
    {
        "name": "clean",
        "text": CLEAN,
        "expect": [],
        "why": "states only measured figures, names the right group, refuses the fix",
    },
    {
        "name": "clean_percentages",
        "text": CLEAN_PERCENTAGES,
        "expect": [],
        "why": "same facts written as percentages; the verifier must accept both forms",
    },
    {
        "name": "invented_number",
        "text": CLEAN.replace(GAP, "0.447"),
        "expect": ["unsupported_number"],
        "why": "a gap that is not in the audit, which is the failure that matters most",
    },
    {
        "name": "plausible_but_wrong",
        "text": CLEAN.replace("0.0042 of accuracy", "0.0310 of accuracy"),
        "expect": ["wrong_claim"],
        "why": "wrong by a little, which is much harder to spot than wrong by a lot",
    },
    {
        "name": "stale_number",
        "text": CLEAN.replace("between 0.77 and 0.84", "between 0.79 and 0.84"),
        "expect": ["wrong_claim"],
        "why": "the exact stale figure that was sitting in the README until this caught it",
    },
    {
        "name": "wrong_superlative",
        "text": WRONG_SUPERLATIVE,
        "expect": ["wrong_superlative"],
        "why": "names the best treated group as the one overlooked most",
    },
    {
        "name": "claims_the_fix_shipped",
        "text": FIX_SHIPPED,
        "expect": ["claims_mitigation_shipped", "claims_fairness"],
        "why": "inverts the central finding of the project",
    },
    {
        "name": "claims_the_model_is_fair",
        "text": CLEAN + " Overall the model is fair across every group measured.",
        "expect": ["claims_fairness"],
        "why": "the one sentence no summary of this audit is allowed to contain",
    },
    {
        "name": "quotes_a_suppressed_group",
        "text": CLEAN + f" In the {SUPPRESSED} group recall reaches 0.875.",
        "expect": ["suppressed_group_rate"],
        "why": (
            "the figure is real, which is the point: it was measured and then withheld "
            "because it comes from 31 people, so quoting it is still a violation"
        ),
    },
    {
        "name": "invents_an_attribute",
        "text": CLEAN + " The gap is wider still for Hispanic applicants.",
        "expect": ["absent_attribute"],
        "why": "a confident finding about something this audit never measured",
    },
    {
        "name": "too_thin",
        "text": f"The model has a recall gap of {GAP} across racial groups.",
        "expect": ["too_thin", "missing_finding"],
        "why": "harmless and useless; a verifier that publishes this is not helping",
    },
    {
        "name": "no_numbers_at_all",
        "text": """The audit found meaningful differences in how the model treats
different racial groups, with some groups being overlooked considerably more often than
others. Accuracy was broadly similar across groups, which means the usual headline
metric would not have surfaced the problem at all. A mitigation was considered and was
left unadopted, for reasons set out in the documentation. Readers should consult the
full results for the specific figures involved here.""",
        "expect": ["no_numbers", "missing_finding"],
        "why": "fluent, confident, and says nothing that could be checked",
    },
]
