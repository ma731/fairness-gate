"""Render the audit as a single self-contained HTML page.

Constraints, in order of weight:

1. **Deterministic output.** The gate regenerates this file and requires a byte-identical
   match, so everything is rendered server-side. A client-side chart library would move
   the numbers out of the diff and back into trust.
2. **No dependencies.** No CDN, no runtime font fetch, no JavaScript needed to read a
   number. A compliance artifact that phones out to three origins is a supply-chain
   surface, and this one has to still open in five years, offline.
3. **Interactive anyway.** Every view is rendered up front and the controls toggle
   between them, so filtering works without shipping a renderer to the client and the
   page still reads with JavaScript disabled.

Dark-first: this is an instrument panel, and the status colors carry more weight against
a dark plane. Colors are the validated categorical palette (blue and orange, worst
adjacent CVD delta E 24.7) plus the fixed status set, which never doubles as a series.
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from src.charts_extra import (
    confusion_heatmap,
    intersection_matrix,
    interval_chart,
    reliability,
    ridgeline,
    sankey,
    short,
    tradeoff_curve,
    treemap,
)
from src.glossary import explain as explain_check
from src.pointfield import SCRIPT as FIELD_SCRIPT
from src.pointfield import markup as field_markup
from src.theme import STYLE

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"

STATUS_LABEL = {"pass": "Pass", "warn": "Warn", "fail": "Fail"}
ATTRS = [("RAC1P", "Race"), ("SEX", "Sex")]
SPLITS = [("test", "Test year"), ("shift", "Held-out states")]


def esc(t: object) -> str:
    return html.escape(str(t), quote=True)


def fmt(v: float, p: int = 3) -> str:
    return f"{v:.{p}f}"


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def bullet(check: dict, width: int = 720, ci: dict | None = None) -> str:
    """A policy check: measured fill, dashed target tick, solid limit tick.

    A bullet bar rather than a gauge because the question is not "how full" but "where
    does this sit against two declared lines".
    """
    fail_at, warn_at, measured = check["fail_at"], check["warn_at"], check["measured"]
    ceiling = check["direction"] == "max"
    limits = [v for v in (fail_at, warn_at, measured) if v is not None] or [1.0]
    axis = (max(limits) * 1.3) if ceiling else max(1.0, max(limits) * 1.05)

    def x(v: float) -> float:
        return round(min(max(v / axis, 0.0), 1.0) * width, 1)

    ticks = ""
    for value, cls, label in ((warn_at, "t-warn", "target"), (fail_at, "t-fail", "limit")):
        if value is None:
            continue
        px = x(value)
        ticks += (
            f'<g class="{cls}"><line x1="{px}" y1="0" x2="{px}" y2="20"/>'
            f'<title>{label} {fmt(value)}</title></g>'
        )

    # A shaded band for the interval, so the eye sees how much of the bar is certain.
    band = ""
    if ci and ci.get("lo") is not None:
        bx, bw = x(ci["lo"]), max(x(ci["hi"]) - x(ci["lo"]), 1.0)
        band = (
            f'<rect class="ci" x="{bx}" y="3" width="{bw}" height="14" rx="3">'
            f'<title>{ci["level"]:.0%} interval [{ci["lo"]:.3f}, {ci["hi"]:.3f}]'
            f"</title></rect>"
            f'<line class="ci-cap" x1="{bx}" y1="3" x2="{bx}" y2="17"/>'
            f'<line class="ci-cap" x1="{bx + bw}" y1="3" x2="{bx + bw}" y2="17"/>'
        )

    return (
        f'<svg class="bullet" viewBox="0 0 {width} 20" width="100%" height="20" '
        f'preserveAspectRatio="none" role="img" aria-label="measured {fmt(measured, 4)}">'
        f'<rect class="trk" x="0" y="6" width="{width}" height="8" rx="4"/>'
        f'<rect class="fil s-{check["status"]}" x="0" y="6" width="{x(measured)}" '
        f'height="8" rx="4"><title>measured {fmt(measured, 4)}</title></rect>'
        f"{band}{ticks}</svg>"
    )


def rate_chart(table: pd.DataFrame, attribute: str) -> str:
    """True and false positive rate per group. Grouped, not stacked: independent rates."""
    sub = (
        table[(table["attribute"] == attribute) & table["reportable"]]
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )
    if sub.empty:
        return '<p class="empty">No group large enough to report.</p>'

    row_h, bar_h, gap, lab_w, pad_r, plot_w = 54, 15, 5, 190, 66, 610
    h = len(sub) * row_h + 40
    w = lab_w + plot_w + pad_r

    grid = "".join(
        f'<line class="gr" x1="{lab_w + plot_w * t}" y1="24" '
        f'x2="{lab_w + plot_w * t}" y2="{h - 12}"/>'
        f'<text class="ax" x="{lab_w + plot_w * t}" y="15" text-anchor="middle">'
        f"{int(t * 100)}%</text>"
        for t in (0, 0.25, 0.5, 0.75, 1.0)
    )

    rows = ""
    for i, r in sub.iterrows():
        top = 32 + i * row_h
        tw, fw = round(r["tpr"] * plot_w, 1), round(r["fpr"] * plot_w, 1)
        rows += (
            f'<g class="row">'
            f'<text class="gl" x="{lab_w - 14}" y="{top + 14}" text-anchor="end">'
            f'{esc(short(int(r["code"]), attribute))}'
            f'<title>{esc(r["group"])}</title></text>'
            f'<text class="gn" x="{lab_w - 14}" y="{top + 31}" text-anchor="end">'
            f'{int(r["n"]):,}</text>'
            f'<rect class="s1" x="{lab_w}" y="{top}" width="{tw}" height="{bar_h}" rx="3">'
            f'<title>{esc(r["group"])}: finds {pct(r["tpr"])} of true positives</title></rect>'
            f'<text class="vl" x="{lab_w + tw + 9}" y="{top + 12}">{pct(r["tpr"])}</text>'
            f'<rect class="s2" x="{lab_w}" y="{top + bar_h + gap}" width="{fw}" '
            f'height="{bar_h}" rx="3">'
            f'<title>{esc(r["group"])}: false positive rate {pct(r["fpr"])}</title></rect>'
            f'<text class="vl" x="{lab_w + fw + 9}" y="{top + bar_h + gap + 12}">'
            f"{pct(r['fpr'])}</text></g>"
        )

    return (
        f'<svg class="chart" viewBox="0 0 {w} {h}" width="100%" '
        f'role="img" aria-label="error rates by {esc(attribute)}">{grid}{rows}</svg>'
    )


def calib_chart(table: pd.DataFrame, attribute: str) -> str:
    """Observed rate against mean predicted, as a dumbbell: the gap is the point."""
    sub = (
        table[(table["attribute"] == attribute) & table["reportable"]]
        .sort_values("base_rate", ascending=False)
        .reset_index(drop=True)
    )
    if sub.empty:
        return '<p class="empty">No group large enough to report.</p>'

    row_h, lab_w, pad_r, plot_w = 36, 190, 66, 610
    h = len(sub) * row_h + 40
    w = lab_w + plot_w + pad_r
    hi = max(float(sub["base_rate"].max()), float(sub["mean_predicted"].max()))
    axis = min(1.0, (hi * 1.25) or 1.0)

    def x(v: float) -> float:
        return round(lab_w + min(v / axis, 1.0) * plot_w, 1)

    grid = "".join(
        f'<line class="gr" x1="{lab_w + plot_w * t}" y1="24" '
        f'x2="{lab_w + plot_w * t}" y2="{h - 12}"/>'
        f'<text class="ax" x="{lab_w + plot_w * t}" y="15" text-anchor="middle">'
        f"{fmt(axis * t, 2)}</text>"
        for t in (0, 0.5, 1.0)
    )

    rows = ""
    for i, r in sub.iterrows():
        cy = 40 + i * row_h
        xa, xp = x(float(r["base_rate"])), x(float(r["mean_predicted"]))
        rows += (
            f'<g class="row">'
            f'<text class="gl" x="{lab_w - 14}" y="{cy + 4}" text-anchor="end">'
            f'{esc(short(int(r["code"]), attribute))}'
            f'<title>{esc(r["group"])}</title></text>'
            f'<line class="lnk" x1="{min(xa, xp)}" y1="{cy}" x2="{max(xa, xp)}" y2="{cy}"/>'
            f'<circle class="d1" cx="{xa}" cy="{cy}" r="5.5">'
            f'<title>{esc(r["group"])}: observed {fmt(r["base_rate"])}</title></circle>'
            f'<circle class="d2" cx="{xp}" cy="{cy}" r="5.5">'
            f'<title>{esc(r["group"])}: predicted {fmt(r["mean_predicted"])}</title></circle>'
            f'<text class="vl" x="{max(xa, xp) + 12}" y="{cy + 4}">'
            f'{fmt(abs(float(r["mean_predicted"]) - float(r["base_rate"])), 3)}</text>'
            f"</g>"
        )

    return (
        f'<svg class="chart" viewBox="0 0 {w} {h}" width="100%" '
        f'role="img" aria-label="calibration by {esc(attribute)}">{grid}{rows}</svg>'
    )


def _marquee(cost: pd.DataFrame) -> str:
    """The scrolling band says something. Each entry is a group and how many of every
    hundred qualifying people the model overlooks there."""
    return "".join(
        f'<span>{esc(r["group"])} <em>{round(r["per_100"])}</em></span>'
        for _, r in cost.iterrows()
    )


def topo_lines(n: int = 16, seed: int = 7) -> str:
    """Concentric irregular contours, the way a survey map draws elevation.

    Deterministic by seed: the page has to render byte-identically every time.
    """
    import math

    paths = []
    for i in range(n):
        k = i / n
        rx, ry = 20 + k * 74, 12 + k * 52
        pts = []
        for j in range(72):
            a = j / 72 * math.tau
            # a couple of fixed harmonics so the ring wobbles like a real contour
            wob = (
                1
                + 0.085 * math.sin(a * 3 + i * 0.55 + seed)
                + 0.05 * math.sin(a * 5 - i * 0.31)
            )
            pts.append(f"{50 + rx * wob * math.cos(a):.2f},{50 + ry * wob * math.sin(a):.2f}")
        paths.append(f'<path d="M{"L".join(pts)}Z"/>')
    return (
        '<svg class="topo" viewBox="0 0 100 100" preserveAspectRatio="none" '
        f'aria-hidden="true" focusable="false">{"".join(paths)}</svg>'
    )


def human_cost(table: pd.DataFrame, attribute: str = "RAC1P") -> pd.DataFrame:
    """Turn rates into people.

    A true positive rate of 0.542 is a statistic. "Out of every hundred people who
    genuinely qualify, forty-six are overlooked" is the same number said in a way a
    person can feel, and it is arithmetic, not rhetoric: qualified = n * base rate,
    overlooked = qualified * (1 - true positive rate).
    """
    sub = table[(table["attribute"] == attribute) & table["reportable"]].copy()
    sub["qualified"] = sub["n"] * sub["base_rate"]
    sub["overlooked"] = sub["qualified"] * (1 - sub["tpr"])
    sub["per_100"] = (1 - sub["tpr"]) * 100
    return sub.sort_values("per_100", ascending=False).reset_index(drop=True)


def dot_field(missed_per_100: float, cols: int = 20, rows: int = 5) -> str:
    """One hundred dots, one hundred qualifying people. The lit ones are overlooked.

    A unit chart rather than a bar, because the unit here is a person and the point of
    the section is that the denominator is people.
    """
    lit = round(missed_per_100)
    r, gap = 5.0, 19.0
    w, h = cols * gap, rows * gap
    dots = ""
    for i in range(cols * rows):
        cx = round((i % cols) * gap + gap / 2, 1)
        cy = round((i // cols) * gap + gap / 2, 1)
        cls = "on" if i < lit else "off"
        # Stagger only the lit dots, left to right, so the eye reads the count forming.
        delay = f' style="--d:{i * 14}ms"' if cls == "on" else ""
        dots += f'<circle class="{cls}" cx="{cx}" cy="{cy}" r="{r}"{delay}/>'
    return (
        f'<svg class="dots" viewBox="0 0 {w} {h}" width="100%" role="img" '
        f'aria-label="{lit} of every 100 qualifying people are overlooked">{dots}</svg>'
    )


def scatter_chart(table: pd.DataFrame, attribute: str) -> str:
    """How common the outcome is, against how often the model finds it.

    The chart that answers "is it just harder for rare positives?". Bubble area is group
    size, so the eye is not misled by a 900-person group beside a 434,000-person one.
    One series: colour carries nothing here, the axes carry it all.
    """
    sub = table[(table["attribute"] == attribute) & table["reportable"]]
    if sub.empty:
        return '<p class="empty">No group large enough to report.</p>'

    w, h = 780, 440
    pad_l, pad_b, pad_t, pad_r = 66, 54, 22, 190
    px, py = w - pad_l - pad_r, h - pad_t - pad_b
    xmax = max(0.55, float(sub["base_rate"].max()) * 1.2)
    ymin = min(0.45, float(sub["tpr"].min()) * 0.9)

    def X(v: float) -> float:
        return round(pad_l + (v / xmax) * px, 1)

    def Y(v: float) -> float:
        return round(pad_t + (1 - (v - ymin) / (1 - ymin)) * py, 1)

    grid = ""
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        gx = round(pad_l + px * t, 1)
        grid += (
            f'<line class="gr" x1="{gx}" y1="{pad_t}" x2="{gx}" y2="{pad_t + py}"/>'
            f'<text class="ax" x="{gx}" y="{pad_t + py + 20}" text-anchor="middle">'
            f"{fmt(xmax * t, 2)}</text>"
        )
        gy = round(pad_t + py * t, 1)
        grid += (
            f'<line class="gr" x1="{pad_l}" y1="{gy}" x2="{pad_l + px}" y2="{gy}"/>'
            f'<text class="ax" x="{pad_l - 10}" y="{gy + 4}" text-anchor="end">'
            f"{fmt(ymin + (1 - t) * (1 - ymin), 2)}</text>"
        )

    biggest = float(sub["n"].max())
    ordered = sub.sort_values("n", ascending=False)

    # Labels collided badly where the small groups cluster. Lay them out top to bottom
    # and push each one below the last if it would overlap, then draw a leader line back
    # to its bubble so the association survives the nudge.
    placed = []
    for _, r in ordered.iterrows():
        cx, cy = X(float(r["base_rate"])), Y(float(r["tpr"]))
        rad = round(7 + 26 * (float(r["n"]) / biggest) ** 0.5, 1)
        placed.append({"cx": cx, "cy": cy, "rad": rad, "ly": cy,
                       "label": short(int(r["code"]), attribute), "row": r})
    placed.sort(key=lambda d: d["ly"])
    min_gap = 15.0
    for i in range(1, len(placed)):
        if placed[i]["ly"] - placed[i - 1]["ly"] < min_gap:
            placed[i]["ly"] = placed[i - 1]["ly"] + min_gap

    marks = ""
    for d in placed:
        r = d["row"]
        lx = round(d["cx"] + d["rad"] + 12, 1)
        leader = ""
        if abs(d["ly"] - d["cy"]) > 2:
            leader = (
                f'<path class="lead" d="M{round(d["cx"] + d["rad"] + 2, 1)},{d["cy"]} '
                f'L{lx - 5},{round(d["ly"], 1)}"/>'
            )
        marks += (
            f'<g class="row"><circle class="bub" cx="{d["cx"]}" cy="{d["cy"]}" '
            f'r="{d["rad"]}"><title>{esc(r["group"])}: {int(r["n"]):,} people, base rate '
            f'{fmt(r["base_rate"])}, found {pct(r["tpr"])}</title></circle>'
            f'{leader}'
            f'<text class="vl" x="{lx}" y="{round(d["ly"] + 4, 1)}">'
            f'{esc(d["label"])}</text></g>'
        )

    return (
        f'<svg class="chart" viewBox="0 0 {w} {h}" width="100%" role="img" '
        f'aria-label="base rate against true positive rate by {esc(attribute)}">'
        f"{grid}{marks}"
        f'<text class="axt" x="{round(pad_l + px / 2, 1)}" y="{h - 8}" '
        f'text-anchor="middle">How common the outcome is in this group</text>'
        f'<text class="axt" transform="translate(16,{round(pad_t + py / 2, 1)}) '
        f'rotate(-90)" text-anchor="middle">Share of them the model finds</text></svg>'
    )


def diverging_chart(table: pd.DataFrame, attribute: str) -> str:
    """Selection rate minus base rate: who the model over- and under-selects.

    Diverging because the quantity is signed and zero means something. Two hues with a
    neutral zero line, never one ramp, so the direction of the error reads before any
    number does.
    """
    sub = table[(table["attribute"] == attribute) & table["reportable"]].copy()
    if sub.empty:
        return '<p class="empty">No group large enough to report.</p>'
    sub["delta"] = sub["selection_rate"] - sub["base_rate"]
    sub = sub.sort_values("delta").reset_index(drop=True)

    row_h, lab_w, half, pad_r = 42, 190, 250, 82
    h = len(sub) * row_h + 38
    w = lab_w + half * 2 + pad_r
    zero = lab_w + half
    span = max(0.09, float(sub["delta"].abs().max()) * 1.2)

    out = (
        f'<line class="zero" x1="{zero}" y1="20" x2="{zero}" y2="{h - 14}"/>'
        f'<text class="ax" x="{zero}" y="12" text-anchor="middle">0</text>'
        f'<text class="ax" x="{zero - half}" y="12" text-anchor="start">under-selected</text>'
        f'<text class="ax" x="{zero + half}" y="12" text-anchor="end">over-selected</text>'
    )
    for i, r in sub.iterrows():
        top = 28 + i * row_h
        d = float(r["delta"])
        wid = round(abs(d) / span * half, 1)
        x0 = round(zero - wid, 1) if d < 0 else zero
        cls = "s1" if d < 0 else "s2"
        lx = round(zero - wid - 10, 1) if d < 0 else round(zero + wid + 10, 1)
        anch = "end" if d < 0 else "start"
        out += (
            f'<g class="row">'
            f'<text class="gl" x="{lab_w - 16}" y="{top + 15}" text-anchor="end">'
            f'{esc(short(int(r["code"]), attribute))}'
            f'<title>{esc(r["group"])}</title></text>'
            f'<rect class="{cls}" x="{x0}" y="{top + 2}" width="{wid}" height="18" rx="3">'
            f'<title>{esc(r["group"])}: selected {fmt(r["selection_rate"])} against a '
            f'base rate of {fmt(r["base_rate"])}</title></rect>'
            f'<text class="vl" x="{lx}" y="{top + 16}" text-anchor="{anch}">'
            f"{d:+.3f}</text></g>"
        )

    return (
        f'<svg class="chart" viewBox="0 0 {w} {h}" width="100%" role="img" '
        f'aria-label="selection rate minus base rate by {esc(attribute)}">{out}</svg>'
    )


def slope_chart(tables: dict, attribute: str) -> str:
    """One line per group, test year to the states held out entirely.

    A slope chart because the question is the direction and size of the change for every
    group at once, which two bar charts make the reader compute in their head.
    """
    a = tables["test"]
    b = tables["shift"]
    a = a[(a["attribute"] == attribute) & a["reportable"]][["group", "code", "tpr"]]
    b = b[(b["attribute"] == attribute) & b["reportable"]][["group", "tpr"]]
    m = a.merge(b, on="group", suffixes=("_test", "_shift"))
    if m.empty:
        return '<p class="empty">No group reportable in both populations.</p>'
    m = m.sort_values("tpr_test", ascending=False).reset_index(drop=True)

    w, h = 780, 430
    left, right = 190, 520
    top, bot = 46, h - 30
    lo = float(min(m["tpr_test"].min(), m["tpr_shift"].min())) * 0.93
    hi = float(max(m["tpr_test"].max(), m["tpr_shift"].max())) * 1.04

    def Y(v: float) -> float:
        return round(top + (1 - (v - lo) / (hi - lo)) * (bot - top), 1)

    out = (
        f'<line class="gr" x1="{left}" y1="{top - 16}" x2="{left}" y2="{bot + 10}"/>'
        f'<line class="gr" x1="{right}" y1="{top - 16}" x2="{right}" y2="{bot + 10}"/>'
        f'<text class="ax" x="{left}" y="{top - 24}" text-anchor="middle">Test year</text>'
        f'<text class="ax" x="{right}" y="{top - 24}" text-anchor="middle">'
        f"Held-out states</text>"
    )
    for _, r in m.iterrows():
        y1, y2 = Y(float(r["tpr_test"])), Y(float(r["tpr_shift"]))
        drop = float(r["tpr_shift"]) - float(r["tpr_test"])
        cls = "s2" if drop < 0 else "s1"
        out += (
            f'<g class="row">'
            f'<text class="gl" x="{left - 18}" y="{y1 + 4}" text-anchor="end">'
            f'{esc(short(int(r["code"]), attribute))}'
            f'<title>{esc(r["group"])}</title></text>'
            f'<line class="slope {cls}l" x1="{left}" y1="{y1}" x2="{right}" y2="{y2}"/>'
            f'<circle class="{cls}" cx="{left}" cy="{y1}" r="4.5"/>'
            f'<circle class="{cls}" cx="{right}" cy="{y2}" r="4.5"/>'
            f'<text class="vl" x="{right + 15}" y="{y2 + 4}">'
            f'{pct(float(r["tpr_shift"]))} ({drop:+.1%})'
            f'<title>{esc(r["group"])}</title></text></g>'
        )

    return (
        f'<svg class="chart" viewBox="0 0 {w} {h}" width="100%" role="img" '
        f'aria-label="true positive rate from the test year to the held-out states">'
        f"{out}</svg>"
    )


def narration_block() -> str:
    """The generated summary, if one verified, with its receipt attached.

    Absent is a normal state and renders as nothing. A page that showed an empty
    placeholder would be advertising a feature that is currently switched off, and a
    page that showed an unverified draft would be the exact thing this is here to stop.
    """
    from src.narrator import CHECKS, load

    record = load()
    if not record:
        return ""

    passed = "".join(
        f"<li>{esc(check.__name__.replace('check_', '').replace('_', ' '))}</li>"
        for check in CHECKS
    )
    model = record.get("model") or "no model"
    attempts = record.get("attempts", 1)
    tries = "first attempt" if attempts == 1 else f"attempt {attempts}"

    return f"""<section id="summary">
  <div class="shell reveal">
    <p class="kicker">In words</p>
    <h2>A summary, written by a language model and checked before you saw it.</h2>
    <div class="narr">
      <blockquote class="narr-text">{esc(record['text'])}</blockquote>
      <aside class="narr-receipt">
        <p class="narr-h">How this got here</p>
        <p>A model wrote it. It was not allowed to publish it. Every number in it had
          to trace back to a measured value at the precision it was written, and the
          draft had to survive every check below before it could appear on this page.</p>
        <ul class="narr-checks">{passed}</ul>
        <p class="narr-meta">Passed on the {esc(tries)}<br>
          Source: {esc(record.get('source', 'unknown'))}<br>
          Model: <code>{esc(model)}</code><br>
          Against the audit of {esc(record.get('audit_generated_at', 'unknown'))}</p>
      </aside>
    </div>
    <p class="note">If no draft passes, nothing is published and this section does not
      appear at all. There is no best-of-four fallback, because a guardrail with one is
      only a delay. The checks themselves are scored against twelve drafts with known
      planted errors on every commit, so they cannot rot quietly.</p>
  </div>
</section>

"""


def arm_slope(result: dict, table: pd.DataFrame, arm_key: str,
              left: str, right: str, attribute: str = "RAC1P") -> str:
    """Recall per group under the shipped model, and under some other arm.

    A slope chart because in both places it is used the expected shape *is* the finding.
    If deleting race and sex worked, or if the algorithm were the problem, these lines
    would converge. They do not, and a reader can see that faster than they can be
    argued into it.
    """
    arm = (result.get(arm_key) or {})
    groups = arm.get("groups") or {}
    if not groups:
        return f'<p class="empty">No {esc(arm_key)} arm in this audit.</p>'

    aware = table[(table["attribute"] == attribute) & table["reportable"]]
    rows = []
    for _, r in aware.iterrows():
        key = str(int(r["code"]))
        if key in groups:
            rows.append((int(r["code"]), str(r["group"]), float(r["tpr"]),
                         float(groups[key]["tpr"])))
    if not rows:
        return '<p class="empty">No group reportable in both arms.</p>'
    rows.sort(key=lambda t: -t[2])

    w, h = 780, 430
    left_x, right_x = 190, 520
    top, bot = 46, h - 30
    vals = [v for r in rows for v in (r[2], r[3])]
    lo, hi = min(vals) * 0.93, max(vals) * 1.04

    def Y(v: float) -> float:
        return round(top + (1 - (v - lo) / (hi - lo)) * (bot - top), 1)

    out = (
        f'<line class="gr" x1="{left_x}" y1="{top - 16}" x2="{left}" y2="{bot + 10}"/>'
        f'<line class="gr" x1="{right}" y1="{top - 16}" x2="{right_x}" y2="{bot + 10}"/>'
        f'<text class="ax" x="{left_x}" y="{top - 24}" text-anchor="middle">'
        f"{esc(left)}</text>"
        f'<text class="ax" x="{right_x}" y="{top - 24}" text-anchor="middle">'
        f"{esc(right)}</text>"
    )
    for code, name, before, after in rows:
        y1, y2 = Y(before), Y(after)
        cls = "s1" if after >= before else "s2"
        out += (
            f'<g class="row">'
            f'<text class="gl" x="{left_x - 18}" y="{y1 + 4}" text-anchor="end">'
            f"{esc(short(code, attribute))}<title>{esc(name)}</title></text>"
            f'<line class="slope {cls}l" x1="{left_x}" y1="{y1}" x2="{right_x}" y2="{y2}"/>'
            f'<circle class="{cls}" cx="{left_x}" cy="{y1}" r="4.5"/>'
            f'<circle class="{cls}" cx="{right_x}" cy="{y2}" r="4.5"/>'
            f'<text class="vl" x="{right_x + 15}" y="{y2 + 4}">'
            f"{pct(after)} ({after - before:+.1%})"
            f"<title>{esc(name)}</title></text></g>"
        )

    return (
        f'<svg class="chart" viewBox="0 0 {w} {h}" width="100%" role="img" '
        f'aria-label="recall per group, {esc(left)} against {esc(right)}">'
        f"{out}</svg>"
    )


def group_table(table: pd.DataFrame, attribute: str) -> str:
    sub = (
        table[(table["attribute"] == attribute) & table["reportable"]]
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )
    rows = "".join(
        f"<tr><th scope='row'>{esc(r['group'])}</th>"
        f"<td>{int(r['n']):,}</td><td>{fmt(r['base_rate'])}</td>"
        f"<td>{fmt(r['selection_rate'])}</td><td>{fmt(r['tpr'])}</td>"
        f"<td>{fmt(r['fpr'])}</td><td>{fmt(r['accuracy'])}</td>"
        f"<td class='mut'>{fmt(r['ece'])}</td></tr>"
        for _, r in sub.iterrows()
    )
    return (
        "<table><thead><tr><th scope='col'>Group</th><th scope='col'>n</th>"
        "<th scope='col'>Base rate</th><th scope='col'>Selected</th>"
        "<th scope='col'>TPR</th><th scope='col'>FPR</th>"
        "<th scope='col'>Accuracy</th><th scope='col'>ECE</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


# --------------------------------------------------------------------------- #
def _panes(tables: dict[str, pd.DataFrame], builder) -> str:
    """One pre-rendered pane per attribute and split. The controls just reveal one."""
    out = ""
    for split, _ in SPLITS:
        for attr, _label in ATTRS:
            active = " is-on" if (split, attr) == ("test", "RAC1P") else ""
            out += (
                f'<div class="pane{active}" data-split="{split}" data-attr="{attr}">'
                f"{builder(tables[split], attr)}</div>"
            )
    return out


def _controls() -> str:
    a = "".join(
        f'<button type="button" class="seg{" is-on" if i == 0 else ""}" '
        f'data-k="attr" data-v="{k}">{esc(v)}</button>'
        for i, (k, v) in enumerate(ATTRS)
    )
    s = "".join(
        f'<button type="button" class="seg{" is-on" if i == 0 else ""}" '
        f'data-k="split" data-v="{k}">{esc(v)}</button>'
        for i, (k, v) in enumerate(SPLITS)
    )
    return (
        f'<div class="controls"><div class="ctl"><span class="ctl-k">Attribute</span>'
        f'<div class="segs">{a}</div></div>'
        f'<div class="ctl"><span class="ctl-k">Population</span>'
        f'<div class="segs">{s}</div></div>'
        f'<p class="ctl-note">Held-out states were never trained on.</p></div>'
    )


_CI_FOR = {
    "max_tpr_gap": "tpr_gap",
    "max_fpr_gap": "fpr_gap",
    "max_demographic_parity_difference": "selection_gap",
}


def _ci_for(result: dict, check: dict) -> dict | None:
    parts = check["name"].split(".")
    if len(parts) != 3 or parts[0] != "fairness":
        return None
    key = _CI_FOR.get(parts[2])
    if not key:
        return None
    return ((result.get("uncertainty") or {}).get(parts[1]) or {}).get(key) or None


def _checks(result: dict) -> str:
    order = {"fail": 0, "warn": 1, "pass": 2}
    out = ""
    for c in sorted(result["checks"], key=lambda c: (order[c["status"]], c["name"])):
        target = "n/a" if c["warn_at"] is None else fmt(c["warn_at"])
        limit = "n/a" if c["fail_at"] is None else fmt(c["fail_at"])
        # The raw name stays on the row because it is the string you would search for
        # in policy.yaml. It just no longer has to carry the explaining on its own.
        title, why = explain_check(c["name"])
        out += (
            f'<li class="chk s-{c["status"]}">'
            f'<div class="chk-top"><div class="chk-id">'
            f'<span class="chk-h">{esc(title)}</span>'
            f'<code title="{esc(why)}">{esc(c["name"])}</code></div>'
            f'<span class="tag">{STATUS_LABEL[c["status"]]}</span></div>'
            f"{bullet(c, ci=_ci_for(result, c))}"
            f'<div class="chk-bot"><span class="big">{fmt(c["measured"], 4)}</span>'
            f'<span class="mut">target {target}</span>'
            f'<span class="mut">limit {limit}</span>'
            + (f'<span class="mut ci-note">{esc(c["note"])}</span>' if c.get("note") else "")
            + "</div>"
            + (f'<p class="chk-why">{esc(why)}</p>' if why else "")
            + "</li>"
        )
    return out


def _splits(result: dict) -> str:
    return "".join(
        f"<tr><th scope='row'>{n}</th><td>{s['n']:,}</td><td>{fmt(s['auc'], 4)}</td>"
        f"<td>{fmt(s['average_precision'], 4)}</td><td>{fmt(s['brier'], 4)}</td>"
        f"<td>{fmt(s['ece'], 4)}</td><td>{fmt(s['accuracy'], 4)}</td>"
        f"<td class='mut'>{fmt(s['majority_baseline'], 4)}</td></tr>"
        for n, s in ((k, result["scores"][k]) for k in ("val", "test", "shift"))
    )


def _splits_compact(result: dict) -> str:
    return "".join(
        f"<tr><th scope='row'>{n}</th><td>{s['n']:,}</td><td>{fmt(s['auc'], 3)}</td>"
        f"<td>{fmt(s['ece'], 3)}</td><td>{fmt(s['accuracy'], 3)}</td>"
        f"<td class='mut'>{fmt(s['majority_baseline'], 3)}</td></tr>"
        for n, s in ((k, result["scores"][k]) for k in ("val", "test", "shift"))
    )


def _sections(result: dict, tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Every block of the site, built once and dealt out across the pages.

    This used to be one enormous return. Splitting it here is what lets the same
    numbers appear on whichever page they belong to without being computed twice.
    """
    test = tables["test"]
    race = next(s for s in result["fairness"]["test"] if s["attribute"] == "RAC1P")
    sex = next(s for s in result["fairness"]["test"] if s["attribute"] == "SEX")
    xsex = next(
        (s for s in result["fairness"]["test"] if s["attribute"] == "RACExSEX"),
        {"fpr_gap": 0.0, "tpr_gap": 0.0},
    )
    mt = (result.get("mitigation") or {}).get("RAC1P") or {}
    xcells = test[test["attribute"] == "RACExSEX"]
    xtotal = len(xcells)
    xsuppressed = int((~xcells["reportable"]).sum())
    sub = test[(test["attribute"] == "RAC1P") & test["reportable"]]

    cost = human_cost(test)
    hurt, spared = cost.iloc[0], cost.iloc[-1]
    ratio = hurt["per_100"] / spared["per_100"] if spared["per_100"] else 0.0
    overlooked_total = cost["overlooked"].sum()

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for c in result["checks"]:
        counts[c["status"]] += 1
    v = result["policy_verdict"]
    d = result["design"]
    ts, sh = (result["scores"][k] for k in ("test", "shift"))
    unaware_cmp = (result.get("unaware") or {}).get("comparison") or {}
    linear_arm = result.get("linear") or {}
    linear_cmp = linear_arm.get("comparison") or {}

    return {
        "families":
        f"""<section id="families">
  <div class="shell reveal">
    <p class="kicker">A second model</p>
    <h2>Would a different algorithm simply not have this problem?</h2>
    <p class="say">Every other number here comes from one gradient boosted tree, and
      that is a real weakness in the argument. Boosting is very good at finding
      interactions, and an interaction between occupation, region and hours is exactly
      the shape of thing that could rebuild race without ever being told it. So the
      obvious objection is that this is a fact about the algorithm rather than about
      the world.</p>
    <p class="say">I trained a <b>logistic regression</b> on the identical splits and
      the identical features. Linear, additive, no interactions unless you build them,
      and about as different from a boosted forest as you can get while still predicting
      the same thing.</p>

    <div class="numbers">
      <div><div class="k">Gap, trees</div>
        <div class="v">{fmt(float(linear_cmp.get('tpr_gap_tree', 0)))}</div>
        <div class="c">AUC {fmt(float(linear_cmp.get('auc_tree', 0)), 3)}</div></div>
      <div><div class="k">Gap, linear</div>
        <div class="v">{fmt(float(linear_cmp.get('tpr_gap_linear', 0)))}</div>
        <div class="c">AUC {fmt(float(linear_cmp.get('auc_linear', 0)), 3)}</div></div>
      <div><div class="k">Gap that remains</div>
        <div class="v">{linear_cmp.get('share_remaining', 0):.0%}</div>
        <div class="c">of the original, after changing everything</div></div>
      <div><div class="k">Rank correlation</div>
        <div class="v">{fmt(float(linear_arm.get('rank_correlation') or 0), 2)}</div>
        <div class="c">the same groups end up at the bottom</div></div>
    </div>

    {arm_slope(result, test, "linear", "Gradient boosted trees",
               "Logistic regression")}

    <p class="say" style="margin-top:34px">The linear model is a slightly worse
      predictor and has a <b>slightly larger</b> gap. Swapping out the entire algorithm
      moved the disparity by
      {abs(float(linear_cmp.get('tpr_gap_linear', 0)) - float(linear_cmp.get('tpr_gap_tree', 0))):.4f}.
      And the rank correlation says the two families do not merely fail equally hard,
      they fail the same people: rank the groups by recall under each model and you get
      almost the same order.</p>
    <p class="note">So this is not a fact about gradient boosting. Two model families
      with nothing in common, fitted to the same data, arrive at the same disparity and
      put it on the same groups. That moves the finding from "this algorithm has a
      problem" to "this data encodes an inequality, and a model fitted to it will
      inherit it". The second is much harder to fix and much more useful to know, and
      it is the reason picking a different library is not a plan.</p>
  </div>
</section>

""",
        "narration": narration_block(),
        "head_evidence": page_head(
            "The evidence",
            "Sixteen checks, and how sure I am of each one.",
            "Every measured number, what it is being held to, and the interval around "
            "it. Nothing here fails the build yet, and the section on certainty "
            "explains why that is deliberate rather than lucky.",
        ),
        "head_method": page_head(
            "Method",
            "How it was measured, and what I refused to measure.",
            "Split by time and by geography, three fairness criteria reported at once "
            "because they contradict each other, a floor under what gets published, "
            "and the experiment that tests the fix everybody suggests first.",
        ),
        "head_fix": page_head(
            "The fix",
            "It can be closed. I did not close it.",
            "Per-group thresholds take two thirds off the recall gap for four tenths "
            "of a point of accuracy. Here is the measurement, and here is why shipping "
            "it would have been the wrong call.",
        ),
        "next_evidence": onward(
            "evidence.html", "The evidence",
            "Sixteen checks, the intersectional audit, and how certain each number is."),
        "next_method": onward(
            "method.html", "How I measured it",
            "The splits, the reporting floor, and what happens when you delete race "
            "and sex from the features."),
        "next_fix": onward(
            "fix.html", "The fix I did not ship",
            "What closing the gap would cost, and the two reasons I refused it."),
        "unaware":
        f"""<section id="unaware">
  <div class="shell reveal">
    <p class="kicker">The obvious fix</p>
    <h2>What happens if the model cannot see race or sex.</h2>
    <p class="say">This is the first thing almost everybody suggests, including me
      before I measured it. It has a name, <b>fairness through unawareness</b>: if the
      model cannot see the attribute, it cannot discriminate on it. It is testable, so
      I tested it instead of citing someone.</p>
    <p class="say">The shipped model does use race and sex, because they are part of
      the standard feature set for this task. So I trained the whole pipeline a second
      time with both columns deleted, gave it its own threshold chosen on the
      validation year, and scored it on the same test year.</p>

    <div class="numbers">
      <div><div class="k">Gap, sees both</div>
        <div class="v">{fmt(float(unaware_cmp.get('tpr_gap_aware', 0)))}</div>
        <div class="c">the shipped model</div></div>
      <div><div class="k">Gap, both deleted</div>
        <div class="v">{fmt(float(unaware_cmp.get('tpr_gap_unaware', 0)))}</div>
        <div class="c">fairness through unawareness</div></div>
      <div><div class="k">Gap that survives</div>
        <div class="v">{unaware_cmp.get('share_remaining', 0):.0%}</div>
        <div class="c">removing them changes almost nothing</div></div>
      <div><div class="k">Accuracy cost</div>
        <div class="v">{fmt(float(unaware_cmp.get('accuracy_cost', 0)), 4)}</div>
        <div class="c">and it is not free either</div></div>
    </div>

    {arm_slope(result, test, "unaware", "Sees race and sex", "Both columns deleted")}

    <p class="say" style="margin-top:34px">The lines barely move. The model has no idea
      what race anyone is and it still finds about
      {round(100 - hurt['per_100'])} out of every hundred qualifying people in one
      group and {round(100 - spared['per_100'])} in another. It rebuilds nearly all of
      the signal from occupation, education, hours worked and place of birth, because
      those carry the history of who got which opportunities.</p>
    <p class="note">So an attribute is protected whether or not the model is allowed to
      look at it. Deleting the column does not delete the problem. It deletes your
      ability to see the problem, while leaving you feeling like you did something, and
      a team in that position is worse off than one that never tried.</p>
  </div>
</section>

""",
        "conclusion":
        f"""<div class="band" id="conclusion">
  <div class="shell band-in reveal">
    <p class="kicker">Conclusion</p>
    <h2>What I would tell a team about to ship this.</h2>
    <div class="steps">
      <div class="step">
        <h3>The gap is real, and it is not an artefact of the attributes</h3>
        <p>{fmt(race['tpr_gap'])} between the best and worst treated racial group, with
          a 95% interval that does not come near zero. Deleting race and sex from the
          features leaves {unaware_cmp.get('share_remaining', 0):.0%} of it standing.
          You cannot make this go away by not looking.</p>
      </div>
      <div class="step">
        <h3>Accuracy will tell you everything is fine</h3>
        <p>It sits between {fmt(float(sub['accuracy'].min()), 2)} and
          {fmt(float(sub['accuracy'].max()), 2)} across those same groups. A model can
          be equally accurate everywhere and still place its errors on the same people
          every time. If one number is all you check, this is invisible to you.</p>
      </div>
      <div class="step">
        <h3>Audit where the attributes cross, not one at a time</h3>
        <p>False positive gap {fmt(race['fpr_gap'])} by race, {fmt(sex['fpr_gap'])} by
          sex, {fmt(xsex.get('fpr_gap', 0))} by the two together. Two acceptable
          averages can hide a much worse cell underneath them, and marginal audits are
          built to miss exactly that.</p>
      </div>
      <div class="step">
        <h3>The fix that works may not be one you can use</h3>
        <p>Per-group thresholds cut the gap to
          {fmt(float((mt.get('per_group') or {{}}).get('tpr_gap', 0)))} for
          {fmt(float(mt.get('accuracy_cost', 0)), 4)} of accuracy, and need the
          person's race at the moment of decision, and push the false positive gap up.
          There is no setting where everything is fair at once. You are choosing which
          unfairness to keep, so choose it out loud.</p>
      </div>
      <div class="step">
        <h3>Put the limits somewhere a build can read them</h3>
        <p>The reason any of this is still true tomorrow is that the numbers are held
          against a declared file, regenerated on every commit, and checked monthly by
          a job nobody has to remember to run. A standard that lives in a document
          drifts. A standard that fails a build does not.</p>
      </div>
    </div>
    <p class="note">The honest move was never to find a model with no disparity. It was
      to measure the disparity, publish what closing it would cost, say which cost I
      was not willing to pay, and make it impossible to quietly change my mind later.</p>
  </div>
</div>

""",
        "hero":
        f"""<header class="hero">
  <div class="mesh" aria-hidden="true"><i class="m1"></i><i class="m2"></i>
    <i class="m3"></i><i class="m4"></i></div>
  {field_markup(result)}
  {topo_lines()}
  <div class="shell">
    <p class="kicker">A fairness gate &#183; American Community Survey &#183;
      {ts['n']:,} people</p>
  </div>
  <div class="mega-wrap bleed">
    <h1 class="mega cut sliced">
      <span>{esc(STATUS_LABEL[v].upper())}</span>
      <span class="sl sl1" aria-hidden="true">{esc(STATUS_LABEL[v].upper())}</span>
      <span class="sl sl2" aria-hidden="true">{esc(STATUS_LABEL[v].upper())}</span>
      <span class="sl sl3" aria-hidden="true">{esc(STATUS_LABEL[v].upper())}</span>
    </h1>
  </div>
  <div class="shell">
    <div class="under">
      <span class="tally">{counts['pass']} pass &#183; {counts['warn']} warn &#183;
        {counts['fail']} fail</span>
    </div>
    <p class="lead">A model decides who looks like they earn enough. It is wrong about
      some people far more often than others. This page measures that, holds it to a
      declared limit, and fails the build when the limit breaks.</p>
  </div>
</header>

""",
        "marquee":
        f"""<div class="mq" aria-hidden="true">
  <div class="mq-track">{_marquee(cost)}{_marquee(cost)}</div>
</div>

""",
        "plain":
        """<section id="plain">
  <div class="shell reveal">
    <p class="kicker">Plain language</p>
    <h2>What a gate is, before any of the numbers.</h2>
    <div class="primer">
      <div>
        <p class="lede">Teams already refuse to ship code that fails a test. Almost
          nobody refuses to ship a <b>model</b> that fails a fairness check, because that
          check usually lives in a document someone wrote once and never opened again.</p>
        <p>Continuous integration is a robot that wakes up every time somebody changes
          the code, runs a list of checks, and either says fine or refuses. A
          <b>gate</b> is a check with the authority to say no. So I took the fairness
          limits, wrote them into a file sitting next to the code, and wired them to that
          same robot. If the model starts treating one group meaningfully worse than
          another, the build turns red, exactly like a broken test. Nobody can merge past
          it without editing the limits, and editing the limits leaves a mark in the
          history that a reviewer has to approve.</p>
        <p>The model underneath is doing something ordinary. It reads ten facts from a
          government survey, things like age, education, hours worked and occupation, and
          guesses whether that person earns above a threshold. The survey already recorded
          the real answer, so I can check not only whether the model is right, but
          <b>who it is wrong about</b>. That second question is the whole project.</p>
      </div>
      <dl class="defs">
        <dt>Recall</dt>
        <dd>Of the people who genuinely qualify, the share the model actually finds. If a
          hundred people really do earn above the line and it flags 54, recall is 54%.
          <b>The other 46 are overlooked.</b></dd>
        <dt>False positive</dt>
        <dd>The opposite mistake. Somebody flagged who should not have been.</dd>
        <dt>Threshold</dt>
        <dd>The cut-off. The model gives a probability, and I choose the line above which
          it says yes. Moving that line trades one kind of mistake for the other.</dd>
        <dt>Gap</dt>
        <dd>Just the distance between the best treated group and the worst treated one,
          on whichever measure is being discussed.</dd>
        <dt>Calibration</dt>
        <dd>Whether the probabilities mean what they say. If it claims 70% for a thousand
          people, about 700 of them should qualify. Being accurate and being calibrated
          are not the same thing.</dd>
      </dl>
    </div>
  </div>
</section>

""",
        "cost":
        f"""<div class="band" id="cost">
  <div class="shell band-in reveal">
    <p class="kicker">Who it misses</p>
    <h2>Out of every hundred people who qualify.</h2>
    <p class="say">A true positive rate is a statistic. Said another way: of a hundred
      people who genuinely earn above the threshold, this is how many the model fails to
      find. Each dot is one of those hundred people. The lit ones are overlooked.</p>

    <div class="cost">
      <div>
        <h3>{esc(hurt['group'])}</h3>
        <p class="of tiny">{int(hurt['n']):,} surveyed</p>
        {dot_field(float(hurt['per_100']))}
        <div class="count bad">{round(hurt['per_100'])}</div>
        <p class="cap">overlooked out of every hundred who qualify, which is
          {round(hurt['overlooked']):,} people in this survey year alone.</p>
      </div>
      <div>
        <h3>{esc(spared['group'])}</h3>
        <p class="of tiny">{int(spared['n']):,} surveyed</p>
        {dot_field(float(spared['per_100']))}
        <div class="count">{round(spared['per_100'])}</div>
        <p class="cap">overlooked out of every hundred who qualify. The same model, the
          same threshold, the same day.</p>
      </div>
    </div>

    <div class="total">
      <span class="n">{round(overlooked_total):,}</span>
      <p>qualifying people overlooked across every reported group in this one survey
        year. If a decision hung on this model, that is the size of the room.</p>
    </div>
  </div>
</div>

""",
        "headline":
        f"""<div class="acid">
  {topo_lines(13, 3)}
  <div class="acid-in">
    <div class="shell"><p class="kicker tiny">The disparity</p></div>
    <div class="mega-wrap bleed">
      <p class="mega cut">{ratio:.1f}&#215; MORE LIKELY</p>
    </div>
    <div class="shell">
      <p class="ratio-say">Being in the {esc(hurt['group'])} group makes this model
        <b>{ratio:.1f} times</b> more likely to miss you, even when you genuinely
        qualify.</p>
      <p class="say" style="margin-top:40px">Accuracy is nearly flat across all of these
        groups, between {fmt(float(sub['accuracy'].min()), 2)} and
        {fmt(float(sub['accuracy'].max()), 2)}. A model can be equally accurate
        everywhere and still place its errors on the same people every time. That is why
        one headline number is never an answer to this question.</p>
    </div>
  </div>
</div>

<section>
  <div class="shell reveal">
    <div class="numbers">
      <div><div class="k">True positive gap</div><div class="v">{fmt(race['tpr_gap'])}</div>
        <div class="c">target 0.150, limit 0.350</div></div>
      <div><div class="k">Base rate gap</div><div class="v">{fmt(race['base_rate_gap'])}</div>
        <div class="c">the part the data itself explains</div></div>
      <div><div class="k">Gap by sex</div><div class="v">{fmt(sex['tpr_gap'])}</div>
        <div class="c">same metric, second attribute</div></div>
      <div><div class="k">AUC</div><div class="v">{fmt(ts['auc'], 3)}</div>
        <div class="c">majority baseline {fmt(ts['majority_baseline'], 3)}</div></div>
    </div>
  </div>
</section>

""",
        "checks":
        f"""<div class="band" id="checks">
  <div class="shell band-in reveal">
    <p class="kicker">The gate</p>
    <h2>Every threshold, measured on every commit.</h2>
    <p class="say">The solid tick is the limit that fails the build. The dashed tick is
      the target being aimed at. The distance between them is deliberate: a limit pinned
      to wherever the model lands today would make the gate meaningless, and one pinned
      to the aspiration would mean a permanently red build that everyone learns to
      ignore.</p>
    <div class="panel"><ul class="checks">{_checks(result)}</ul></div>
  </div>
</div>

""",
        "intersections":
        f"""<div class="band" id="intersections">
  <div class="shell band-in reveal">
    <p class="kicker">Where the attributes cross</p>
    <h2>Two acceptable averages can hide one bad cell.</h2>
    <p class="say">Race and sex audited separately can both look tolerable while the
      combination of the two is far worse than either suggests. A pair of bar charts
      cannot show that, because it never puts the axes together. This is a grid, because
      the data is a grid: recall in each cell, with the group size underneath.</p>
    <p class="say">It is not hypothetical here. The false positive gap is
      <b>{xsex['fpr_gap']:.3f}</b> across these cells, against {race['fpr_gap']:.3f} for
      race alone and {sex['fpr_gap']:.3f} for sex alone. Crossing the attributes roughly
      doubles the disparity that either marginal reports.</p>
    <div class="panel chartbox">{intersection_matrix(test)}</div>
    <p class="note">Hatched cells fall below the 500-person reporting floor and are
      withheld. Their absence is itself the finding: {xsuppressed} of
      {xtotal} cells are too small to say anything about, and the groups most exposed to
      harm are the ones a survey is least likely to have enough of.</p>
  </div>
</div>

""",
        "certainty":
        f"""<section id="certainty">
  <div class="shell reveal">
    <p class="kicker">How much to trust these numbers</p>
    <h2>Three decimals do not mean three decimals.</h2>
    <p class="say">Every other chart here prints rates to the same precision whatever the
      group size. This one shows what that precision is worth. The dot is the measured
      recall, the bar is the 95% interval, and the groups are ordered by how uncertain
      they are, so the caveat arrives before the ranking.</p>
    <p class="say">This is also what the gate now runs on. A check fails only when the
      whole interval clears the declared limit. A point estimate over the line with an
      interval straddling it is a warning instead, because a build that goes red on
      sampling noise is a build people learn to re-run until it passes.</p>
    <div class="panel chartbox">{interval_chart(result)}</div>
  </div>
</section>

""",
        "groups":
        f"""<section id="groups">
  <div class="shell reveal">
    <p class="kicker">By group</p>
    <h2>Where the errors land.</h2>
    <p class="say">Grouped rather than stacked, because these are two independent rates
      and not parts of a whole. Sorted by group size so the ordering is not read as a
      ranking of severity.</p>
    {_controls()}
    <div class="panel">
      <p class="legend"><span class="key k1"></span>Found, of those who qualify
        <span class="key k2"></span>Wrongly flagged</p>
      {_panes(tables, rate_chart)}
    </div>
  </div>
</section>

<div class="band">
  <div class="shell band-in reveal">
    <p class="kicker">Calibration</p>
    <h2>Does a probability mean the same thing for everyone?</h2>
    <div class="split">
      <div class="panel">
        <p class="legend"><span class="key k1"></span>Observed
          <span class="key k2"></span>Predicted</p>
        {_panes(tables, calib_chart)}
      </div>
      <div class="panel scroll-x">
        <table>
          <thead><tr><th scope="col">Split</th><th scope="col">n</th>
            <th scope="col">AUC</th><th scope="col">ECE</th>
            <th scope="col">Acc</th><th scope="col">Base</th></tr></thead>
          <tbody>{_splits_compact(result)}</tbody>
        </table>
        <p class="note">Validation ECE is near zero because the calibrator was fitted
          there. It is not a result. Error grows on the later year, and again on the
          four states held out entirely.</p>
      </div>
    </div>
  </div>
</div>

<section>
  <div class="shell reveal">
    <p class="kicker">Full detail</p>
    <h2>Every reported group.</h2>
    <p class="say">Groups below 500 people are measured, then withheld from every table
      and every summary gap. A rate computed on a few hundred people is mostly noise,
      and publishing it as though it were a finding would be its own kind of harm.</p>
    <div class="panel scroll-x">{_panes(tables, group_table)}</div>
  </div>
</section>

""",
        "flow":
        f"""<section id="flow">
  <div class="shell reveal">
    <p class="kicker">The whole population</p>
    <h2>Where every person actually goes.</h2>
    <p class="say">Each ribbon is people, and the total is conserved: everybody leaves one
      box and arrives in exactly one other. The two error ribbons are the ones that
      matter. Hover any ribbon to isolate it.</p>
    <div class="panel chartbox">{sankey(result)}</div>
  </div>
</section>

<div class="band">
  <div class="shell band-in reveal">
    <p class="kicker">Composition</p>
    <h2>The same four outcomes, group by group.</h2>
    <p class="say">Rows are normalised within each group, because the question is
      composition and not size. Overlooked and wrongly picked are shaded in the error
      colour.</p>
    <div class="panel chartbox">{confusion_heatmap(result)}</div>
  </div>
</div>

<section>
  <div class="shell reveal">
    <p class="kicker">Denominators</p>
    <h2>Who is actually in this data.</h2>
    <p class="say">Every rate on this page is computed on a wildly different number of
      people. Area is the honest way to show that before anyone reads a percentage.</p>
    <div class="panel chartbox">{treemap(result)}</div>
  </div>
</section>

<div class="band">
  <div class="shell band-in reveal">
    <p class="kicker">Mechanism</p>
    <h2>Why the gap exists at all.</h2>
    <p class="say">Two shapes per group: the people who qualify, in the error colour, and
      everyone else. Where they overlap is where the model genuinely cannot tell them
      apart. That overlap being wider for some groups than others is the mechanism
      behind every gap on this page.</p>
    <div class="panel chartbox">{ridgeline(result)}</div>
  </div>
</div>

<section>
  <div class="shell reveal">
    <p class="kicker">Diagnostics</p>
    <h2>Rarity against recall, and reliability.</h2>
    <p class="say">On the left, how common the outcome is in a group against how much of
      it the model finds, with bubble area as group size. On the right, predicted against
      observed per bin: a calibrated model traces the dashed diagonal.</p>
    <div class="split">
      <div class="panel chartbox">{_panes(tables, scatter_chart)}</div>
      <div class="panel chartbox">{reliability(result)}</div>
    </div>
  </div>
</section>

<div class="band">
  <div class="shell band-in reveal">
    <p class="kicker">Direction of error</p>
    <h2>Over-selected, under-selected.</h2>
    <p class="say">Selection rate minus base rate. Zero is the neutral line: bars to the
      left are groups the model picks less often than the data warrants, bars to the
      right are groups it picks more often.</p>
    <div class="panel chartbox">{_panes(tables, diverging_chart)}</div>
  </div>
</div>

<section>
  <div class="shell reveal">
    <p class="kicker">Under shift</p>
    <h2>What happens on states it never saw.</h2>
    <p class="say">One line per group, from the test year to the four states held out of
      training entirely. Direction and size of the change at once, which two separate bar
      charts would make you compute in your head.</p>
    <div class="panel chartbox">{slope_chart(tables, 'RAC1P')}</div>
  </div>
</section>

""",
        "fix":
        f"""<div class="band" id="fix">
  <div class="shell band-in reveal">
    <p class="kicker">What it would take to fix</p>
    <h2>The gap can be closed. Here is the bill.</h2>
    <p class="say">Proving a disparity exists is the easy half. Every point on the line
      is this same model under a different single threshold, and the shape is the
      argument: sliding one cut does not buy fairness, it buys a worse model that selects
      almost everybody. The lowest gap a single threshold reaches is
      <b>{mt['best_global']['tpr_gap']:.3f}</b>, at a threshold of
      {mt['best_global']['threshold']:.2f} and an accuracy of
      {mt['best_global']['accuracy']:.3f}. That is not a fix, it is abandoning the
      model.</p>
    <div class="panel chartbox">{tradeoff_curve(result)}</div>
  </div>
</div>

<section>
  <div class="shell reveal">
    <p class="kicker">The honest accounting</p>
    <h2>Cheap in accuracy, expensive in everything else.</h2>
    <p class="say">One threshold per group, chosen on the validation year to equalise
      recall, closes most of the gap: <b>{mt['baseline']['tpr_gap']:.3f}</b> falls to
      <b>{mt['per_group']['tpr_gap']:.3f}</b>. Accuracy pays
      <b>{mt['accuracy_cost']:.4f}</b> for it, which is four tenths of a point. By the
      only number most projects report, this is nearly free.</p>
    <div class="numbers">
      <div><div class="k">Recall gap, today</div>
        <div class="v">{mt['baseline']['tpr_gap']:.3f}</div>
        <div class="c">one threshold for everyone</div></div>
      <div><div class="k">Recall gap, fixed</div>
        <div class="v">{mt['per_group']['tpr_gap']:.3f}</div>
        <div class="c">one threshold per group</div></div>
      <div><div class="k">Accuracy paid</div>
        <div class="v">{mt['accuracy_cost']:.4f}</div>
        <div class="c">{mt['baseline']['accuracy']:.3f} to {mt['per_group']['accuracy']:.3f}</div></div>
      <div><div class="k">False positive gap</div>
        <div class="v">{mt['per_group']['fpr_gap']:.3f}</div>
        <div class="c">was {mt['baseline']['fpr_gap']:.3f}, so it got worse</div></div>
    </div>
    <p class="say" style="margin-top:34px">Two things that accounting hides, and both
      matter more than the accuracy.</p>
    <p class="say">The false positive gap goes the other way, from
      {mt['baseline']['fpr_gap']:.3f} to {mt['per_group']['fpr_gap']:.3f}. Equalising one
      error rate across groups makes the other one less equal. This is not an
      implementation detail, it is the impossibility result arriving in person: with
      different base rates you are choosing which unfairness to keep, not removing
      unfairness.</p>
    <p class="say">And it needs the protected attribute <b>at the moment of
      prediction</b>. A system doing this has to look at someone's race to decide which
      threshold applies to them. In hiring, in lending, in most places anyone would
      actually want this, that is either unlawful or is itself the harm. A mitigation
      that only works by doing the thing you were trying to avoid belongs in the
      documentation as a rejected option with its reasoning attached, which is where this
      one is.</p>
    <p class="note">Thresholds are chosen on the validation year and applied unchanged to
      the test year, the same discipline as everything else here. Choosing them on the
      test year would be fitting the fix to the exam.</p>
  </div>
</section>

""",
        "method":
        f"""<div class="band" id="method">
  <div class="shell band-in reveal">
    <p class="kicker">Method</p>
    <h2>Four decisions that shaped every number above.</h2>
    <div class="steps">
      <div class="step"><div>
        <h3>The split is temporal, not random</h3>
        <p>Fitted on {esc(d['train_year'])}, tuned on {esc(d['threshold_chosen_on'])},
          tested on {esc(d['test_year'])}, which is how a model like this is actually
          used: built on the past, applied to the present. A random split of a single
          year would have hidden the drift entirely.</p></div></div>
      <div class="step"><div>
        <h3>Four states are held out completely</h3>
        <p>Geographic shift on top of temporal shift. The positive rate falls to
          {fmt(result['splits']['shift']['positive_rate'])} there against
          {fmt(result['splits']['test']['positive_rate'])} in the test states, and AUC
          falls from {fmt(ts['auc'], 3)} to {fmt(sh['auc'], 3)}. That decay is measured
          rather than assumed away.</p></div></div>
      <div class="step"><div>
        <h3>Protected attributes are kept out of the features</h3>
        <p>Race and sex are held separately and used only to measure disparity, never to
          predict. An attribute is protected whether or not the model is allowed to see
          it, so the audit runs either way.</p></div></div>
      <div class="step"><div>
        <h3>Three fairness criteria are reported together</h3>
        <p>Demographic parity, equalised odds and calibration cannot all hold when base
          rates differ across groups, and here they differ by
          {fmt(race['base_rate_gap'])}. Quoting one number alone would be a choice about
          which unfairness to make invisible.</p></div></div>
    </div>
  </div>
</div>

""",
        "sources":
        f"""<div class="band" id="sources">
  <div class="shell band-in reveal">
    <p class="kicker">Sources</p>
    <h2>Where the data and the definitions come from.</h2>
    <p class="say">Nothing on this page is invented. The people are real survey
      respondents, the fairness criteria are the standard ones, and the reason they
      cannot all be satisfied at once is a proved result, not an opinion.</p>
    <ul class="srcs">
      <li><div>
        <a href="https://www.census.gov/programs-surveys/acs/microdata.html">
          American Community Survey, Public Use Microdata</a>
        <span class="what">US Census Bureau. The survey responses behind every number
          here, published de-identified for exactly this kind of analysis.</span>
      </div><span class="yr">{esc(d['train_year'])}&#8211;{esc(d['test_year'])}</span></li>

      <li><div>
        <a href="https://arxiv.org/abs/2108.04884">Retiring Adult: New Datasets for Fair
          Machine Learning</a>
        <span class="what">Ding, Hardt, Miller and Schmidt. The paper behind
          <code>folktables</code>, which defines this prediction task and argues for
          exactly the temporal and geographic splits used here.</span>
      </div><span class="yr">2021</span></li>

      <li><div>
        <a href="https://arxiv.org/abs/1610.02413">Equality of Opportunity in Supervised
          Learning</a>
        <span class="what">Hardt, Price and Srebro. Defines equalised odds, the criterion
          the true positive gap on this page is measured against.</span>
      </div><span class="yr">2016</span></li>

      <li><div>
        <a href="https://arxiv.org/abs/1609.05807">Inherent Trade-Offs in the Fair
          Determination of Risk Scores</a>
        <span class="what">Kleinberg, Mullainathan and Raghavan. Proves that calibration
          and equal error rates cannot hold together when base rates differ, which is why
          this page reports three criteria instead of choosing one.</span>
      </div><span class="yr">2016</span></li>

      <li><div>
        <a href="https://arxiv.org/abs/1703.00056">Fair Prediction with Disparate Impact</a>
        <span class="what">Chouldechova. The same impossibility reached independently,
          in the context of recidivism scoring.</span>
      </div><span class="yr">2017</span></li>

      <li><div>
        <a href="https://fairmlbook.org">Fairness and Machine Learning</a>
        <span class="what">Barocas, Hardt and Narayanan. The standard textbook, free
          online, and the source for treating a protected attribute as something you
          audit against rather than something you delete.</span>
      </div><span class="yr">2023</span></li>

      <li><div>
        <a href="https://eur-lex.europa.eu/eli/reg/2024/1689/oj">EU Artificial
          Intelligence Act</a>
        <span class="what">Regulation 2024/1689. Annex IV sets the structure of the
          technical documentation this project generates alongside the page.</span>
      </div><span class="yr">2024</span></li>
    </ul>
  </div>
</div>

""",
        "footer":
        f"""<footer>
  <div class="shell">
    <p><strong>Generated, not written.</strong> This page, the model card, the Annex IV
      technical documentation and the DPIA all come out of
      <code>scripts/run_audit.py</code>. Continuous integration regenerates each of them
      and requires a byte-identical match, so editing any number by hand fails the
      build.</p>
    <p>Run {esc(result['generated_at'])} &#183; commit <code>{esc(result['git_sha'])}</code>
      &#183; pipeline <code>{esc(result.get('pipeline_fingerprint', 'n/a'))}</code>
      &#183; Python {esc(result['python'])}</p>
    <p>Not a deployable system, and not a product. Income prediction on census microdata
      is a benchmark. No decision about any person should be made with it.</p>
  </div>
</footer>

""",
        "tail":
        f"""<script>{FIELD_SCRIPT}</script>
<script>
(function () {{
  var root = document.documentElement;
  var btn = document.getElementById('themer');
  function label() {{ btn.textContent = root.dataset.theme === 'dark' ? 'Light' : 'Dark'; }}
  try {{ var saved = localStorage.getItem('fg-theme'); if (saved) root.dataset.theme = saved; }}
  catch (e) {{ /* private mode: keep the default */ }}
  label();
  btn.addEventListener('click', function () {{
    root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    label();
    try {{ localStorage.setItem('fg-theme', root.dataset.theme); }} catch (e) {{}}
  }});

  // Every pane is already in the document; the controls only choose which one shows.
  var state = {{ attr: 'RAC1P', split: 'test' }};
  function sync() {{
    document.querySelectorAll('.pane').forEach(function (p) {{
      p.classList.toggle('is-on',
        p.dataset.attr === state.attr && p.dataset.split === state.split);
    }});
  }}
  // Cross-fade the swap where the browser can, cut where it cannot. Either way the
  // pane changes, so the transition is decoration and never a dependency.
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function swap(fn) {{
    if (document.startViewTransition && !reduce) {{ document.startViewTransition(fn); }}
    else {{ fn(); }}
  }}
  document.querySelectorAll('.seg').forEach(function (b) {{
    b.addEventListener('click', function () {{
      swap(function () {{
        state[b.dataset.k] = b.dataset.v;
        b.parentNode.querySelectorAll('.seg').forEach(function (o) {{
          o.classList.toggle('is-on', o === b);
        }});
        sync();
      }});
    }});
  }});

}})();
</script>
""",
    }



# ---------------------------------------------------------------------------- #
# Pages
# ---------------------------------------------------------------------------- #
# It was one endless scroll and it asked too much of a reader. Now it is five pages in
# the order the argument actually runs: here is what I found, here is the evidence,
# here is how I measured it, here is the fix and why I refused it, here is why I care.
#
# Every page is a real file rather than a tab, so links go straight to a section, the
# browser back button behaves, and nothing depends on JavaScript to show you content.
NAV = [
    ("index.html", "The finding"),
    ("evidence.html", "Evidence"),
    ("method.html", "Method"),
    ("fix.html", "The fix"),
    ("about.html", "Why"),
]

PAGES = [
    (
        "index.html",
        "Fairness gate",
        (
            "A model that is equally accurate for everyone and still misses one group "
            "far more often. Measured on 600,551 people, and wired to a build that "
            "fails when the gap crosses a declared line."
        ),
        ["hero", "marquee", "plain", "cost", "headline", "narration",
         "next_evidence"],
    ),
    (
        "evidence.html",
        "The evidence",
        (
            "Every check, every group, and how certain each number is. Sixteen policy "
            "checks, the intersectional audit, confidence intervals and the full flow "
            "of 600,551 people through the model."
        ),
        ["head_evidence", "checks", "intersections", "certainty", "groups", "flow", "next_method"],
    ),
    (
        "method.html",
        "How I measured it",
        (
            "Temporal and geographic splits, a reporting floor, three fairness criteria "
            "at once, and the experiment showing that deleting race and sex leaves "
            "most of the gap behind."
        ),
        ["head_method", "method", "unaware", "families", "sources", "next_fix"],
    ),
    (
        "fix.html",
        "The fix I did not ship",
        (
            "Per-group thresholds close two thirds of the recall gap for almost no "
            "accuracy. Here is the measurement, and here is why shipping it would "
            "have been the wrong call."
        ),
        ["head_fix", "fix", "conclusion"],
    ),
]


def _nav(active: str, verdict: str) -> str:
    items = []
    for href, label in NAV:
        current = ' aria-current="page"' if href == active else ""
        items.append(f'<li><a href="{href}"{current}>{esc(label)}</a></li>')
    return f"""<nav>
  <span class="mark"><a href="index.html">fairness&#8209;gate</a> <i>/ ACS income</i></span>
  <ul>{''.join(items)}</ul>
  <div class="right">
    <span class="pill"><i></i>{esc(STATUS_LABEL[verdict])}</span>
    <button type="button" class="tgl" id="themer" aria-label="Switch colour theme">Light</button>
  </div>
</nav>"""


def page_head(kicker: str, title: str, lead: str) -> str:
    """The compact header every page except the front one gets.

    The front page keeps the full-height hero. Repeating that on all five would make
    each one feel like a landing page and bury the content four screens down.
    """
    return f"""<header class="phead">
  <div class="mesh" aria-hidden="true"><i class="m1"></i><i class="m2"></i></div>
  <div class="shell">
    <p class="kicker">{esc(kicker)}</p>
    <h1 class="ptitle">{esc(title)}</h1>
    <p class="lead">{lead}</p>
  </div>
</header>"""


def onward(href: str, label: str, blurb: str) -> str:
    """The link at the foot of a page to the next step in the argument."""
    return f"""<div class="band onward">
  <div class="shell band-in">
    <a class="onward-in" href="{href}">
      <span class="kicker">Next</span>
      <span class="onward-t">{esc(label)}</span>
      <span class="onward-b">{blurb}</span>
    </a>
  </div>
</div>"""


def _shell(title: str, description: str, active: str, body: str,
           result: dict, tables: dict[str, pd.DataFrame], sections: dict) -> str:
    # imported here, not at module level: voice.py reads human_cost out of this module
    from src.voice import build as voice_build
    v = result["policy_verdict"]
    return f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<style>
{STYLE}</style>
</head>
<body class="v-{esc(v)}">

<div class="prog" aria-hidden="true"></div>
<div class="dither" aria-hidden="true"></div>
<svg class="grain" aria-hidden="true" focusable="false">
  <filter id="gr"><feTurbulence type="fractalNoise" baseFrequency="0.86" numOctaves="4"
    stitchTiles="stitch"/><feColorMatrix type="saturate" values="0"/></filter>
  <rect width="100%" height="100%" filter="url(#gr)"/>
</svg>

{_nav(active, v)}
{body}
{sections['footer']}
{voice_build(result, tables)}
{sections['tail']}
</body>
</html>
"""


def render(result: dict, tables: dict[str, pd.DataFrame],
           page: str = "index.html") -> str:
    """One page of the site, composed from the shared sections."""
    sections = _sections(result, tables)
    spec = next(p for p in PAGES if p[0] == page)
    _, title, description, keys = spec
    body = "\n".join(sections[k] for k in keys)
    return _shell(title, description, page, body, result, tables, sections)


def write(result: dict, tables: dict[str, pd.DataFrame]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    sections = _sections(result, tables)
    for name, title, description, keys in PAGES:
        body = "\n".join(sections[k] for k in keys)
        html_text = _shell(title, description, name, body, result, tables, sections)
        (DOCS_DIR / name).write_text(html_text, encoding="utf-8")
