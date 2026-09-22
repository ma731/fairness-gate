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
    reliability,
    ridgeline,
    sankey,
    treemap,
)

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
def bullet(check: dict, width: int = 720) -> str:
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

    return (
        f'<svg class="bullet" viewBox="0 0 {width} 20" width="100%" height="20" '
        f'preserveAspectRatio="none" role="img" aria-label="measured {fmt(measured, 4)}">'
        f'<rect class="trk" x="0" y="6" width="{width}" height="8" rx="4"/>'
        f'<rect class="fil s-{check["status"]}" x="0" y="6" width="{x(measured)}" '
        f'height="8" rx="4"><title>measured {fmt(measured, 4)}</title></rect>'
        f"{ticks}</svg>"
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

    row_h, bar_h, gap, lab_w, pad_r, plot_w = 54, 15, 5, 280, 62, 520
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
            f'{esc(r["group"])}</text>'
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

    row_h, lab_w, pad_r, plot_w = 36, 280, 62, 520
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
            f'{esc(r["group"])}</text>'
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
        f'<span>{esc(r["group"])} <em>{round(r['per_100'])}</em></span>'
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
    pad_l, pad_b, pad_t, pad_r = 66, 54, 22, 150
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
    marks = ""
    for _, r in sub.sort_values("n", ascending=False).iterrows():
        cx, cy = X(float(r["base_rate"])), Y(float(r["tpr"]))
        rad = round(7 + 26 * (float(r["n"]) / biggest) ** 0.5, 1)
        marks += (
            f'<g class="row"><circle class="bub" cx="{cx}" cy="{cy}" r="{rad}">'
            f'<title>{esc(r["group"])}: {int(r["n"]):,} people, base rate '
            f'{fmt(r["base_rate"])}, found {pct(r["tpr"])}</title></circle>'
            f'<text class="vl" x="{round(cx + rad + 9, 1)}" y="{cy + 4}">'
            f'{esc(r["group"])}</text></g>'
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

    row_h, lab_w, half, pad_r = 42, 290, 220, 80
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
            f'{esc(r["group"])}</text>'
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
    a = a[(a["attribute"] == attribute) & a["reportable"]][["group", "tpr"]]
    b = b[(b["attribute"] == attribute) & b["reportable"]][["group", "tpr"]]
    m = a.merge(b, on="group", suffixes=("_test", "_shift"))
    if m.empty:
        return '<p class="empty">No group reportable in both populations.</p>'
    m = m.sort_values("tpr_test", ascending=False).reset_index(drop=True)

    w, h = 780, 430
    left, right = 270, 500
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
            f'{esc(r["group"])}</text>'
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


def _checks(result: dict) -> str:
    order = {"fail": 0, "warn": 1, "pass": 2}
    out = ""
    for c in sorted(result["checks"], key=lambda c: (order[c["status"]], c["name"])):
        target = "n/a" if c["warn_at"] is None else fmt(c["warn_at"])
        limit = "n/a" if c["fail_at"] is None else fmt(c["fail_at"])
        name = c["name"].replace("fairness.", "").replace("performance.", "")
        out += (
            f'<li class="chk s-{c["status"]}">'
            f'<div class="chk-top"><code>{esc(name)}</code>'
            f'<span class="tag">{STATUS_LABEL[c["status"]]}</span></div>'
            f"{bullet(c)}"
            f'<div class="chk-bot"><span class="big">{fmt(c["measured"], 4)}</span>'
            f'<span class="mut">target {target}</span>'
            f'<span class="mut">limit {limit}</span></div></li>'
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


def render(result: dict, tables: dict[str, pd.DataFrame]) -> str:
    test = tables["test"]
    race = next(s for s in result["fairness"]["test"] if s["attribute"] == "RAC1P")
    sex = next(s for s in result["fairness"]["test"] if s["attribute"] == "SEX")
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

    return f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fairness gate</title>
<meta name="description" content="A CI gate for model fairness. Declared thresholds, a measured audit, and a build that fails when a disparity crosses the line.">
<style>
:root, :root[data-theme="dark"] {{
  color-scheme: dark;
  --plane:#070809; --surface:#0f1113; --raised:#15181b;
  --hair:rgba(255,255,255,.07); --hair-2:rgba(255,255,255,.15);
  --ink:#f6f8f9; --ink-2:#b4bbc1; --mut:#7d848a;
  --grid:#1d2125; --axis:#2a2f34;
  --s1:#3987e5; --s2:#d95926;
  --good:#0ca30c; --warn:#fab219; --crit:#d03b3b;
  --glow:rgba(57,135,229,.15);
  /* Narrative accent. Used only in the story sections, never in the gate, where a
     colour has to mean pass, warn or fail and nothing else. */
  --acid:#ccff00; --acid-ink:#0a0b00;
  --grain:.055;
}}
:root[data-theme="light"] {{
  color-scheme: light;
  --plane:#f2f2f0; --surface:#fcfcfb; --raised:#ffffff;
  --hair:rgba(11,11,11,.09); --hair-2:rgba(11,11,11,.18);
  --ink:#0b0b0b; --ink-2:#52514e; --mut:#898781;
  --grid:#e6e5df; --axis:#c3c2b7;
  --s1:#2a78d6; --s2:#eb6834;
  --good:#0ca30c; --warn:#fab219; --crit:#d03b3b;
  --glow:rgba(42,120,214,.10);
  --acid:#5c7400; --acid-ink:#ffffff;
  --grain:.03;
}}
*,*::before,*::after {{ box-sizing:border-box; }}
html {{ -webkit-text-size-adjust:100%; scroll-behavior:smooth; scroll-padding-top:80px; }}
/* 100vw bleeds are wider than the content box once a vertical scrollbar exists. */
html, body {{ overflow-x:clip; max-width:100%; }}
body {{
  margin:0; background:var(--plane); color:var(--ink);
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,sans-serif;
  font-size:17px; line-height:1.6; letter-spacing:-.008em;
  font-feature-settings:"kern" 1; text-rendering:optimizeLegibility;
}}
.mono, code, table, .num {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums lining-nums; }}
.shell {{ max-width:1240px; margin:0 auto; padding:0 32px; }}
.mut {{ color:var(--mut); }}
h2 {{ font-size:clamp(32px,3.9vw,52px); line-height:1.02; letter-spacing:-.04em;
  font-weight:900; margin:0 0 18px; max-width:18ch; }}
.kicker {{ font-size:11px; letter-spacing:.2em; text-transform:uppercase;
  color:var(--mut); margin:0 0 20px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
.say {{ max-width:58ch; font-size:18px; line-height:1.6; color:var(--ink-2);
  margin:0 0 32px; }}
.say b {{ color:var(--ink); font-weight:600; }}

/* nav */
nav {{ position:sticky; top:0; z-index:20; height:64px; display:flex; align-items:center;
  gap:30px; padding:0 32px; border-bottom:1px solid var(--hair);
  background:color-mix(in srgb, var(--plane) 82%, transparent);
  backdrop-filter:blur(16px) saturate(140%);
  -webkit-backdrop-filter:blur(16px) saturate(140%); }}
nav .mark {{ font-size:14px; font-weight:660; letter-spacing:-.015em; }}
nav .mark i {{ font-style:normal; color:var(--mut); font-weight:400; }}
nav ul {{ display:flex; gap:26px; list-style:none; margin:0; padding:0; }}
nav a {{ color:var(--ink-2); text-decoration:none; font-size:12px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  letter-spacing:.06em; text-transform:uppercase;
  transition:color 150ms ease; }}
nav a:hover {{ color:var(--ink); }}
nav .right {{ margin-left:auto; display:flex; align-items:center; gap:14px; }}
.pill {{ display:inline-flex; align-items:center; gap:7px; font-size:12px;
  padding:5px 12px; border-radius:999px; border:1px solid var(--hair-2);
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
.pill i {{ width:7px; height:7px; border-radius:50%; background:var(--vc); font-style:normal; }}
.tgl {{ appearance:none; border:1px solid var(--hair-2); background:transparent;
  color:var(--ink-2); border-radius:8px; padding:6px 12px; font:inherit; font-size:12.5px;
  cursor:pointer; transition:transform 130ms cubic-bezier(.23,1,.32,1),
    color 150ms ease, border-color 150ms ease; }}
.tgl:hover {{ color:var(--ink); border-color:var(--ink-2); }}
.tgl:active {{ transform:scale(.96); }}
@media (max-width:860px) {{ nav ul {{ display:none; }} }}

/* hero */
.hero {{ min-height:calc(100dvh - 64px); display:flex; flex-direction:column;
  justify-content:center; padding:72px 0 64px; position:relative; overflow:hidden; }}
.hero::before {{ content:""; position:absolute; inset:-20% -10% auto auto;
  width:min(78vw,900px); aspect-ratio:1; pointer-events:none; z-index:0;
  background:radial-gradient(circle at 62% 34%,
    color-mix(in srgb, var(--vc) 15%, transparent), transparent 62%); }}
.hero > * {{ position:relative; z-index:1; }}
.v-pass {{ --vc:var(--good); }} .v-warn {{ --vc:var(--warn); }} .v-fail {{ --vc:var(--crit); }}
.hero h1 {{ font-size:clamp(76px,16vw,232px); line-height:.8; letter-spacing:-.07em;
  font-weight:900; margin:0; color:var(--vc); }}
.hero .under {{ display:flex; align-items:baseline; gap:20px; flex-wrap:wrap;
  margin-top:26px; }}
.hero .lead {{ max-width:54ch; font-size:clamp(18px,2vw,23px); line-height:1.42;
  color:var(--ink-2); letter-spacing:-.015em; margin:30px 0 0; }}
.hero .lead b {{ color:var(--ink); font-weight:600; }}
.tally {{ font-size:14px; color:var(--ink-2);
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}

/* the finding: a full-bleed contrast band */
.band {{ background:var(--surface); border-block:1px solid var(--hair); margin-top:0;
  position:relative; }}
/* corner ticks: drafting marks, not decoration for its own sake. They mark where the
   measured content starts and stops. */
.band::before, .band::after {{ content:""; position:absolute; width:11px; height:11px;
  pointer-events:none; opacity:.5; }}
.band::before {{ top:-1px; left:24px; border-top:1px solid var(--acid);
  border-left:1px solid var(--acid); }}
.band::after {{ bottom:-1px; right:24px; border-bottom:1px solid var(--acid);
  border-right:1px solid var(--acid); }}
.band-in {{ padding:104px 0; }}
.duel {{ display:grid; grid-template-columns:1fr 1fr; gap:56px; margin-top:44px; }}
.duel > div {{ border-top:2px solid currentColor; padding-top:20px; }}
.duel .hi {{ color:var(--s1); }} .duel .lo {{ color:var(--s2); }}
.duel .fig {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:clamp(44px,7vw,84px); line-height:1; letter-spacing:-.05em; }}
.duel .who {{ color:var(--ink); font-size:16px; margin-top:12px; font-weight:560; }}
.duel .what {{ color:var(--mut); font-size:13.5px; margin-top:4px; }}
@media (max-width:760px) {{ .duel {{ grid-template-columns:1fr; gap:34px; }} }}

/* stat band: numbers on hairlines, no cards */
section {{ padding:104px 0; }}
.numbers {{ display:grid; grid-template-columns:repeat(4,1fr); gap:1px;
  background:var(--hair); border-block:1px solid var(--hair); }}
.numbers > div {{ background:var(--plane); padding:30px 26px 34px; }}
.numbers .k {{ font-size:10.5px; letter-spacing:.16em; text-transform:uppercase;
  color:var(--mut); font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
.numbers .v {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums; font-size:clamp(30px,3.6vw,44px);
  letter-spacing:-.04em; line-height:1.08; margin-top:14px; }}
.numbers .c {{ font-size:12.5px; color:var(--mut); margin-top:8px; }}
@media (max-width:900px) {{ .numbers {{ grid-template-columns:repeat(2,1fr); }} }}
@media (max-width:520px) {{ .numbers {{ grid-template-columns:1fr; }} }}

.panel {{ background:var(--surface); border:1px solid var(--hair); border-radius:18px;
  padding:26px 28px; box-shadow:inset 0 1px 0 var(--hair-2); }}
.split {{ display:grid; grid-template-columns:1.35fr 1fr; gap:22px; align-items:start; }}
@media (max-width:960px) {{ .split {{ grid-template-columns:1fr; }} }}

/* checks */
.checks {{ list-style:none; margin:0; padding:0; }}
.chk {{ padding:17px 0; border-bottom:1px solid var(--hair); }}
.chk:last-child {{ border-bottom:0; }}
.chk-top {{ display:flex; align-items:center; gap:12px; margin-bottom:10px; }}
.chk-top code {{ font-size:13px; color:var(--ink-2); }}
.tag {{ margin-left:auto; font-size:10.5px; letter-spacing:.09em; text-transform:uppercase;
  padding:3px 10px; border-radius:999px; border:1px solid var(--hair-2); color:var(--mut); }}
.s-warn .tag {{ color:var(--warn); border-color:color-mix(in srgb,var(--warn) 42%,transparent); }}
.s-fail .tag {{ color:var(--crit); border-color:color-mix(in srgb,var(--crit) 46%,transparent); }}
.s-pass .tag {{ color:var(--good); border-color:color-mix(in srgb,var(--good) 42%,transparent); }}
.chk-bot {{ display:flex; align-items:baseline; gap:22px; margin-top:10px; font-size:12px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; color:var(--mut); }}
.chk-bot .big {{ font-size:16px; color:var(--ink); letter-spacing:-.02em; }}
svg.bullet {{ display:block; }}
svg.bullet .trk {{ fill:var(--grid); }}
svg.bullet .fil {{ transition:width 480ms cubic-bezier(.23,1,.32,1); }}
.s-pass svg.bullet .fil {{ fill:var(--good); }}
.s-warn svg.bullet .fil {{ fill:var(--warn); }}
.s-fail svg.bullet .fil {{ fill:var(--crit); }}
svg.bullet line {{ stroke:var(--ink); stroke-width:2; vector-effect:non-scaling-stroke; }}
svg.bullet .t-warn line {{ opacity:.4; stroke-dasharray:3 3; }}
svg.bullet .t-fail line {{ opacity:.75; }}

/* controls */
.controls {{ display:flex; align-items:center; gap:26px; flex-wrap:wrap; margin:0 0 24px; }}
.ctl {{ display:flex; align-items:center; gap:10px; }}
.ctl-k {{ font-size:10.5px; letter-spacing:.12em; text-transform:uppercase; color:var(--mut); }}
.segs {{ display:inline-flex; background:var(--surface); border:1px solid var(--hair);
  border-radius:10px; padding:3px; gap:2px; }}
.seg {{ appearance:none; border:0; background:transparent; color:var(--ink-2); font:inherit;
  font-size:13px; padding:6px 14px; border-radius:7px; cursor:pointer;
  transition:background 160ms cubic-bezier(.23,1,.32,1), color 150ms ease,
    transform 120ms cubic-bezier(.23,1,.32,1); }}
.seg:hover {{ color:var(--ink); }}
.seg:active {{ transform:scale(.97); }}
.seg.is-on {{ background:var(--raised); color:var(--ink);
  box-shadow:inset 0 1px 0 var(--hair-2), 0 1px 3px rgba(0,0,0,.3); }}
.ctl-note {{ font-size:12.5px; color:var(--mut); margin:0 0 0 auto; }}
.pane {{ display:none; }} .pane.is-on {{ display:block; }}

/* charts */
svg.chart {{ display:block; overflow:visible; height:auto; }}
svg.chart .gr {{ stroke:var(--grid); stroke-width:1; }}
svg.chart .ax {{ fill:var(--mut); font-size:10.5px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
svg.chart .gl {{ fill:var(--ink); font-size:12.5px; }}
svg.chart .gn, svg.chart .vl {{ fill:var(--mut); font-size:10.5px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
svg.chart .s1, svg.chart .d1 {{ fill:var(--s1); }}
svg.chart .s2, svg.chart .d2 {{ fill:var(--s2); }}
svg.chart .lnk {{ stroke:var(--axis); stroke-width:2.5; }}
svg.chart .row {{ transition:opacity 150ms cubic-bezier(.23,1,.32,1); }}
@media (hover:hover) and (pointer:fine) {{
  svg.chart:hover .row {{ opacity:.35; }}
  svg.chart .row:hover {{ opacity:1; }}
}}
/* sankey */
svg.sankey .rb {{ opacity:.5; transition:opacity 180ms ease; }}
svg.sankey .rb.ok {{ fill:var(--s1); }}
svg.sankey .rb.bad {{ fill:var(--acid); opacity:.82; }}
svg.sankey .rb.warnrb {{ fill:var(--s2); opacity:.62; }}
svg.sankey:hover .rb {{ opacity:.22; }}
svg.sankey .rb:hover {{ opacity:.95; }}
svg.sankey .nd {{ fill:var(--ink); opacity:.85; }}
svg.sankey .nl {{ fill:var(--ink); font-size:13px; font-weight:600; }}
svg.sankey .nn {{ fill:var(--ink); font-size:16px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
svg.sankey .ns {{ fill:var(--mut); font-size:11px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}

/* heatmap */
svg.heat .cell {{ fill:var(--s1); fill-opacity:calc(.12 + var(--v) * .8); }}
svg.heat .cell.bad {{ fill:var(--acid); fill-opacity:calc(.14 + var(--v) * 1.5); }}
svg.heat .cv {{ fill:var(--ink); font-size:12px; font-weight:600;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  pointer-events:none; }}

/* treemap */
svg.tree .tm {{ fill:var(--s1); fill-opacity:calc(.2 + var(--v) * .7);
  stroke:var(--surface); stroke-width:1; }}
svg.tree .tml {{ fill:var(--ink); font-size:14px; font-weight:600; }}
svg.tree .tmn {{ fill:var(--ink); opacity:.72; font-size:13px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}

/* reliability */
svg.rel .diag {{ stroke:var(--ink); stroke-width:1.5; stroke-dasharray:5 5;
  opacity:.34; }}
svg.rel .cal {{ fill:none; stroke:var(--s1); stroke-width:2; opacity:.72; }}
svg.rel .cdot {{ fill:var(--s1); }}

/* ridgeline */
svg.ridge-c .ridge {{ stroke-width:1.5; }}
svg.ridge-c .neg {{ fill:var(--ink); fill-opacity:.1; stroke:var(--ink);
  stroke-opacity:.3; }}
svg.ridge-c .pos {{ fill:var(--acid); fill-opacity:.2; stroke:var(--acid);
  stroke-opacity:.78; }}

svg.chart .bub {{ fill:var(--s1); opacity:.78; stroke:var(--surface);
  stroke-width:2; }}
svg.chart .zero {{ stroke:var(--axis); stroke-width:1.5; }}
svg.chart .axt {{ fill:var(--mut); font-size:11px; letter-spacing:.04em; }}
svg.chart .slope {{ stroke-width:2.5; fill:none; opacity:.85; }}
svg.chart .s1l {{ stroke:var(--s1); }} svg.chart .s2l {{ stroke:var(--s2); }}

/* charts breathe: fill the panel rather than sitting in it */
.chartbox {{ padding:26px 12px 10px; }}
.chartbox .legend {{ padding-left:14px; }}

.legend {{ display:flex; align-items:center; gap:9px; font-size:12.5px; color:var(--ink-2);
  margin:0 0 16px; }}
.key {{ width:11px; height:11px; border-radius:3px; display:inline-block; }}
.k1 {{ background:var(--s1); }} .k2 {{ background:var(--s2); margin-left:16px; }}
.empty {{ color:var(--mut); font-size:13px; margin:0; }}

table {{ border-collapse:collapse; width:100%; font-size:12.5px; }}
th, td {{ padding:10px 12px; text-align:right; white-space:nowrap; }}
thead th {{ font-size:10.5px; letter-spacing:.09em; text-transform:uppercase;
  color:var(--mut); font-weight:600; border-bottom:1px solid var(--axis); }}
tbody th {{ text-align:left; font-weight:500; color:var(--ink);
  font-family:ui-sans-serif,system-ui,sans-serif; }}
tbody tr {{ transition:background 130ms ease; }}
tbody tr:hover {{ background:var(--raised); }}
tbody tr + tr th, tbody tr + tr td {{ border-top:1px solid var(--hair); }}
.scroll-x {{ overflow-x:auto; }}
.note {{ font-size:13.5px; color:var(--ink-2); border-left:2px solid var(--axis);
  padding-left:16px; margin:22px 0 0; max-width:66ch; }}

/* method: a numbered narrative, not cards */
.steps {{ counter-reset:s; display:grid; gap:2px; margin-top:40px; }}
.step {{ display:grid; grid-template-columns:64px 1fr; gap:24px; padding:26px 0;
  border-top:1px solid var(--hair); align-items:start; }}
.step::before {{ counter-increment:s; content:counter(s,decimal-leading-zero);
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:13px;
  color:var(--mut); padding-top:4px; }}
.step h3 {{ font-size:19px; margin:0 0 8px; font-weight:600; letter-spacing:-.02em; }}
.step p {{ margin:0; color:var(--ink-2); max-width:64ch; font-size:15px; }}
@media (max-width:620px) {{ .step {{ grid-template-columns:1fr; gap:8px; }} }}

footer {{ border-top:1px solid var(--hair); padding:44px 0 64px; font-size:13px;
  color:var(--mut); line-height:1.75; }}
footer code {{ color:var(--ink-2); }}
footer strong {{ color:var(--ink-2); font-weight:600; }}

.grain {{ position:fixed; inset:0; z-index:90; pointer-events:none;
  opacity:var(--grain); mix-blend-mode:overlay; }}
/* Dither: shade simulated with a dot pattern on a limited palette, which is also what
   the unit chart further down does with people. Same device, two scales. */
.dither {{ position:fixed; inset:0; z-index:89; pointer-events:none; opacity:.5;
  background-image:radial-gradient(currentColor .5px, transparent .5px);
  background-size:3px 3px; color:var(--ink); mix-blend-mode:soft-light; }}

/* ---- full-bleed acid field: the single dominant colour moment ---- */
.acid {{ background:var(--acid); color:var(--acid-ink); position:relative;
  overflow:hidden; padding:0; }}
.acid .kicker, .acid .yr {{ color:color-mix(in srgb, var(--acid-ink) 58%, transparent); }}
.acid .say, .acid p {{ color:color-mix(in srgb, var(--acid-ink) 78%, transparent); }}
.acid .say b, .acid b {{ color:var(--acid-ink); }}
.topo {{ position:absolute; inset:0; width:100%; height:100%; pointer-events:none;
  opacity:.16; }}
.topo path {{ fill:none; stroke:currentColor; stroke-width:1.2;
  vector-effect:non-scaling-stroke; }}
.acid-in {{ position:relative; z-index:1; padding:120px 0 128px; }}

/* ---- type that runs off both edges, the way a poster does ---- */
.mega {{ font-size:clamp(84px,21vw,300px); line-height:.78; letter-spacing:-.07em;
  font-weight:900; margin:0; white-space:nowrap; }}
.bleed {{ width:100vw; margin-left:calc(50% - 50vw); padding:0 24px; }}
.mega-wrap {{ overflow:hidden; }}
.mega.cut {{ margin-left:-.06em; }}

/* horizontal slice displacement, the way the reference slices a portrait */
.sliced {{ position:relative; display:inline-block; isolation:isolate; }}
.sliced > span {{ display:block; position:relative; z-index:2; }}
.sliced .sl {{ position:absolute; left:0; top:0; white-space:nowrap; z-index:1; }}
/* two inks, slightly out of register, the way a risograph misprints */
.sliced .sl1 {{ transform:translate(.9%, .055em); color:var(--acid); opacity:.85;
  mix-blend-mode:screen; }}
.sliced .sl2 {{ transform:translate(-.7%, -.045em); color:var(--s1); opacity:.7;
  mix-blend-mode:screen; }}
.sliced .sl3 {{ display:none; }}
:root[data-theme="light"] .sliced .sl1,
:root[data-theme="light"] .sliced .sl2 {{ mix-blend-mode:multiply; opacity:.5; }}

/* ---- one marquee, and only one ---- */
.mq {{ width:100vw; margin-left:calc(50% - 50vw); overflow:hidden; padding:22px 0;
  border-block:1px solid currentColor; }}
.mq-track {{ display:flex; gap:56px; width:max-content;
  animation:slide 34s linear infinite; }}
.mq span {{ font-size:clamp(22px,3.4vw,46px); font-weight:700; letter-spacing:-.03em;
  white-space:nowrap; }}
.mq em {{ font-style:normal; opacity:.42; }}
@keyframes slide {{ to {{ transform:translateX(-50%); }} }}
@media (prefers-reduced-motion:reduce) {{ .mq-track {{ animation:none; }} }}

/* ---- scale contrast: tiny chrome beside monumental figures ---- */
.tiny {{ font-size:10px; letter-spacing:.2em; text-transform:uppercase;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}

/* human cost */
.cost {{ display:grid; grid-template-columns:1fr 1fr; gap:64px; margin-top:56px; }}
.cost h3 {{ font-size:15px; font-weight:600; margin:0 0 6px; letter-spacing:-.01em; }}
.cost .of {{ font-size:12.5px; color:var(--mut); margin:0 0 20px; }}
.cost .count {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:clamp(46px,7.4vw,92px); line-height:.94; letter-spacing:-.05em;
  margin:20px 0 4px; }}
.cost .count.bad {{ color:var(--acid); }}
.cost .cap {{ font-size:13.5px; color:var(--ink-2); max-width:34ch; }}
svg.dots {{ display:block; overflow:visible; height:auto; }}
svg.dots .off {{ fill:var(--ink); opacity:.13; }}
svg.dots .on {{ fill:var(--acid); transform-origin:center; transform-box:fill-box; }}
@media (max-width:820px) {{ .cost {{ grid-template-columns:1fr; gap:48px; }} }}

.ratio {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:clamp(84px,17vw,240px); line-height:.82; letter-spacing:-.06em;
  color:var(--acid); margin:0; }}
.ratio-say {{ max-width:30ch; font-size:clamp(18px,2.1vw,26px); line-height:1.38;
  letter-spacing:-.02em; color:var(--ink); margin:26px 0 0; }}
.total {{ border-top:1px solid var(--hair); margin-top:64px; padding-top:34px;
  display:flex; align-items:baseline; gap:24px; flex-wrap:wrap; }}
.total .n {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:clamp(38px,5.4vw,68px); letter-spacing:-.045em; line-height:1; }}
.total p {{ margin:0; max-width:46ch; color:var(--ink-2); font-size:15px; }}

/* sources */
/* Sources as cards with a numbered bubble. A list of hairline rows reads as fine
   print; a card with a badge reads as something worth clicking. */
.srcs {{ list-style:none; margin:44px 0 0; padding:0; display:grid;
  grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:14px; }}
.srcs li {{ position:relative; }}
.srcs a.card {{ display:grid; grid-template-columns:44px 1fr; gap:16px;
  align-items:start; padding:22px 22px 24px; border-radius:18px;
  background:var(--raised); border:1px solid var(--hair);
  box-shadow:inset 0 1px 0 var(--hair-2); text-decoration:none; color:inherit;
  height:100%; position:relative; overflow:hidden;
  transition:transform 200ms cubic-bezier(.23,1,.32,1),
    border-color 200ms ease, box-shadow 200ms ease; }}
.srcs a.card::after {{ content:""; position:absolute; inset:0; pointer-events:none;
  opacity:0; transition:opacity 220ms ease;
  background:radial-gradient(360px 180px at 88% -10%,
    color-mix(in srgb, var(--acid) 16%, transparent), transparent 66%); }}
@media (hover:hover) and (pointer:fine) {{
  .srcs a.card:hover {{ transform:translateY(-3px); border-color:var(--hair-2);
    box-shadow:inset 0 1px 0 var(--hair-2), 0 14px 34px rgba(0,0,0,.34); }}
  .srcs a.card:hover::after {{ opacity:1; }}
  .srcs a.card:hover .bubble {{ background:var(--acid); color:var(--acid-ink);
    border-color:var(--acid); }}
}}
.srcs a.card:active {{ transform:translateY(-1px) scale(.995); }}
.bubble {{ width:44px; height:44px; border-radius:999px; display:grid;
  place-items:center; border:1px solid var(--hair-2); color:var(--ink-2);
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:13px;
  transition:background 200ms ease, color 200ms ease, border-color 200ms ease; }}
.srcs .ttl {{ font-size:16.5px; font-weight:600; letter-spacing:-.018em;
  line-height:1.25; display:block; }}
.srcs .what {{ display:block; font-weight:400; font-size:13.5px; color:var(--mut);
  margin-top:9px; line-height:1.5; }}
.srcs .yr {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:11px; color:var(--mut); letter-spacing:.1em; margin-top:12px;
  display:inline-block; border:1px solid var(--hair); border-radius:999px;
  padding:3px 10px; }}
@media (max-width:620px) {{ .srcs {{ grid-template-columns:1fr; }} }}

/* Reveal is scroll-driven CSS, not script. If the browser cannot do it, content is
   simply visible: an entrance animation must never be load-bearing for legibility. */
@supports (animation-timeline: view()) {{
  @media (prefers-reduced-motion: no-preference) {{
    .reveal {{ animation:rise both; animation-timeline:view();
      animation-range:entry 0% cover 20%; }}
    .cost svg.dots .on {{ animation:pop 420ms cubic-bezier(.23,1,.32,1) both;
      animation-delay:var(--d,0ms); animation-timeline:view();
      animation-range:entry 0% cover 16%; }}
  }}
}}
@keyframes rise {{ from {{ opacity:0; transform:translateY(20px); }} }}
@keyframes pop {{ from {{ opacity:0; transform:scale(.4); }} }}
@media (prefers-reduced-motion:reduce) {{
  html {{ scroll-behavior:auto; }}
  *, *::before, *::after {{ transition-duration:1ms !important; animation:none !important; }}
}}
@media (max-width:760px) {{ .shell {{ padding:0 20px; }} section, .band-in {{ padding:64px 0; }} }}
@media print {{
  body {{ background:#fff; }} nav, .controls {{ display:none; }}
  .pane {{ display:block !important; }} .reveal {{ opacity:1; transform:none; }}
  .hero {{ min-height:auto; }}
}}
</style>
</head>
<body class="v-{esc(v)}">

<div class="dither" aria-hidden="true"></div>
<svg class="grain" aria-hidden="true" focusable="false">
  <filter id="gr"><feTurbulence type="fractalNoise" baseFrequency="0.86" numOctaves="4"
    stitchTiles="stitch"/><feColorMatrix type="saturate" values="0"/></filter>
  <rect width="100%" height="100%" filter="url(#gr)"/>
</svg>

<nav>
  <span class="mark">fairness&#8209;gate <i>/ ACS income</i></span>
  <ul>
    <li><a href="#cost">Who it misses</a></li>
    <li><a href="#checks">Checks</a></li>
    <li><a href="#groups">Groups</a></li>
    <li><a href="#flow">Flow</a></li>
    <li><a href="#method">Method</a></li>
    <li><a href="#sources">Sources</a></li>
  </ul>
  <div class="right">
    <span class="pill"><i></i>{esc(STATUS_LABEL[v])}</span>
    <button type="button" class="tgl" id="themer" aria-label="Switch colour theme">Light</button>
  </div>
</nav>

<header class="hero">
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

<div class="mq" aria-hidden="true">
  <div class="mq-track">{_marquee(cost)}{_marquee(cost)}</div>
</div>

<div class="band" id="cost">
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

<div class="acid">
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

<div class="band" id="checks">
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

<section id="groups">
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

<section id="flow">
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

<div class="band" id="method">
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

<div class="band" id="sources">
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

<footer>
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
  document.querySelectorAll('.seg').forEach(function (b) {{
    b.addEventListener('click', function () {{
      state[b.dataset.k] = b.dataset.v;
      b.parentNode.querySelectorAll('.seg').forEach(function (o) {{
        o.classList.toggle('is-on', o === b);
      }});
      sync();
    }});
  }});

}})();
</script>
</body>
</html>
"""


def write(result: dict, tables: dict[str, pd.DataFrame]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "index.html").write_text(render(result, tables), encoding="utf-8")
