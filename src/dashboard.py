"""Render the audit as a single self-contained HTML page.

Constraints that shaped this, in order of weight:

1. **Deterministic output.** The gate regenerates every artifact and requires a
   byte-identical match, so the page is rendered server-side. A client-side chart
   library would move the numbers out of the diff and back into trust.
2. **No dependencies.** No CDN, no fonts fetched at runtime, no JavaScript needed to
   read a single number. A compliance artifact that phones out to three origins is a
   supply-chain surface, and this one has to still open in five years.
3. **Dense, not decorative.** Numbers are tabular-figure monospace and right-aligned.
   Charts are thin marks on a recessive grid.

Colors come from the validated categorical palette (blue/orange, worst adjacent CVD
delta E 24.7) and the fixed status palette, which never doubles as a series color.
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"

STATUS_LABEL = {"pass": "Pass", "warn": "Warn", "fail": "Fail"}


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def fmt(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}"


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


# --------------------------------------------------------------------------- #
# Charts. Each returns a complete <svg>. Geometry is integer-ish so the output
# is stable across platforms.
# --------------------------------------------------------------------------- #
def bullet_row(check: dict, width: int = 560) -> str:
    """One policy check as a bullet bar: measured mark, target tick, tolerance tick.

    Chosen over a gauge or a progress bar because the question is not "how full" but
    "where does this sit relative to two declared lines", which is exactly what a
    bullet chart encodes.
    """
    fail_at, warn_at = check["fail_at"], check["warn_at"]
    measured = check["measured"]
    is_ceiling = check["direction"] == "max"

    limits = [v for v in (fail_at, warn_at, measured) if v is not None]
    axis_max = max(limits) * 1.25 if is_ceiling else max(1.0, max(limits) * 1.05)
    axis_max = axis_max or 1.0

    def x(v: float) -> float:
        return round(min(max(v / axis_max, 0.0), 1.0) * width, 1)

    status = check["status"]
    bar_w = x(measured)

    ticks = []
    for value, cls, label in ((warn_at, "tick-warn", "target"), (fail_at, "tick-fail", "limit")):
        if value is None:
            continue
        pos = x(value)
        ticks.append(
            f'<line class="{cls}" x1="{pos}" y1="2" x2="{pos}" y2="22" />'
            f'<title>{label} {fmt(value)}</title>'
        )

    return (
        f'<svg class="bullet" viewBox="0 0 {width} 24" width="100%" height="24" '
        f'role="img" aria-label="measured {fmt(measured, 4)}">'
        f'<rect class="track" x="0" y="7" width="{width}" height="10" rx="5" />'
        f'<rect class="fill status-{status}" x="0" y="7" width="{bar_w}" height="10" rx="5">'
        f'<title>measured {fmt(measured, 4)}</title></rect>'
        + "".join(ticks)
        + "</svg>"
    )


def grouped_bars(table: pd.DataFrame, attribute: str) -> str:
    """True positive rate and false positive rate, per group.

    Grouped rather than stacked: these are two independent rates, not parts of a whole.
    Sorted by group size so the reader meets the populous groups first and the eye does
    not read the ordering as a ranking of severity.
    """
    sub = (
        table[(table["attribute"] == attribute) & table["reportable"]]
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )
    if sub.empty:
        return ""

    row_h, bar_h, gap, label_w, pad_r = 46, 13, 4, 250, 56
    plot_w = 430
    height = len(sub) * row_h + 34
    total_w = label_w + plot_w + pad_r

    grid = "".join(
        f'<line class="grid" x1="{label_w + plot_w * t}" y1="20" '
        f'x2="{label_w + plot_w * t}" y2="{height - 14}" />'
        f'<text class="axis" x="{label_w + plot_w * t}" y="14" text-anchor="middle">'
        f'{int(t * 100)}%</text>'
        for t in (0, 0.25, 0.5, 0.75, 1.0)
    )

    rows = []
    for i, r in sub.iterrows():
        top = 26 + i * row_h
        tpr_w = round(r["tpr"] * plot_w, 1)
        fpr_w = round(r["fpr"] * plot_w, 1)
        rows.append(
            f'<g class="bar-row">'
            f'<text class="grp" x="{label_w - 12}" y="{top + 15}" text-anchor="end">'
            f'{esc(r["group"])}</text>'
            f'<text class="grp-n" x="{label_w - 12}" y="{top + 30}" text-anchor="end">'
            f'n&#8239;=&#8239;{int(r["n"]):,}</text>'
            f'<rect class="s1" x="{label_w}" y="{top}" width="{tpr_w}" height="{bar_h}" rx="3">'
            f'<title>{esc(r["group"])}: true positive rate {fmt(r["tpr"])}</title></rect>'
            f'<text class="val" x="{label_w + tpr_w + 8}" y="{top + 11}">{pct(r["tpr"])}</text>'
            f'<rect class="s2" x="{label_w}" y="{top + bar_h + gap}" width="{fpr_w}" '
            f'height="{bar_h}" rx="3">'
            f'<title>{esc(r["group"])}: false positive rate {fmt(r["fpr"])}</title></rect>'
            f'<text class="val" x="{label_w + fpr_w + 8}" y="{top + bar_h + gap + 11}">'
            f'{pct(r["fpr"])}</text>'
            f'</g>'
        )

    return (
        f'<svg class="chart" viewBox="0 0 {total_w} {height}" width="100%" '
        f'height="{height}" role="img" '
        f'aria-label="true and false positive rate by {esc(attribute)}">'
        + grid + "".join(rows) + "</svg>"
    )


def dumbbell(table: pd.DataFrame, attribute: str) -> str:
    """Predicted probability against observed rate, per group.

    A dumbbell rather than two bars: the quantity of interest is the distance between
    the pair, and a dumbbell draws that distance as an actual line.
    """
    sub = (
        table[(table["attribute"] == attribute) & table["reportable"]]
        .sort_values("base_rate", ascending=False)
        .reset_index(drop=True)
    )
    if sub.empty:
        return ""

    row_h, label_w, pad_r, plot_w = 30, 250, 56, 430
    height = len(sub) * row_h + 34
    total_w = label_w + plot_w + pad_r
    hi = max(float(sub["base_rate"].max()), float(sub["mean_predicted"].max()))
    axis_max = min(1.0, (hi * 1.2) or 1.0)

    def x(v: float) -> float:
        return round(label_w + min(v / axis_max, 1.0) * plot_w, 1)

    grid = "".join(
        f'<line class="grid" x1="{label_w + plot_w * t}" y1="20" '
        f'x2="{label_w + plot_w * t}" y2="{height - 14}" />'
        f'<text class="axis" x="{label_w + plot_w * t}" y="14" text-anchor="middle">'
        f'{fmt(axis_max * t, 2)}</text>'
        for t in (0, 0.5, 1.0)
    )

    rows = []
    for i, r in sub.iterrows():
        cy = 34 + i * row_h
        xa, xp = x(float(r["base_rate"])), x(float(r["mean_predicted"]))
        rows.append(
            f'<g class="bar-row">'
            f'<text class="grp" x="{label_w - 12}" y="{cy + 4}" text-anchor="end">'
            f'{esc(r["group"])}</text>'
            f'<line class="link" x1="{min(xa, xp)}" y1="{cy}" x2="{max(xa, xp)}" y2="{cy}" />'
            f'<circle class="s1" cx="{xa}" cy="{cy}" r="5">'
            f'<title>{esc(r["group"])}: observed {fmt(r["base_rate"])}</title></circle>'
            f'<circle class="s2" cx="{xp}" cy="{cy}" r="5">'
            f'<title>{esc(r["group"])}: mean predicted {fmt(r["mean_predicted"])}</title></circle>'
            f'</g>'
        )

    return (
        f'<svg class="chart" viewBox="0 0 {total_w} {height}" width="100%" '
        f'height="{height}" role="img" '
        f'aria-label="observed rate against mean predicted probability by {esc(attribute)}">'
        + grid + "".join(rows) + "</svg>"
    )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def _legend(a: str, b: str) -> str:
    return (
        f'<p class="legend"><span class="key k1"></span>{esc(a)}'
        f'<span class="key k2"></span>{esc(b)}</p>'
    )


def _checks_section(result: dict) -> str:
    order = {"fail": 0, "warn": 1, "pass": 2}
    checks = sorted(result["checks"], key=lambda c: (order[c["status"]], c["name"]))
    rows = []
    for c in checks:
        target = "" if c["warn_at"] is None else fmt(c["warn_at"])
        limit = "" if c["fail_at"] is None else fmt(c["fail_at"])
        rows.append(
            f'<div class="check">'
            f'<div class="check-head">'
            f'<span class="dot status-{c["status"]}" aria-hidden="true"></span>'
            f'<code>{esc(c["name"])}</code>'
            f'<span class="chip chip-{c["status"]}">{STATUS_LABEL[c["status"]]}</span>'
            f'</div>'
            f'{bullet_row(c)}'
            f'<div class="check-foot"><span>measured <b>{fmt(c["measured"], 4)}</b></span>'
            f'<span>target {target or "n/a"}</span><span>limit {limit or "n/a"}</span></div>'
            f'</div>'
        )
    return "".join(rows)


def _splits_table(result: dict) -> str:
    rows = []
    for name in ("val", "test", "shift"):
        s = result["scores"][name]
        rows.append(
            f"<tr><th scope='row'>{name}</th>"
            f"<td>{s['n']:,}</td><td>{fmt(s['auc'], 4)}</td>"
            f"<td>{fmt(s['average_precision'], 4)}</td><td>{fmt(s['brier'], 4)}</td>"
            f"<td>{fmt(s['ece'], 4)}</td><td>{fmt(s['accuracy'], 4)}</td>"
            f"<td class='muted'>{fmt(s['majority_baseline'], 4)}</td></tr>"
        )
    return "".join(rows)


def render(result: dict, tables: dict[str, pd.DataFrame]) -> str:
    test = tables["test"]
    race = next(s for s in result["fairness"]["test"] if s["attribute"] == "RAC1P")
    sub = test[(test["attribute"] == "RAC1P") & test["reportable"]]
    best = sub.loc[sub["tpr"].idxmax()]
    worst = sub.loc[sub["tpr"].idxmin()]

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for c in result["checks"]:
        counts[c["status"]] += 1
    verdict = result["policy_verdict"]
    d = result["design"]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fairness gate</title>
<meta name="description" content="Policy gate result for the ACS income model.">
<style>
:root {{
  color-scheme: light;
  --plane:#f9f9f7; --surface:#fcfcfb;
  --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --s1:#2a78d6; --s2:#eb6834;
  --good:#0ca30c; --warn:#fab219; --crit:#d03b3b;
  --radius:10px;
  --ease-out:cubic-bezier(.23,1,.32,1);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --plane:#0d0d0d; --surface:#1a1a19;
    --ink:#fff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
    --s1:#3987e5; --s2:#d95926;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --plane:#0d0d0d; --surface:#1a1a19;
  --ink:#fff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --s1:#3987e5; --s2:#d95926;
}}
* {{ box-sizing:border-box; }}
html {{ -webkit-text-size-adjust:100%; }}
body {{
  margin:0; background:var(--plane); color:var(--ink);
  font-family:ui-sans-serif,system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:15px; line-height:1.45;
  font-feature-settings:"kern" 1; text-rendering:optimizeLegibility;
}}
.wrap {{ max-width:1080px; margin:0 auto; padding:40px 20px 72px; }}
h1 {{ font-size:22px; letter-spacing:-.01em; margin:0 0 4px; }}
h2 {{ font-size:15px; letter-spacing:.06em; text-transform:uppercase;
      color:var(--ink-2); margin:44px 0 14px; font-weight:600; }}
p {{ max-width:68ch; color:var(--ink-2); margin:0 0 12px; }}
code, .num, td, th {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
      font-variant-numeric:tabular-nums lining-nums; }}
.sub {{ color:var(--muted); font-size:13px; margin:0; }}

.masthead {{ display:flex; justify-content:space-between; align-items:flex-start;
  gap:20px; flex-wrap:wrap; padding-bottom:20px; border-bottom:1px solid var(--border); }}
.verdict {{ display:inline-flex; align-items:center; gap:8px; padding:7px 14px;
  border-radius:999px; font-size:13px; font-weight:600; letter-spacing:.04em;
  text-transform:uppercase; border:1px solid var(--border); background:var(--surface); }}
.verdict .dot {{ width:9px; height:9px; }}

.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  gap:1px; background:var(--border); border:1px solid var(--border);
  border-radius:var(--radius); overflow:hidden; margin-top:24px; }}
.tile {{ background:var(--surface); padding:16px 18px; }}
.tile .k {{ font-size:12px; color:var(--muted); letter-spacing:.06em;
  text-transform:uppercase; }}
.tile .v {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums; font-size:26px; letter-spacing:-.02em;
  margin-top:6px; }}
.tile .c {{ font-size:12px; color:var(--muted); margin-top:2px; }}

.panel {{ background:var(--surface); border:1px solid var(--border);
  border-radius:var(--radius); padding:20px; }}

.check {{ padding:14px 0; border-bottom:1px solid var(--border); }}
.check:last-child {{ border-bottom:0; }}
.check-head {{ display:flex; align-items:center; gap:10px; margin-bottom:8px; }}
.check-head code {{ font-size:13px; color:var(--ink); }}
.check-foot {{ display:flex; gap:20px; font-size:12px; color:var(--muted);
  margin-top:6px; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
.check-foot b {{ color:var(--ink); font-weight:600; }}
.dot {{ width:8px; height:8px; border-radius:50%; flex:0 0 auto; }}
.status-pass {{ background:var(--good); }} .status-warn {{ background:var(--warn); }}
.status-fail {{ background:var(--crit); }}
.chip {{ margin-left:auto; font-size:11px; letter-spacing:.06em; text-transform:uppercase;
  padding:3px 8px; border-radius:5px; border:1px solid var(--border); color:var(--ink-2); }}
.chip-fail {{ color:var(--crit); }} .chip-warn {{ color:var(--ink-2); }}

svg.bullet .track {{ fill:var(--grid); }}
svg.bullet .fill {{ transition:opacity 160ms var(--ease-out); }}
svg.bullet .fill.status-pass {{ fill:var(--good); }}
svg.bullet .fill.status-warn {{ fill:var(--warn); }}
svg.bullet .fill.status-fail {{ fill:var(--crit); }}
svg.bullet .tick-warn, svg.bullet .tick-fail {{ stroke:var(--ink); stroke-width:2; }}
svg.bullet .tick-warn {{ stroke-dasharray:3 2; opacity:.45; }}
svg.bullet .tick-fail {{ opacity:.8; }}

svg.chart .grid {{ stroke:var(--grid); stroke-width:1; }}
svg.chart .axis {{ fill:var(--muted); font-size:10px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
svg.chart .grp {{ fill:var(--ink); font-size:12px; }}
svg.chart .grp-n {{ fill:var(--muted); font-size:10.5px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
svg.chart .val {{ fill:var(--ink-2); font-size:10.5px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
svg.chart .s1 {{ fill:var(--s1); }} svg.chart .s2 {{ fill:var(--s2); }}
svg.chart .link {{ stroke:var(--axis); stroke-width:2; }}
svg.chart .bar-row {{ transition:opacity 140ms var(--ease-out); }}
@media (hover:hover) and (pointer:fine) {{
  svg.chart:hover .bar-row {{ opacity:.45; }}
  svg.chart .bar-row:hover {{ opacity:1; }}
}}

.legend {{ display:flex; align-items:center; gap:8px; font-size:12px;
  color:var(--ink-2); margin:0 0 10px; }}
.key {{ width:10px; height:10px; border-radius:3px; display:inline-block; }}
.key.k1 {{ background:var(--s1); }}
.key.k2 {{ background:var(--s2); margin-left:14px; }}

table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th, td {{ padding:8px 12px; text-align:right; }}
thead th {{ font-size:11px; letter-spacing:.06em; text-transform:uppercase;
  color:var(--muted); font-weight:600; border-bottom:1px solid var(--axis); }}
tbody th {{ text-align:left; color:var(--ink); font-weight:600; }}
tbody tr + tr td, tbody tr + tr th {{ border-top:1px solid var(--border); }}
td.muted {{ color:var(--muted); }}

.note {{ font-size:13px; color:var(--ink-2); border-left:2px solid var(--axis);
  padding-left:14px; margin:14px 0 0; max-width:68ch; }}
footer {{ margin-top:56px; padding-top:20px; border-top:1px solid var(--border);
  font-size:12px; color:var(--muted); }}
a {{ color:var(--ink); text-decoration-thickness:1px; text-underline-offset:2px; }}

.stagger > * {{ animation:rise 320ms var(--ease-out) backwards; }}
.stagger > :nth-child(2) {{ animation-delay:40ms; }}
.stagger > :nth-child(3) {{ animation-delay:80ms; }}
.stagger > :nth-child(4) {{ animation-delay:120ms; }}
@keyframes rise {{ from {{ opacity:0; transform:translateY(6px); }} }}
@media (prefers-reduced-motion:reduce) {{
  .stagger > * {{ animation:none; }}
  * {{ transition-duration:1ms !important; }}
}}
@media print {{ body {{ background:#fff; }} .panel {{ break-inside:avoid; }} }}
</style>
</head>
<body>
<div class="wrap">

  <header class="masthead">
    <div>
      <h1>Fairness gate</h1>
      <p class="sub">ACS income classifier, test year {esc(d['test_year'])}.
        Thresholds declared in <code>policy.yaml</code>.</p>
    </div>
    <span class="verdict"><span class="dot status-{esc(verdict)}"></span>
      {esc(STATUS_LABEL[verdict])}</span>
  </header>

  <div class="tiles stagger">
    <div class="tile"><div class="k">Checks</div>
      <div class="v">{counts['pass']}&#8202;/&#8202;{counts['pass'] + counts['warn'] + counts['fail']}</div>
      <div class="c">{counts['warn']} warn, {counts['fail']} fail</div></div>
    <div class="tile"><div class="k">AUC, test</div>
      <div class="v">{fmt(result['scores']['test']['auc'], 3)}</div>
      <div class="c">majority baseline {fmt(result['scores']['test']['majority_baseline'], 3)}</div></div>
    <div class="tile"><div class="k">TPR gap, race</div>
      <div class="v">{fmt(race['tpr_gap'])}</div>
      <div class="c">base rate gap {fmt(race['base_rate_gap'])}</div></div>
    <div class="tile"><div class="k">People audited</div>
      <div class="v">{result['scores']['test']['n']:,}</div>
      <div class="c">{race['groups_suppressed']} group too small to report</div></div>
  </div>

  <h2>What the gap means</h2>
  <p>Among people who genuinely earn above the threshold, the model finds
    <b>{pct(float(best['tpr']))}</b> of the {esc(best['group'])} group and
    <b>{pct(float(worst['tpr']))}</b> of the {esc(worst['group'])} group. Accuracy is
    close to flat across groups, which is why accuracy is the wrong thing to look at:
    a model can be equally accurate everywhere and still place its errors where they
    matter most.</p>

  <h2>Policy checks</h2>
  <div class="panel">{_checks_section(result)}</div>
  <p class="note">The solid tick is the limit that fails the build. The dashed tick is
    the target the project is aiming at. The distance between them is deliberate and is
    argued in <code>policy.yaml</code>.</p>

  <h2>Error rates by race</h2>
  <div class="panel">
    {_legend('True positive rate', 'False positive rate')}
    {grouped_bars(test, 'RAC1P')}
  </div>

  <h2>Error rates by sex</h2>
  <div class="panel">
    {_legend('True positive rate', 'False positive rate')}
    {grouped_bars(test, 'SEX')}
  </div>

  <h2>Predicted against observed</h2>
  <div class="panel">
    {_legend('Observed rate', 'Mean predicted')}
    {dumbbell(test, 'RAC1P')}
  </div>
  <p class="note">A calibrated model puts the two dots on top of each other. The
    distance is the calibration gap for that group, and it is bounded by policy rather
    than assumed away.</p>

  <h2>Performance across splits</h2>
  <div class="panel">
    <table>
      <thead><tr><th scope="col">Split</th><th scope="col">n</th><th scope="col">AUC</th>
        <th scope="col">AP</th><th scope="col">Brier</th><th scope="col">ECE</th>
        <th scope="col">Accuracy</th><th scope="col">Majority</th></tr></thead>
      <tbody>{_splits_table(result)}</tbody>
    </table>
  </div>
  <p class="note">The validation ECE is near zero because the calibrator was fitted on
    that split. It is not a result. Calibration error grows on the later year and grows
    again on the four held-out states.</p>

  <footer>
    Generated {esc(result['generated_at'])} from commit
    <code>{esc(result['git_sha'])}</code>, pipeline
    <code>{esc(result.get('pipeline_fingerprint', 'n/a'))}</code>.
    This page is written by <code>scripts/run_audit.py</code> and is verified against the
    audit on every commit. Editing it by hand fails the build.
    <br>Not a deployable system. No decision about any person should be made with it.
  </footer>

</div>
</body>
</html>
"""


def write(result: dict, tables: dict[str, pd.DataFrame]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "index.html").write_text(render(result, tables), encoding="utf-8")
