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
        f'<svg class="chart" viewBox="0 0 {w} {h}" width="100%" height="{h}" '
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
        f'<svg class="chart" viewBox="0 0 {w} {h}" width="100%" height="{h}" '
        f'role="img" aria-label="calibration by {esc(attribute)}">{grid}{rows}</svg>'
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
    sub = test[(test["attribute"] == "RAC1P") & test["reportable"]]
    best, worst = sub.loc[sub["tpr"].idxmax()], sub.loc[sub["tpr"].idxmin()]

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for c in result["checks"]:
        counts[c["status"]] += 1
    v = result["policy_verdict"]
    d = result["design"]
    ts = result["scores"]["test"]

    return f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fairness gate</title>
<meta name="description" content="Policy gate for the ACS income model. {counts['pass']} pass, {counts['warn']} warn, {counts['fail']} fail.">
<style>
:root, :root[data-theme="dark"] {{
  color-scheme: dark;
  --plane:#08090a; --surface:#101214; --raised:#16191d; --hair:rgba(255,255,255,.075);
  --hair-2:rgba(255,255,255,.14);
  --ink:#f3f5f6; --ink-2:#9aa1a7; --mut:#6b7176;
  --grid:#1e2226; --axis:#2b3035;
  --s1:#3987e5; --s2:#d95926;
  --good:#0ca30c; --warn:#fab219; --crit:#d03b3b;
  --glow:rgba(57,135,229,.16);
}}
:root[data-theme="light"] {{
  color-scheme: light;
  --plane:#f4f4f2; --surface:#fcfcfb; --raised:#ffffff; --hair:rgba(11,11,11,.10);
  --hair-2:rgba(11,11,11,.18);
  --ink:#0b0b0b; --ink-2:#52514e; --mut:#898781;
  --grid:#e6e5df; --axis:#c3c2b7;
  --s1:#2a78d6; --s2:#eb6834;
  --good:#0ca30c; --warn:#fab219; --crit:#d03b3b;
  --glow:rgba(42,120,214,.10);
}}
*,*::before,*::after {{ box-sizing:border-box; }}
html {{ -webkit-text-size-adjust:100%; scroll-behavior:smooth; }}
body {{
  margin:0; background:var(--plane); color:var(--ink);
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,sans-serif;
  font-size:15px; line-height:1.5; letter-spacing:-.005em;
  font-feature-settings:"kern" 1; text-rendering:optimizeLegibility;
  background-image:radial-gradient(900px 480px at 78% -8%, var(--glow), transparent 70%);
  background-repeat:no-repeat;
}}
.mono, code, .num, table {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums lining-nums; }}
.wrap {{ max-width:1160px; margin:0 auto; padding:0 24px 96px; }}
.mut {{ color:var(--mut); }}

/* bar */
.bar {{ position:sticky; top:0; z-index:5; display:flex; align-items:center; gap:14px;
  padding:13px 24px; background:color-mix(in srgb, var(--plane) 86%, transparent);
  backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);
  border-bottom:1px solid var(--hair); }}
.brand {{ font-size:13px; font-weight:650; letter-spacing:.02em; }}
.brand span {{ color:var(--mut); font-weight:400; }}
.bar .fp {{ margin-left:auto; font-size:11.5px; color:var(--mut);
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
.tgl {{ appearance:none; border:1px solid var(--hair-2); background:transparent;
  color:var(--ink-2); border-radius:7px; padding:5px 11px; font-size:12px; cursor:pointer;
  transition:transform 140ms cubic-bezier(.23,1,.32,1), color 140ms ease,
             border-color 140ms ease; }}
.tgl:hover {{ color:var(--ink); border-color:var(--ink-2); }}
.tgl:active {{ transform:scale(.96); }}

/* hero */
.hero {{ padding:68px 0 44px; border-bottom:1px solid var(--hair); }}
.eyebrow {{ font-size:11.5px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--mut); margin:0 0 20px; }}
.verdict {{ display:flex; align-items:baseline; gap:20px; flex-wrap:wrap; }}
.verdict h1 {{ font-size:clamp(52px,9vw,104px); line-height:.88; letter-spacing:-.045em;
  margin:0; font-weight:680; }}
.verdict .tally {{ font-size:15px; color:var(--ink-2);
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
.v-pass h1 {{ color:var(--good); }} .v-warn h1 {{ color:var(--warn); }}
.v-fail h1 {{ color:var(--crit); }}
.lede {{ max-width:60ch; font-size:17px; line-height:1.55; color:var(--ink-2);
  margin:26px 0 0; }}
.lede b {{ color:var(--ink); font-weight:600; }}

/* tiles */
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
  gap:14px; margin:34px 0 0; }}
.tile {{ background:linear-gradient(180deg,var(--raised),var(--surface));
  border:1px solid var(--hair); border-radius:14px; padding:18px 20px 20px;
  box-shadow:inset 0 1px 0 var(--hair-2); }}
.tile .k {{ font-size:11px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--mut); }}
.tile .v {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums; font-size:34px; letter-spacing:-.035em;
  line-height:1.1; margin-top:10px; }}
.tile .c {{ font-size:12px; color:var(--mut); margin-top:5px; }}

h2 {{ font-size:13px; letter-spacing:.12em; text-transform:uppercase; color:var(--mut);
  font-weight:600; margin:64px 0 6px; }}
.h2sub {{ max-width:64ch; color:var(--ink-2); margin:0 0 20px; font-size:14.5px; }}

.card {{ background:var(--surface); border:1px solid var(--hair); border-radius:16px;
  padding:22px 24px; box-shadow:inset 0 1px 0 var(--hair-2); }}

/* Bento: one grid, cells of different spans. Exactly as many cells as there is
   content for, so nothing is padded out with a blank tile. */
.bento {{ display:grid; grid-template-columns:repeat(12,1fr); gap:12px; margin-top:12px; }}
.cell {{ background:var(--surface); border:1px solid var(--hair); border-radius:16px;
  padding:20px 22px; box-shadow:inset 0 1px 0 var(--hair-2); min-width:0;
  transition:border-color 180ms cubic-bezier(.23,1,.32,1); }}
@media (hover:hover) and (pointer:fine) {{ .cell:hover {{ border-color:var(--hair-2); }} }}
.c3 {{ grid-column:span 3; }} .c4 {{ grid-column:span 4; }} .c5 {{ grid-column:span 5; }}
.c6 {{ grid-column:span 6; }} .c7 {{ grid-column:span 7; }} .c8 {{ grid-column:span 8; }}
.c12 {{ grid-column:span 12; }}
.cell.tall {{ grid-row:span 2; display:flex; flex-direction:column;
  justify-content:space-between; }}
.cell.flush {{ padding:20px 10px 8px; }}
.cell-k {{ font-size:10.5px; letter-spacing:.11em; text-transform:uppercase;
  color:var(--mut); margin:0 0 14px; }}

/* the verdict cell carries the only large colour field on the page */
.cell.verdict-cell {{ position:relative; overflow:hidden; }}
.cell.verdict-cell::after {{ content:""; position:absolute; inset:0; pointer-events:none;
  background:radial-gradient(420px 220px at 12% 112%,
    color-mix(in srgb, var(--vc) 22%, transparent), transparent 70%); }}
.v-pass {{ --vc:var(--good); }} .v-warn {{ --vc:var(--warn); }} .v-fail {{ --vc:var(--crit); }}
.verdict-cell h1 {{ font-size:clamp(46px,6.4vw,78px); line-height:.86; letter-spacing:-.045em;
  margin:0; font-weight:680; color:var(--vc); position:relative; z-index:1; }}
.verdict-cell .tally {{ font-size:13px; color:var(--ink-2); margin-top:16px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; position:relative;
  z-index:1; }}
.stat .v {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums; font-size:clamp(26px,3.1vw,36px);
  letter-spacing:-.035em; line-height:1.05; }}
.stat .c {{ font-size:11.5px; color:var(--mut); margin-top:7px; }}
@media (max-width:980px) {{
  .c3,.c4,.c5,.c6,.c7,.c8 {{ grid-column:span 6; }}
  .cell.tall {{ grid-row:auto; }}
}}
@media (max-width:620px) {{
  .bento {{ gap:10px; }}
  .c3,.c4,.c5,.c6,.c7,.c8,.c12 {{ grid-column:span 12; }}
}}

/* controls */
.controls {{ display:flex; align-items:center; gap:26px; flex-wrap:wrap;
  margin:0 0 18px; }}
.ctl {{ display:flex; align-items:center; gap:10px; }}
.ctl-k {{ font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:var(--mut); }}
.segs {{ display:inline-flex; background:var(--surface); border:1px solid var(--hair);
  border-radius:9px; padding:3px; gap:2px; }}
.seg {{ appearance:none; border:0; background:transparent; color:var(--ink-2);
  font:inherit; font-size:13px; padding:5px 13px; border-radius:6px; cursor:pointer;
  transition:background 150ms cubic-bezier(.23,1,.32,1), color 150ms ease,
             transform 120ms cubic-bezier(.23,1,.32,1); }}
.seg:hover {{ color:var(--ink); }}
.seg:active {{ transform:scale(.97); }}
.seg.is-on {{ background:var(--raised); color:var(--ink);
  box-shadow:inset 0 1px 0 var(--hair-2), 0 1px 2px rgba(0,0,0,.28); }}
.ctl-note {{ font-size:12.5px; color:var(--mut); margin:0 0 0 auto; }}
.pane {{ display:none; }} .pane.is-on {{ display:block; }}

/* checks */
.checks {{ list-style:none; margin:0; padding:0; display:grid; gap:2px; }}
.chk {{ padding:15px 0; border-bottom:1px solid var(--hair); }}
.chk:last-child {{ border-bottom:0; }}
.chk-top {{ display:flex; align-items:center; gap:12px; margin-bottom:9px; }}
.chk-top code {{ font-size:12.5px; color:var(--ink-2); }}
.tag {{ margin-left:auto; font-size:10.5px; letter-spacing:.09em; text-transform:uppercase;
  padding:3px 9px; border-radius:999px; border:1px solid var(--hair-2); color:var(--mut); }}
.s-warn .tag {{ color:var(--warn); border-color:color-mix(in srgb,var(--warn) 40%,transparent); }}
.s-fail .tag {{ color:var(--crit); border-color:color-mix(in srgb,var(--crit) 45%,transparent); }}
.s-pass .tag {{ color:var(--good); border-color:color-mix(in srgb,var(--good) 40%,transparent); }}
.chk-bot {{ display:flex; align-items:baseline; gap:20px; margin-top:9px; font-size:12px;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; color:var(--mut); }}
.chk-bot .big {{ font-size:15px; color:var(--ink); letter-spacing:-.02em; }}
svg.bullet {{ display:block; }}
svg.bullet .trk {{ fill:var(--grid); }}
svg.bullet .fil {{ transition:width 420ms cubic-bezier(.23,1,.32,1); }}
.s-pass svg.bullet .fil {{ fill:var(--good); }}
.s-warn svg.bullet .fil {{ fill:var(--warn); }}
.s-fail svg.bullet .fil {{ fill:var(--crit); }}
svg.bullet line {{ stroke:var(--ink); stroke-width:2; vector-effect:non-scaling-stroke; }}
svg.bullet .t-warn line {{ opacity:.4; stroke-dasharray:3 3; }}
svg.bullet .t-fail line {{ opacity:.75; }}

/* charts */
svg.chart {{ display:block; overflow:visible; }}
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
  svg.chart:hover .row {{ opacity:.38; }}
  svg.chart .row:hover {{ opacity:1; }}
}}
.legend {{ display:flex; align-items:center; gap:9px; font-size:12.5px; color:var(--ink-2);
  margin:0 0 14px; }}
.key {{ width:11px; height:11px; border-radius:3px; display:inline-block; }}
.k1 {{ background:var(--s1); }} .k2 {{ background:var(--s2); margin-left:16px; }}
.empty {{ color:var(--mut); font-size:13px; margin:0; }}

/* tables */
table {{ border-collapse:collapse; width:100%; font-size:12.5px; }}
th, td {{ padding:9px 12px; text-align:right; white-space:nowrap; }}
thead th {{ font-size:10.5px; letter-spacing:.09em; text-transform:uppercase;
  color:var(--mut); font-weight:600; border-bottom:1px solid var(--axis); }}
tbody th {{ text-align:left; font-weight:500; color:var(--ink);
  font-family:ui-sans-serif,system-ui,sans-serif; }}
tbody tr {{ transition:background 130ms ease; }}
tbody tr:hover {{ background:var(--raised); }}
tbody tr + tr th, tbody tr + tr td {{ border-top:1px solid var(--hair); }}
.scroll-x {{ overflow-x:auto; }}

.note {{ font-size:13.5px; color:var(--ink-2); border-left:2px solid var(--axis);
  padding-left:15px; margin:18px 0 0; max-width:66ch; }}
footer {{ margin-top:72px; padding-top:22px; border-top:1px solid var(--hair);
  font-size:12px; color:var(--mut); line-height:1.7; }}
footer code {{ color:var(--ink-2); }}

.rise {{ animation:rise 460ms cubic-bezier(.23,1,.32,1) backwards; }}
.rise:nth-child(2) {{ animation-delay:50ms; }}
.rise:nth-child(3) {{ animation-delay:100ms; }}
.rise:nth-child(4) {{ animation-delay:150ms; }}
@keyframes rise {{ from {{ opacity:0; transform:translateY(10px); }} }}
@media (prefers-reduced-motion:reduce) {{
  .rise {{ animation:none; }} html {{ scroll-behavior:auto; }}
  *, *::before, *::after {{ transition-duration:1ms !important; }}
}}
@media (max-width:760px) {{
  .wrap {{ padding:0 16px 64px; }}
  .ctl-note {{ margin-left:0; }}
  .verdict h1 {{ font-size:clamp(44px,15vw,64px); }}
}}
@media print {{ body {{ background:#fff; }} .bar,.controls {{ display:none; }}
  .pane {{ display:block !important; }} .card {{ break-inside:avoid; }} }}
</style>
</head>
<body>

<div class="bar">
  <span class="brand">fairness&#8209;gate <span>/ ACS income</span></span>
  <span class="fp">{esc(result.get('pipeline_fingerprint', 'n/a'))}</span>
  <button type="button" class="tgl" id="themer" aria-label="Switch colour theme">Light</button>
</div>

<div class="wrap">

  <p class="eyebrow">Policy gate &#183; ACS income model &#183; test year {esc(d['test_year'])}</p>

  <div class="bento">

    <div class="cell tall c5 verdict-cell v-{esc(v)}">
      <div><p class="cell-k">Verdict</p>
        <h1>{esc(STATUS_LABEL[v].upper())}</h1></div>
      <p class="tally">{counts['pass']} pass &#183; {counts['warn']} warn &#183;
        {counts['fail']} fail</p>
    </div>

    <div class="cell c7">
      <p class="cell-k">What the gap means</p>
      <p class="lede">Among people who genuinely earn above the threshold, the model
        finds <b>{pct(float(best['tpr']))}</b> of the {esc(best['group'])} group and
        <b>{pct(float(worst['tpr']))}</b> of the {esc(worst['group'])} group. Accuracy
        is nearly flat across groups, which is exactly why accuracy is the wrong thing
        to look at.</p>
    </div>

    <div class="cell stat c3"><p class="cell-k">True positive gap</p>
      <div class="v">{fmt(race['tpr_gap'])}</div>
      <div class="c">target 0.150 &#183; limit 0.350</div></div>
    <div class="cell stat c4"><p class="cell-k">Base rate gap</p>
      <div class="v">{fmt(race['base_rate_gap'])}</div>
      <div class="c">the part the data itself explains</div></div>

    <div class="cell c12">
      <p class="cell-k">Policy checks</p>
      <p class="h2sub">Declared in <code>policy.yaml</code>. The solid tick is the limit
        that fails the build; the dashed tick is the target being aimed at. The distance
        between them is deliberate.</p>
      <ul class="checks">{_checks(result)}</ul>
    </div>

    <div class="cell c12">
      <p class="cell-k">Error rates by group</p>
      {_controls()}
      <p class="legend"><span class="key k1"></span>Found, of those who qualify
        <span class="key k2"></span>Wrongly flagged</p>
      {_panes(tables, rate_chart)}
    </div>

    <div class="cell c7">
      <p class="cell-k">Predicted against observed</p>
      <p class="legend"><span class="key k1"></span>Observed
        <span class="key k2"></span>Predicted</p>
      {_panes(tables, calib_chart)}
      <p class="note">A calibrated model puts the two dots on top of each other. The
        number beside each pair is the distance.</p>
    </div>

    <div class="cell c5 scroll-x">
      <p class="cell-k">Across splits</p>
      <table>
        <thead><tr><th scope="col">Split</th><th scope="col">n</th>
          <th scope="col">AUC</th><th scope="col">ECE</th>
          <th scope="col">Acc</th><th scope="col">Base</th></tr></thead>
        <tbody>{_splits_compact(result)}</tbody>
      </table>
      <p class="note">Validation ECE is near zero because the calibrator was fitted
        there. It is not a result. Error grows on the later year and again on the
        held-out states.</p>
    </div>

    <div class="cell c8 scroll-x">
      <p class="cell-k">Every reported group</p>
      {_panes(tables, group_table)}
    </div>

    <div class="cell c4 stat">
      <p class="cell-k">Scale</p>
      <div class="v">{ts['n']:,}</div>
      <div class="c">people in the test year. {race['groups_suppressed']} group sits
        below the 500-person reporting floor and is measured but never concluded
        from.</div>
      <div class="v" style="margin-top:22px">{fmt(ts['auc'], 3)}</div>
      <div class="c">AUC, against a majority baseline of
        {fmt(ts['majority_baseline'], 3)}</div>
    </div>

  </div>

  <footer>
    Generated {esc(result['generated_at'])} from commit
    <code>{esc(result['git_sha'])}</code>, pipeline
    <code>{esc(result.get('pipeline_fingerprint', 'n/a'))}</code>.
    Written by <code>scripts/run_audit.py</code> and verified against the audit on every
    commit, so editing this page by hand fails the build.
    <br>Not a deployable system. No decision about any person should be made with it.
  </footer>
</div>

<script>
// Two behaviours, both tiny and both optional: the page is complete without them.
(function () {{
  var root = document.documentElement;
  var btn = document.getElementById('themer');
  function label() {{ btn.textContent = root.dataset.theme === 'dark' ? 'Light' : 'Dark'; }}
  try {{
    var saved = localStorage.getItem('fg-theme');
    if (saved) root.dataset.theme = saved;
  }} catch (e) {{ /* private mode: keep the default */ }}
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
