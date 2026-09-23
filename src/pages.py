"""Pages other than the evidence page.

The About page carries the only claims on this site that are not measured. Everything
else is generated from results/audit.json; this is a person explaining why they built
the thing, in their own words, and it is marked as such rather than dressed up as a
finding.
"""

from __future__ import annotations

import html
from pathlib import Path

from src.theme import STYLE

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"


def esc(t: object) -> str:
    return html.escape(str(t), quote=True)


def _nav(active: str, verdict: str, verdict_label: str) -> str:
    # Mirrors dashboard.NAV so the bar does not change shape when you land here.
    items = [
        ("index.html", "The finding", "index.html"),
        ("evidence.html", "Evidence", "evidence.html"),
        ("method.html", "Method", "method.html"),
        ("fix.html", "The fix", "fix.html"),
        ("about.html", "Why", "about.html"),
    ]
    def link(href: str, label: str, key: str) -> str:
        # Built outside the f-string: a backslash inside an f-string expression is
        # a syntax error before Python 3.12, and CI runs 3.11.
        current = ' aria-current="page"' if key == active else ""
        return f'<li><a href="{href}"{current}>{esc(label)}</a></li>'

    links = "".join(link(h, lb, k) for h, lb, k in items)
    return f"""<nav>
  <a class="mark" href="index.html">fairness&#8209;gate <i>/ ACS income</i></a>
  <ul>{links}</ul>
  <div class="right">
    <span class="pill"><i></i>{esc(verdict_label)}</span>
    <button type="button" class="tgl" id="themer" aria-label="Switch colour theme">Light</button>
  </div>
</nav>"""


THEME_SCRIPT = """
(function () {
  var root = document.documentElement;
  var btn = document.getElementById('themer');
  if (!btn) return;
  function label() { btn.textContent = root.dataset.theme === 'dark' ? 'Light' : 'Dark'; }
  try { var saved = localStorage.getItem('fg-theme'); if (saved) root.dataset.theme = saved; }
  catch (e) { /* private mode: keep the default */ }
  label();
  btn.addEventListener('click', function () {
    root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    label();
    try { localStorage.setItem('fg-theme', root.dataset.theme); } catch (e) {}
  });
})();
"""


def about(result: dict) -> str:
    """Why this project exists. Written by Marco, kept in his framing."""
    v = result["policy_verdict"]
    label = {"pass": "Pass", "warn": "Warn", "fail": "Fail"}[v]
    race = next(s for s in result["fairness"]["test"] if s["attribute"] == "RAC1P")

    return f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Why this exists &#183; fairness-gate</title>
<meta name="description" content="Why an audit of algorithmic hiring decisions is worth building, and who built it.">
<style>{STYLE}</style>
</head>
<body class="v-{esc(v)}">

<div class="prog" aria-hidden="true"></div>
<div class="dither" aria-hidden="true"></div>
<svg class="grain" aria-hidden="true" focusable="false">
  <filter id="gr"><feTurbulence type="fractalNoise" baseFrequency="0.86" numOctaves="4"
    stitchTiles="stitch"/><feColorMatrix type="saturate" values="0"/></filter>
  <rect width="100%" height="100%" filter="url(#gr)"/>
</svg>

{_nav("about.html", v, label)}

<header class="hero">
  <div class="mesh" aria-hidden="true"><i class="m1"></i><i class="m2"></i>
    <i class="m3"></i><i class="m4"></i></div>
  <div class="shell">
    <p class="kicker">Why this exists</p>
  </div>
  <div class="mega-wrap bleed">
    <h1 class="mega cut sliced">
      <span>EQUAL ODDS</span>
      <span class="sl sl1" aria-hidden="true">EQUAL ODDS</span>
      <span class="sl sl2" aria-hidden="true">EQUAL ODDS</span>
    </h1>
  </div>
  <div class="shell">
    <p class="lead">Increasingly, the first reader of an application is a model. This is
      an argument that what those models do to people should be measured, published, and
      allowed to fail a build.</p>
  </div>
</header>

<div class="band">
  <div class="shell band-in reveal">
    <p class="kicker">In Marco&#8217;s words</p>
    <h2>Everyone should get the same odds.</h2>
    <p class="say">Everyone should have an equal amount of opportunity, whatever their
      race and whatever their background. Some backgrounds are more privileged than
      others, and pretending otherwise would be silly. But everyone has their struggles,
      and everyone contributes something to a room.</p>
    <p class="say">You need variety in people. You need difference of opinion, and people
      arriving with different ideas, because that is how a group actually finds the best
      answer rather than the most familiar one. A process that quietly filters that
      variety out is not neutral, it is just losing information it will never know it
      lost.</p>
    <p class="say">This matters more every year, because more of the filtering is done by
      a model. Applying for work is already brutal. Nearly every posting wants experience
      you are not allowed to get yet, and the process puts you through a run of tests
      before a person ever looks at you. If a model is deciding which of those
      applications a human even sees, then the one thing it must not be doing is tracking
      race or class. It should be reading the person.</p>
  </div>
</div>

<section>
  <div class="shell reveal">
    <p class="kicker">The hard part</p>
    <h2>Optimising one number is not neutral.</h2>
    <p class="say">There is a real tension here and it is worth stating plainly rather
      than smoothing over. Rank purely on a single measure, a grade or a score, and you
      do not get a neutral result. You get whatever that measure already correlates with,
      including every unequal thing that happened to people long before they applied. The
      output looks objective precisely because nobody chose it on purpose.</p>
    <p class="say">So the honest position is not "remove the protected attribute and stop
      thinking". It is that a balance has to be struck deliberately, in the open, by
      someone who can be argued with. That is what a policy file is: a place to write the
      balance down where changing it leaves a mark.</p>
  </div>
</section>

<div class="band">
  <div class="shell band-in reveal">
    <p class="kicker">What this project does about it</p>
    <h2>Measured, published, and able to fail.</h2>
    <p class="say">This site audits an income model built on real US Census responses.
      It is a benchmark, not a hiring system, because there is no public hiring dataset
      with ground truth. The machinery is the point: a declared threshold, a measured
      audit, documents generated from the run so they cannot drift, and a build that goes
      red when a disparity crosses the line.</p>
    <p class="say">On the current model, a person who genuinely qualifies is
      <b>{race['tpr_gap'] * 100:.0f} percentage points</b> more likely to be missed in one
      racial group than in another. Accuracy is almost identical across those same
      groups. That is the whole argument for measuring more than one number.</p>
    <p class="note">This page is the only one I wrote by hand. Every number on the
      others comes straight from the audit and is checked on every change.</p>
  </div>
</div>

<footer>
  <div class="shell">
    <p><strong>Built by Marco Ortiz Togashi.</strong> Information Management and
      Information Systems, Tsinghua University. Business Analytics and Data Science,
      IE University.</p>
    <p>Not a product. No decision about any person should be made with it.</p>
    <p><a href="https://github.com/ma731/fairness-gate">Source and full results on
      GitHub</a></p>
  </div>
</footer>

<script>{THEME_SCRIPT}</script>
</body>
</html>
"""


def write(result: dict) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "about.html").write_text(about(result), encoding="utf-8")
