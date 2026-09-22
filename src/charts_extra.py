"""The chart forms that need distribution data rather than a summary row.

Every form here was chosen because it answers a question the summary tables cannot, not
because it looks impressive. Where a form would have been decoration it was left out: a
chord diagram encodes pairwise flow between entities and a transit map encodes network
topology, and this data has neither, so drawing one would be an ornament wearing the
costume of evidence.

All rendering is server-side SVG with deterministic geometry, because the gate
regenerates this page and requires a byte-identical match.
"""

from __future__ import annotations

import html

RACE_SHORT = {
    1: "White", 2: "Black", 3: "Am. Indian", 4: "Alaska Native",
    5: "Am. Indian / Alaska Native", 6: "Asian", 7: "Pacific Islander",
    8: "Some other race", 9: "Two or more",
}
SEX_SHORT = {1: "Male", 2: "Female"}


def esc(t: object) -> str:
    return html.escape(str(t), quote=True)


def short(code: int, attribute: str) -> str:
    return (RACE_SHORT if attribute == "RAC1P" else SEX_SHORT).get(code, f"code {code}")


def _dist(result: dict, split: str, attribute: str) -> dict:
    return result.get("distributions", {}).get(split, {}).get(attribute, {})


# --------------------------------------------------------------------------- #
# Sankey: where 600,000 people actually go
# --------------------------------------------------------------------------- #
def sankey(result: dict, split: str = "test", attribute: str = "RAC1P") -> str:
    """Everyone in the split, flowing from truth into the model's decision.

    A flow diagram is right here because the quantity is conserved: every person leaves
    one box and arrives in exactly one other. The two error ribbons are the ones that
    matter, and they are drawn in the error colour so they read before the labels do.
    """
    d = _dist(result, split, attribute)
    conf = d.get("confusion") or []
    if not conf:
        return '<p class="empty">No distribution data.</p>'

    tp = sum(c["tp"] for c in conf)
    fn = sum(c["fn"] for c in conf)
    fp = sum(c["fp"] for c in conf)
    tn = sum(c["tn"] for c in conf)
    total = tp + fn + fp + tn
    if not total:
        return '<p class="empty">No distribution data.</p>'

    w, h = 900, 440
    x0, x1 = 240, 620
    bw = 22
    top, usable, gap = 34, h - 96, 26

    qual, not_qual = tp + fn, fp + tn
    sel, not_sel = tp + fp, fn + tn

    def hgt(n: int) -> float:
        return round(n / total * (usable - gap), 1)

    ly0 = top
    ly1 = top + hgt(qual) + gap
    ry0 = top
    ry1 = top + hgt(sel) + gap

    def ribbon(ax, ay, bx, by, thick, cls, label):
        mid = (ax + bx) / 2
        return (
            f'<path class="rb {cls}" d="M{ax},{ay} C{mid},{ay} {mid},{by} {bx},{by} '
            f'L{bx},{by + thick} C{mid},{by + thick} {mid},{ay + thick} '
            f'{ax},{ay + thick} Z"><title>{label}</title></path>'
        )

    flows = ""
    # qualifies -> selected (true positives), then -> not selected (the overlooked)
    flows += ribbon(x0 + bw, ly0, x1, ry0, hgt(tp), "ok",
                    f"{tp:,} qualify and are selected")
    flows += ribbon(x0 + bw, ly0 + hgt(tp), x1, ry1, hgt(fn), "bad",
                    f"{fn:,} qualify and are OVERLOOKED")
    # does not qualify -> selected (false positives), then -> not selected
    flows += ribbon(x0 + bw, ly1, x1, ry0 + hgt(tp), hgt(fp), "warnrb",
                    f"{fp:,} do not qualify but are selected")
    flows += ribbon(x0 + bw, ly1 + hgt(fp), x1, ry1 + hgt(fn), hgt(tn), "ok",
                    f"{tn:,} do not qualify and are not selected")

    def node(x, y, n, label, sub, anchor, tx):
        # Label block is centred on the node so it never floats away from its box.
        cy = y + hgt(n) / 2
        return (
            f'<rect class="nd" x="{x}" y="{y}" width="{bw}" height="{hgt(n)}" rx="4"/>'
            f'<text class="nl" x="{tx}" y="{cy - 14:.1f}" text-anchor="{anchor}">'
            f"{label}</text>"
            f'<text class="nn" x="{tx}" y="{cy + 8:.1f}" text-anchor="{anchor}">'
            f"{n:,}</text>"
            f'<text class="ns" x="{tx}" y="{cy + 26:.1f}" text-anchor="{anchor}">'
            f"{sub}</text>"
        )

    nodes = (
        node(x0, ly0, qual, "Genuinely qualify", f"{qual / total:.1%} of everyone",
             "end", x0 - 16)
        + node(x0, ly1, not_qual, "Do not qualify", f"{not_qual / total:.1%}",
               "end", x0 - 16)
        + node(x1, ry0, sel, "Model selects", f"{sel / total:.1%}", "start", x1 + bw + 16)
        + node(x1, ry1, not_sel, "Model rejects", f"{not_sel / total:.1%}", "start",
               x1 + bw + 16)
    )

    return (
        f'<svg class="chart sankey" viewBox="0 0 {w} {h}" width="100%" role="img" '
        f'aria-label="flow of {total:,} people from truth to decision">'
        f"{flows}{nodes}"
        f'<text class="axt" x="{x0 - 16}" y="{h - 18}" text-anchor="end">Truth</text>'
        f'<text class="axt" x="{x1 + bw + 16}" y="{h - 18}">Decision</text></svg>'
    )


# --------------------------------------------------------------------------- #
# Heatmap: error composition per group
# --------------------------------------------------------------------------- #
def confusion_heatmap(result: dict, split: str = "test",
                      attribute: str = "RAC1P") -> str:
    """Each group's outcomes as a row of four cells, shaded by share within the group.

    Rows are normalised within the group on purpose: the question is composition, not
    size, and an unnormalised heatmap would only ever show that one group is large.
    """
    d = _dist(result, split, attribute)
    conf = d.get("confusion") or []
    if not conf:
        return '<p class="empty">No distribution data.</p>'

    cols = [("tp", "Found"), ("fn", "Overlooked"), ("fp", "Wrongly picked"),
            ("tn", "Correctly passed")]
    cell_w, cell_h, lab_w, top = 150, 46, 250, 40
    w = lab_w + cell_w * len(cols) + 20
    h = top + cell_h * len(conf) + 16

    head = "".join(
        f'<text class="ax" x="{lab_w + i * cell_w + cell_w / 2}" y="{top - 14}" '
        f'text-anchor="middle">{esc(label)}</text>'
        for i, (_, label) in enumerate(cols)
    )

    rows = ""
    for r, c in enumerate(sorted(conf, key=lambda c: -(c["tp"] + c["fn"] + c["fp"] + c["tn"]))):
        tot = c["tp"] + c["fn"] + c["fp"] + c["tn"]
        y = top + r * cell_h
        rows += (
            f'<text class="gl" x="{lab_w - 16}" y="{y + cell_h / 2 + 4}" '
            f'text-anchor="end">{esc(short(c["code"], attribute))}</text>'
        )
        for i, (key, label) in enumerate(cols):
            share = c[key] / tot if tot else 0.0
            x = lab_w + i * cell_w
            bad = key in ("fn", "fp")
            rows += (
                f'<g class="row"><rect class="cell {"bad" if bad else "ok"}" x="{x + 2}" '
                f'y="{y + 2}" width="{cell_w - 4}" height="{cell_h - 4}" rx="5" '
                f'style="--v:{share:.4f}"><title>{esc(short(c["code"], attribute))}: '
                f'{c[key]:,} {label.lower()} ({share:.1%})</title></rect>'
                f'<text class="cv" x="{x + cell_w / 2}" y="{y + cell_h / 2 + 4}" '
                f'text-anchor="middle">{share:.0%}</text></g>'
            )

    return (
        f'<svg class="chart heat" viewBox="0 0 {w} {h}" width="100%" role="img" '
        f'aria-label="outcome composition by {esc(attribute)}">{head}{rows}</svg>'
    )


# --------------------------------------------------------------------------- #
# Treemap: who is actually in this data
# --------------------------------------------------------------------------- #
def treemap(result: dict, split: str = "test", attribute: str = "RAC1P") -> str:
    """Group sizes as nested area. Squarified rows, deterministic.

    This is the context chart. Every rate on the page is computed on wildly different
    denominators, and area is the honest way to show that before anyone reads a rate.
    """
    d = _dist(result, split, attribute)
    conf = d.get("confusion") or []
    if not conf:
        return '<p class="empty">No distribution data.</p>'

    items = sorted(
        ((c["code"], c["tp"] + c["fn"] + c["fp"] + c["tn"]) for c in conf),
        key=lambda t: -t[1],
    )
    total = sum(n for _, n in items)
    w, h = 880, 380

    # Squarified treemap. Slice-and-dice produced unreadable slivers at the tail, which
    # is exactly the failure mode squarify exists to fix: it keeps cells near square by
    # committing a row only when adding another tile would make the worst aspect ratio
    # worse.
    def worst(row, length, scale):
        if not row or length <= 0:
            return float("inf")
        s_ = sum(row) * scale
        side = s_ / length
        if side <= 0:
            return float("inf")
        return max(max(r * scale / side / side * length for r in row),
                   max(side * side * length / (r * scale) for r in row))

    rects, row = [], []
    x, y, rw, rh = 0.0, 0.0, float(w), float(h)
    scale = (w * h) / total if total else 1.0
    queue = list(items)

    def layout_row(row_items, x, y, rw, rh):
        """Place one committed row along the shorter side and return the leftover box."""
        area = sum(n for _, n in row_items) * scale
        out = []
        if rw >= rh:
            band = area / rh if rh else 0
            oy = y
            for code, n in row_items:
                ch = (n * scale / band) if band else 0
                out.append((code, n, x, oy, band, ch))
                oy += ch
            return out, x + band, y, rw - band, rh
        band = area / rw if rw else 0
        ox = x
        for code, n in row_items:
            cw = (n * scale / band) if band else 0
            out.append((code, n, ox, y, cw, band))
            ox += cw
        return out, x, y + band, rw, rh - band

    while queue:
        nxt = queue[0]
        side = min(rw, rh)
        cur = [n for _, n in row]
        if row and worst(cur + [nxt[1]], side, scale) > worst(cur, side, scale):
            placed, x, y, rw, rh = layout_row(row, x, y, rw, rh)
            rects += placed
            row = []
            continue
        row.append(queue.pop(0))
    if row:
        placed, x, y, rw, rh = layout_row(row, x, y, rw, rh)
        rects += placed

    out = ""
    for code, n, cx, cy, cw, ch in rects:
        label = short(code, attribute)
        big = cw > 96 and ch > 46
        out += (
            f'<g class="row"><rect class="tm" x="{cx + 2:.1f}" y="{cy + 2:.1f}" '
            f'width="{max(cw - 4, 1):.1f}" height="{max(ch - 4, 1):.1f}" rx="7" '
            f'style="--v:{n / total:.4f}"><title>{esc(label)}: {n:,} people '
            f'({n / total:.1%})</title></rect>'
        )
        if big:
            out += (
                f'<text class="tml" x="{cx + 15:.1f}" y="{cy + 28:.1f}">'
                f"{esc(label)}</text>"
                f'<text class="tmn" x="{cx + 15:.1f}" y="{cy + 49:.1f}">{n:,}</text>'
            )
        out += "</g>"

    return (
        f'<svg class="chart tree" viewBox="0 0 {w} {h}" width="100%" role="img" '
        f'aria-label="group sizes by {esc(attribute)}">{out}</svg>'
    )


# --------------------------------------------------------------------------- #
# Reliability diagram
# --------------------------------------------------------------------------- #
def reliability(result: dict, split: str = "test", attribute: str = "RAC1P") -> str:
    """Predicted against observed, per bin, per group. The diagonal is perfection."""
    d = _dist(result, split, attribute)
    curves = d.get("calibration") or []
    if not curves:
        return '<p class="empty">No distribution data.</p>'

    w, h, pad = 560, 560, 56
    px = w - pad * 2

    def X(v):
        return round(pad + v * px, 1)

    def Y(v):
        return round(h - pad - v * px, 1)

    grid = "".join(
        f'<line class="gr" x1="{X(t)}" y1="{Y(0)}" x2="{X(t)}" y2="{Y(1)}"/>'
        f'<line class="gr" x1="{X(0)}" y1="{Y(t)}" x2="{X(1)}" y2="{Y(t)}"/>'
        f'<text class="ax" x="{X(t)}" y="{Y(0) + 20}" text-anchor="middle">{t:.2g}</text>'
        f'<text class="ax" x="{X(0) - 10}" y="{Y(t) + 4}" text-anchor="end">{t:.2g}</text>'
        for t in (0, 0.25, 0.5, 0.75, 1.0)
    )
    diag = (f'<line class="diag" x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}"/>'
            f'<text class="axt" x="{X(0.74)}" y="{Y(0.79)}">perfectly calibrated</text>')

    lines = ""
    for c in curves:
        pts = c["points"]
        if len(pts) < 2:
            continue
        dd = " ".join(f"{X(p['predicted'])},{Y(p['observed'])}" for p in pts)
        lines += (
            f'<g class="row"><polyline class="cal" points="{dd}"/>'
            + "".join(
                f'<circle class="cdot" cx="{X(p["predicted"])}" cy="{Y(p["observed"])}" '
                f'r="3.5"><title>{esc(short(c["code"], attribute))}: predicted '
                f'{p["predicted"]:.2f}, observed {p["observed"]:.2f} on {p["n"]:,} '
                f'people</title></circle>'
                for p in pts
            )
            + f'<text class="vl" x="{X(pts[-1]["predicted"]) + 8}" '
              f'y="{Y(pts[-1]["observed"]) + 4}">{esc(short(c["code"], attribute))}</text>'
              "</g>"
        )

    return (
        f'<svg class="chart rel" viewBox="0 0 {w} {h}" width="100%" role="img" '
        f'aria-label="reliability diagram by {esc(attribute)}">{grid}{diag}{lines}'
        f'<text class="axt" x="{X(0.5)}" y="{h - 14}" text-anchor="middle">'
        f"What the model predicted</text>"
        f'<text class="axt" transform="translate(16,{Y(0.5)}) rotate(-90)" '
        f'text-anchor="middle">What actually happened</text></svg>'
    )


# --------------------------------------------------------------------------- #
# Ridgeline
# --------------------------------------------------------------------------- #
def ridgeline(result: dict, split: str = "test", attribute: str = "RAC1P") -> str:
    """Score distributions per group, qualifiers against everyone else.

    Where the two shapes overlap is where the model genuinely cannot tell people apart.
    That overlap being wider for some groups than others is the mechanism behind every
    gap on this page, so it gets its own chart.
    """
    d = _dist(result, split, attribute)
    hists = d.get("histogram") or []
    if not hists:
        return '<p class="empty">No distribution data.</p>'

    bins = d.get("bins", 20)
    lab_w, plot_w, row_h, amp = 210, 620, 74, 62
    w = lab_w + plot_w + 30
    h = len(hists) * row_h + 56

    def path(counts, base_y):
        top = max(counts) or 1
        pts = []
        for i, c in enumerate(counts):
            x = lab_w + (i + 0.5) / bins * plot_w
            pts.append(f"{x:.1f},{base_y - c / top * amp:.1f}")
        return (f"M{lab_w},{base_y} L" + " L".join(pts)
                + f" L{lab_w + plot_w},{base_y} Z")

    rows = ""
    for i, g in enumerate(sorted(hists, key=lambda g: -sum(g["positive"]) - sum(g["negative"]))):
        base = 44 + i * row_h + amp
        rows += (
            f'<g class="row">'
            f'<text class="gl" x="{lab_w - 16}" y="{base - 16}" text-anchor="end">'
            f'{esc(short(g["code"], attribute))}</text>'
            f'<path class="ridge neg" d="{path(g["negative"], base)}">'
            f"<title>does not qualify</title></path>"
            f'<path class="ridge pos" d="{path(g["positive"], base)}">'
            f"<title>qualifies</title></path>"
            f'<line class="gr" x1="{lab_w}" y1="{base}" x2="{lab_w + plot_w}" '
            f'y2="{base}"/></g>'
        )

    ticks = "".join(
        f'<text class="ax" x="{lab_w + t * plot_w}" y="{h - 14}" text-anchor="middle">'
        f"{t:.2g}</text>"
        for t in (0, 0.25, 0.5, 0.75, 1.0)
    )

    return (
        f'<svg class="chart ridge-c" viewBox="0 0 {w} {h}" width="100%" role="img" '
        f'aria-label="predicted score distributions by {esc(attribute)}">{rows}{ticks}'
        f'<text class="axt" x="{lab_w + plot_w / 2}" y="{h - 1}" text-anchor="middle">'
        f"Predicted probability</text></svg>"
    )
