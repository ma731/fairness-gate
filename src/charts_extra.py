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
    5: "Native tribes", 6: "Asian", 7: "Pacific Islander",
    8: "Some other race", 9: "Two or more",
}
SEX_SHORT = {1: "Male", 2: "Female"}


def esc(t: object) -> str:
    return html.escape(str(t), quote=True)


def short(code: int, attribute: str) -> str:
    if attribute == "RACExSEX":
        race = RACE_SHORT.get(code // 10, f"race {code // 10}")
        sex = "M" if code % 10 == 1 else "F"
        return f"{race} {sex}"
    return (RACE_SHORT if attribute == "RAC1P" else SEX_SHORT).get(code, f"code {code}")


def _dist(result: dict, split: str, attribute: str) -> dict:
    return result.get("distributions", {}).get(split, {}).get(attribute, {})


# --------------------------------------------------------------------------- #
# Sankey: where 600,000 people actually go
# --------------------------------------------------------------------------- #
def sankey(result: dict, split: str = "test", attribute: str = "RAC1P") -> str:
    """Everyone in the split, flowing from truth into the model's decision.

    A flow diagram is right here because the quantity is conserved: every person leaves
    one box and arrives in exactly one other.

    Each node box is sized as the sum of the ribbons that meet it, never computed
    independently. Sizing them separately is what let rounding put a ribbon edge a
    fraction outside its box, which is visible at this scale and reads as a broken
    chart. Deriving the box from its ribbons makes the two agree by construction.
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
    x0, x1, bw = 240, 620, 22
    top, gap = 36, 30
    span = h - top - 48 - gap          # height available to the ribbons themselves

    def px(n: int) -> float:
        return n / total * span

    h_tp, h_fn, h_fp, h_tn = px(tp), px(fn), px(fp), px(tn)

    # Boxes are the sum of their own ribbons. No independent rounding anywhere.
    left = [("Genuinely qualify", tp + fn, h_tp + h_fn),
            ("Do not qualify", fp + tn, h_fp + h_tn)]
    right = [("Model selects", tp + fp, h_tp + h_fp),
             ("Model rejects", fn + tn, h_fn + h_tn)]

    ly0 = top
    ly1 = ly0 + left[0][2] + gap
    ry0 = top
    ry1 = ry0 + right[0][2] + gap

    def ribbon(ay, by, thick, cls, label):
        ax, bx = x0 + bw, x1
        m = (ax + bx) / 2
        return (
            f'<path class="rb {cls}" d="M{ax},{ay:.2f} '
            f"C{m},{ay:.2f} {m},{by:.2f} {bx},{by:.2f} "
            f"L{bx},{by + thick:.2f} "
            f"C{m},{by + thick:.2f} {m},{ay + thick:.2f} {ax},{ay + thick:.2f} Z\">"
            f"<title>{label}</title></path>"
        )

    flows = (
        ribbon(ly0, ry0, h_tp, "ok", f"{tp:,} qualify and are selected")
        + ribbon(ly0 + h_tp, ry1, h_fn, "bad", f"{fn:,} qualify and are OVERLOOKED")
        + ribbon(ly1, ry0 + h_tp, h_fp, "warnrb",
                 f"{fp:,} do not qualify but are selected")
        + ribbon(ly1 + h_fp, ry1 + h_fn, h_tn, "ok",
                 f"{tn:,} do not qualify and are not selected")
    )

    def node(x, y, height, n, label, anchor, tx):
        cy = y + height / 2
        return (
            f'<rect class="nd" x="{x}" y="{y:.2f}" width="{bw}" '
            f'height="{height:.2f}" rx="4"/>'
            f'<text class="nl" x="{tx}" y="{cy - 13:.1f}" text-anchor="{anchor}">'
            f"{label}</text>"
            f'<text class="nn" x="{tx}" y="{cy + 9:.1f}" text-anchor="{anchor}">'
            f"{n:,}</text>"
            f'<text class="ns" x="{tx}" y="{cy + 27:.1f}" text-anchor="{anchor}">'
            f"{n / total:.1%}</text>"
        )

    nodes = (
        node(x0, ly0, left[0][2], left[0][1], left[0][0], "end", x0 - 18)
        + node(x0, ly1, left[1][2], left[1][1], left[1][0], "end", x0 - 18)
        + node(x1, ry0, right[0][2], right[0][1], right[0][0], "start", x1 + bw + 18)
        + node(x1, ry1, right[1][2], right[1][1], right[1][0], "start", x1 + bw + 18)
    )

    return (
        f'<svg class="chart sankey" viewBox="0 0 {w} {h}" width="100%" role="img" '
        f'aria-label="flow of {total:,} people from truth to decision">'
        f"{flows}{nodes}"
        f'<text class="axt" x="{x0 - 18}" y="{h - 16}" text-anchor="end">Truth</text>'
        f'<text class="axt" x="{x1 + bw + 18}" y="{h - 16}">Decision</text></svg>'
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
    """One small reliability panel per group, against the diagonal.

    Eight curves on one pair of axes in a single colour was unreadable: the lines
    crossed, the labels collided, and nothing could be traced. Eight categorical hues
    would not have fixed it either, because eight hues cannot clear the colour-vision
    separation floor when every pair can appear together. Faceting is the sanctioned
    answer to "too many series", so each group gets its own panel on identical axes and
    comparison happens between panels rather than inside one.
    """
    d = _dist(result, split, attribute)
    curves = [c for c in (d.get("calibration") or []) if len(c.get("points", [])) >= 2]
    if not curves:
        return '<p class="empty">No distribution data.</p>'

    cols = 4 if len(curves) > 4 else max(1, len(curves))
    rows = (len(curves) + cols - 1) // cols
    cw, ch, pad = 210, 210, 34
    gx, gy = 22, 40
    w = cols * cw + (cols - 1) * gx
    h = rows * ch + (rows - 1) * gy + 26

    out = ""
    for i, c in enumerate(curves):
        ox = (i % cols) * (cw + gx)
        oy = (i // cols) * (ch + gy)
        p0, p1 = ox + pad, ox + cw - 10
        q0, q1 = oy + ch - pad, oy + 10
        sx = p1 - p0
        sy = q0 - q1

        def X(v, p0=p0, sx=sx):
            return round(p0 + v * sx, 1)

        def Y(v, q0=q0, sy=sy):
            return round(q0 - v * sy, 1)

        frame = (
            f'<rect class="facet" x="{ox}" y="{oy}" width="{cw}" height="{ch}" rx="10"/>'
            f'<line class="diag" x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}"/>'
        )
        for t in (0, 0.5, 1.0):
            frame += (
                f'<line class="gr" x1="{X(t)}" y1="{Y(0)}" x2="{X(t)}" y2="{Y(1)}"/>'
                f'<line class="gr" x1="{X(0)}" y1="{Y(t)}" x2="{X(1)}" y2="{Y(t)}"/>'
            )
        pts = c["points"]
        poly = " ".join(f"{X(p['predicted'])},{Y(p['observed'])}" for p in pts)
        dots = "".join(
            f'<circle class="cdot" cx="{X(p["predicted"])}" cy="{Y(p["observed"])}" '
            f'r="3"><title>predicted {p["predicted"]:.2f}, observed '
            f'{p["observed"]:.2f}, {p["n"]:,} people</title></circle>'
            for p in pts
        )
        # worst vertical distance from the diagonal, which is the thing to compare
        worst = max(abs(p["observed"] - p["predicted"]) for p in pts)
        out += (
            f'<g class="row">{frame}<polyline class="cal" points="{poly}"/>{dots}'
            f'<text class="facet-t" x="{ox + 12}" y="{oy + ch + 19}">'
            f'{esc(short(c["code"], attribute))}</text>'
            f'<text class="facet-n" x="{ox + cw - 10}" y="{oy + ch + 19}" '
            f'text-anchor="end">max gap {worst:.2f}</text></g>'
        )

    return (
        f'<svg class="chart relf" viewBox="0 0 {w} {h}" width="100%" role="img" '
        f'aria-label="reliability per group, predicted against observed">{out}</svg>'
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


# --------------------------------------------------------------------------- #
# Intervals: what the precision is actually worth
# --------------------------------------------------------------------------- #
def interval_chart(result: dict, attribute: str = "RAC1P") -> str:
    """Per-group recall with its 95% interval, ordered by how uncertain it is.

    The chart that stops a reader trusting three decimal places equally. Every other
    chart here prints rates to the same precision whatever the group size; this one
    shows what that precision is worth. The widest bars are the smallest groups, and
    they sit at the top so the caveat arrives before the ranking.
    """
    u = (result.get("uncertainty") or {}).get(attribute) or {}
    groups = u.get("tpr_by_group") or []
    if not groups:
        return '<p class="empty">No interval data.</p>'

    row_h, lab_w, pad_r, plot_w = 46, 190, 150, 520
    h = len(groups) * row_h + 48
    w = lab_w + plot_w + pad_r
    lo = min(min(g["lo"] for g in groups), 0.45)
    hi = max(max(g["hi"] for g in groups), 0.9)
    span = hi - lo or 1.0

    def X(v: float) -> float:
        return round(lab_w + (v - lo) / span * plot_w, 1)

    grid = ""
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        v = lo + span * t
        gx = X(v)
        grid += (
            f'<line class="gr" x1="{gx}" y1="28" x2="{gx}" y2="{h - 18}"/>'
            f'<text class="ax" x="{gx}" y="18" text-anchor="middle">{v:.2f}</text>'
        )

    rows = ""
    for i, g in enumerate(groups):
        cy = 44 + i * row_h
        x0, x1, xc = X(g["lo"]), X(g["hi"]), X(g["tpr"])
        rows += (
            f'<g class="row">'
            f'<text class="gl" x="{lab_w - 16}" y="{cy - 3}" text-anchor="end">'
            f'{esc(short(g["code"], attribute))}'
            f'<title>{esc(g["group"])}</title></text>'
            f'<text class="gn" x="{lab_w - 16}" y="{cy + 14}" text-anchor="end">'
            f'{g["qualifying"]:,} qualify</text>'
            f'<line class="whisk" x1="{x0}" y1="{cy}" x2="{x1}" y2="{cy}"/>'
            f'<line class="cap" x1="{x0}" y1="{cy - 7}" x2="{x0}" y2="{cy + 7}"/>'
            f'<line class="cap" x1="{x1}" y1="{cy - 7}" x2="{x1}" y2="{cy + 7}"/>'
            f'<circle class="pt" cx="{xc}" cy="{cy}" r="5">'
            f'<title>{esc(g["group"])}: {g["tpr"]:.3f}, 95% interval '
            f'[{g["lo"]:.3f}, {g["hi"]:.3f}] on {g["qualifying"]:,} qualifying people'
            f"</title></circle>"
            f'<text class="vl" x="{x1 + 13}" y="{cy + 4}">&#177;{g["width"] / 2:.3f}</text>'
            f"</g>"
        )

    return (
        f'<svg class="chart ivl" viewBox="0 0 {w} {h}" width="100%" role="img" '
        f'aria-label="recall with 95 percent intervals by {esc(attribute)}">'
        f"{grid}{rows}</svg>"
    )


# --------------------------------------------------------------------------- #
# Intersection matrix: where two attributes cross
# --------------------------------------------------------------------------- #
def intersection_matrix(table, metric: str = "tpr") -> str:
    """Race down, sex across, one cell per combination.

    A grid, because the data is a grid. Auditing race and sex separately can leave two
    acceptable marginals sitting on top of a cell that is much worse than either, and a
    pair of bar charts cannot show that because it never puts the two axes together.

    Cells below the reporting floor are drawn hatched rather than dropped. Their absence
    is a finding: the groups most likely to be harmed are the ones a survey is least
    likely to have enough of to say anything about.
    """
    sub = table[table["attribute"] == "RACExSEX"]
    if sub.empty:
        return '<p class="empty">No intersectional data.</p>'

    races, seen = [], set()
    for code in sorted(sub["code"].astype(int)):
        r = code // 10
        if r not in seen:
            seen.add(r)
            races.append(r)
    sexes = [1, 2]

    cell_w, cell_h, lab_w, top = 210, 62, 250, 52
    w = lab_w + cell_w * len(sexes) + 150
    h = top + cell_h * len(races) + 26

    by_code = {int(r["code"]): r for _, r in sub.iterrows()}
    vals = [float(r[metric]) for _, r in sub.iterrows() if r["reportable"]]
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    rng = (hi - lo) or 1.0

    head = "".join(
        f'<text class="ax" x="{lab_w + i * cell_w + cell_w / 2}" y="{top - 18}" '
        f'text-anchor="middle">{"Male" if sx == 1 else "Female"}</text>'
        for i, sx in enumerate(sexes)
    )

    body = ""
    for ri, race in enumerate(races):
        y = top + ri * cell_h
        body += (
            f'<text class="gl" x="{lab_w - 18}" y="{y + cell_h / 2 + 4}" '
            f'text-anchor="end">{esc(RACE_SHORT.get(race, str(race)))}</text>'
        )
        for si, sx in enumerate(sexes):
            x = lab_w + si * cell_w
            row = by_code.get(race * 10 + sx)
            if row is None:
                continue
            if not bool(row["reportable"]):
                body += (
                    f'<g class="row"><rect class="cell muted-cell" x="{x + 3}" '
                    f'y="{y + 3}" width="{cell_w - 6}" height="{cell_h - 6}" rx="6">'
                    f'<title>{esc(row["group"])}: {int(row["n"]):,} people, below the '
                    f"500-person reporting floor</title></rect>"
                    f'<text class="cv sup" x="{x + cell_w / 2}" y="{y + cell_h / 2 + 4}" '
                    f'text-anchor="middle">too few ({int(row["n"]):,})</text></g>'
                )
                continue
            v = float(row[metric])
            # low recall is the bad end, so invert for the shading intensity
            heat = 1 - (v - lo) / rng
            # Light ink on a bright acid cell fails contrast, so the label flips to
            # dark once the fill is strong enough to need it.
            ink = " on-bright" if heat > 0.45 else ""
            body += (
                f'<g class="row"><rect class="cell bad" x="{x + 3}" y="{y + 3}" '
                f'width="{cell_w - 6}" height="{cell_h - 6}" rx="6" '
                f'style="--v:{heat:.3f}">'
                f'<title>{esc(row["group"])}: recall {v:.3f} on {int(row["n"]):,} '
                f"people</title></rect>"
                f'<text class="cv{ink}" x="{x + cell_w / 2}" y="{y + cell_h / 2 - 1}" '
                f'text-anchor="middle">{v:.3f}</text>'
                f'<text class="cn{ink}" x="{x + cell_w / 2}" y="{y + cell_h / 2 + 15}" '
                f'text-anchor="middle">{int(row["n"]):,}</text></g>'
            )

    return (
        f'<svg class="chart heat xmat" viewBox="0 0 {w} {h}" width="100%" role="img" '
        f'aria-label="recall by race and sex">{head}{body}</svg>'
    )
