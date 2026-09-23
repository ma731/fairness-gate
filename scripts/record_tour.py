"""Record the animated tour at the top of the README from the built site.

Drives a headless browser through the five pages, screenshots the key sections, and
stitches them into docs/assets/tour.gif. Rerun after the site changes.

    python scripts/record_tour.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
OUT = DOCS / "assets" / "tour.gif"

VIEW = {"width": 1024, "height": 640}
WIDTH = 800          # GIF width; height follows the 16:10 viewport
COLOURS = 160
HOLD = 1500          # ms on each section

# page, what to frame (a CSS selector), how long to hold it
SHOTS = [
    ("index.html", "#plain", HOLD),
    ("index.html", "#cost .cost", HOLD),
    ("index.html", "#summary", HOLD + 600),
    ("evidence.html", "#checks .checks", HOLD),
    ("evidence.html", "#intersections svg", HOLD),
    ("evidence.html", "#flow svg", HOLD),
    ("method.html", "#unaware .numbers", HOLD),
    ("method.html", "#families .numbers", HOLD),
    ("fix.html", "#fix .chartbox", HOLD),
    ("fix.html", "#conclusion .steps", HOLD + 300),
]


def grab(page) -> Image.Image:
    shot = Image.open(io.BytesIO(page.screenshot(type="png"))).convert("RGB")
    return shot.resize((WIDTH, round(WIDTH * VIEW["height"] / VIEW["width"])),
                       Image.Resampling.LANCZOS)


def go(page, name: str, section: str | None) -> None:
    page.goto((DOCS / name).as_uri())
    page.wait_for_timeout(700)
    if section:
        # Land just under the fixed nav rather than behind it.
        # Aim at the chart, not the heading above it, landing under the fixed nav.
        page.evaluate(
            "sel => window.scrollTo(0, document.querySelector(sel)"
            ".getBoundingClientRect().top + window.scrollY - 120)",
            section,
        )
        page.wait_for_timeout(600)


def main() -> int:
    for name in {s[0] for s in SHOTS}:
        if not (DOCS / name).exists():
            print(f"missing docs/{name}; run scripts/render_docs.py first", file=sys.stderr)
            return 1

    frames: list[tuple[Image.Image, int]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEW, color_scheme="dark")

        # One held frame for the hero. The point cloud needs a GPU, so headless frames
        # of it all come out the same and only add file size.
        go(page, "index.html", None)
        page.wait_for_timeout(900)
        frames.append((grab(page), HOLD + 700))

        for name, section, hold in SHOTS:
            go(page, name, section)
            frames.append((grab(page), hold))

        # End on the voice panel answering the question the whole project turns on.
        go(page, "index.html", "#cost .cost")
        page.click("#vx-open")
        page.fill("#vx-text", "why did you not fix it")
        page.press("#vx-text", "Enter")
        page.wait_for_timeout(700)
        frames.append((grab(page), 3200))
        browser.close()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    palette = [f.quantize(colors=COLOURS, method=Image.Quantize.MEDIANCUT)
               for f, _ in frames]
    palette[0].save(
        OUT, save_all=True, append_images=palette[1:],
        duration=[d for _, d in frames], loop=0, optimize=True, disposal=1,
    )
    size = OUT.stat().st_size / 1e6
    print(f"wrote {OUT.relative_to(REPO_ROOT)}: {len(frames)} frames, {size:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
