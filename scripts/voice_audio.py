"""Render every spoken answer to MP3 once, at build time.

A hosted speech API would need a key, and a key in a public page is a published key. The
answers are fixed at build time, so they're rendered here with edge-tts (Microsoft neural
voices, no key needed). Each clip stores a hash of its text, and the page won't play a
clip whose hash no longer matches.

    python scripts/voice_audio.py [--all] [--voice NAME] [--list]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.report import RESULTS_DIR
from src.voice import _answers

AUDIO_DIR = REPO_ROOT / "docs" / "audio"
MANIFEST = AUDIO_DIR / "manifest.json"

# Warm, unhurried, and not trying to sell anything, which is the right register for a
# page about a model getting things wrong.
DEFAULT_VOICE = "en-US-AndrewMultilingualNeural"
DEFAULT_RATE = "-4%"


def fingerprint(text: str) -> str:
    """What the clip was made from, so a stale clip can be detected and skipped."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_answers() -> dict:
    audit = RESULTS_DIR / "audit.json"
    if not audit.exists():
        raise SystemExit(f"no audit at {audit}. Run scripts/run_audit.py first.")
    result = json.loads(audit.read_text(encoding="utf-8"))
    tables = {
        split: pd.read_csv(RESULTS_DIR / f"groups_{split}.csv")
        for split in ("test", "shift")
    }
    return _answers(result, tables)


def load_manifest() -> dict:
    if not MANIFEST.exists():
        return {}
    return json.loads(MANIFEST.read_text(encoding="utf-8")).get("clips", {})


async def render(text: str, path: Path, voice: str, rate: str) -> None:
    import edge_tts

    await edge_tts.Communicate(text, voice, rate=rate).save(str(path))


async def list_voices() -> None:
    import edge_tts

    for v in await edge_tts.list_voices():
        if v["Locale"].startswith("en-"):
            tags = (v.get("VoiceTag") or {}).get("VoicePersonalities") or []
            print(f"  {v['ShortName']:<38} {v['Gender']:<7} {', '.join(tags)}")


async def main_async(args) -> int:
    if args.list:
        await list_voices()
        return 0

    answers = load_answers()
    previous = load_manifest()
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    clips, made, kept = {}, 0, 0
    for key, text in answers.items():
        digest = fingerprint(text)
        path = AUDIO_DIR / f"{key}.mp3"
        old = previous.get(key) or {}

        fresh = (
            not args.all
            and path.exists()
            and old.get("sha") == digest
            and old.get("voice") == args.voice
        )
        if fresh:
            kept += 1
        else:
            print(f"  rendering {key} ({len(text.split())} words) ...", flush=True)
            await render(text, path, args.voice, args.rate)
            made += 1

        clips[key] = {
            "file": f"audio/{key}.mp3",
            "sha": digest,
            "voice": args.voice,
            "bytes": path.stat().st_size,
        }

    # Clips for answers that no longer exist would otherwise sit there for ever.
    for stale in set(previous) - set(clips):
        leftover = AUDIO_DIR / f"{stale}.mp3"
        if leftover.exists():
            leftover.unlink()
            print(f"  removed {stale}.mp3, no longer an answer")

    MANIFEST.write_text(
        json.dumps({"voice": args.voice, "rate": args.rate, "clips": clips}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    total = sum(c["bytes"] for c in clips.values())
    print(f"\n{made} rendered, {kept} unchanged, {len(clips)} clips, {total / 1e6:.1f} MB")
    print("re-render after any change to the audit or to src/voice.py")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--rate", default=DEFAULT_RATE, help="speaking rate, e.g. -4%%")
    ap.add_argument("--all", action="store_true", help="re-render every clip")
    ap.add_argument("--list", action="store_true", help="list the English voices")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
