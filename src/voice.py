"""A voice agent that can only say things the audit actually measured.

You press the button, ask a question out loud, and it answers. The point is not the
novelty. The point is that a page like this one is dense, and most people who open it
will not read all of it, so being able to just ask "what did you find" and get the real
number back is a better way in than scrolling.

The rule this module exists to enforce: **every sentence it speaks is built here, in
Python, from results/audit.json.** There is no model, no API call, no generated prose.
The answers are assembled from the same figures the charts draw, at build time, so the
agent cannot drift from the page and cannot invent a statistic. If a number changes, the
audit reruns, and what the agent says changes with it.

Matching a question to an answer is a list of regular expressions. That is a small
grammar and it will not understand everything, which is the trade I wanted: a narrow
thing that is always right beats a broad thing that is sometimes confidently wrong. On a
page arguing that you should check what your model is doing, shipping an unverifiable
chatbot would have been a bad joke.

Two notes on being honest with whoever uses it:

- Speech recognition in Chrome sends the audio to Google to transcribe. That is how the
  browser API works and I cannot change it from here, so the panel says so before the
  microphone is ever switched on.
- Firefox and Safari do not implement recognition at all. Rather than hide the feature
  there, the panel falls back to a text box and answers typed questions identically.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.dashboard import human_cost

REPO_ROOT = Path(__file__).resolve().parents[1]

# Where "take me to the intersections" should actually land.
#
# These are page plus anchor now that the site is five pages. If the target is on the
# page you are already on, the panel scrolls; if it is not, it follows the link. The
# spoken answer is identical either way, so asking a question never depends on having
# landed in the right place first.
SECTIONS = {
    "start": "index.html#plain", "plain": "index.html#plain",
    "basic": "index.html#plain", "beginning": "index.html#plain",
    "miss": "index.html#cost", "overlook": "index.html#cost",
    "who": "index.html#cost", "cost": "index.html#cost",
    "check": "evidence.html#checks", "policy": "evidence.html#checks",
    "gate": "evidence.html#checks",
    "intersect": "evidence.html#intersections", "cross": "evidence.html#intersections",
    "certain": "evidence.html#certainty", "confiden": "evidence.html#certainty",
    "noise": "evidence.html#certainty",
    "group": "evidence.html#groups", "table": "evidence.html#groups",
    "flow": "evidence.html#flow", "sankey": "evidence.html#flow",
    "fix": "fix.html#fix", "mitigat": "fix.html#fix", "solution": "fix.html#fix",
    "conclusion": "fix.html#conclusion", "takeaway": "fix.html#conclusion",
    "unaware": "method.html#unaware", "delete race": "method.html#unaware",
    "method": "method.html#method",
    "source": "method.html#sources", "data": "method.html#sources",
}

# Question to answer, walked top to bottom, first match wins.
#
# Two things I got wrong on the first pass, both caught by the tests:
#
# 1. Order. "why did you not fix it" contains the word fix, so the broad rule matched
#    first and the agent cheerfully described the mitigation it had refused to ship.
#    That is the one answer that would misrepresent the project, so the refusal
#    patterns sit above the fix ones.
# 2. Word boundaries. I closed every pattern with \b, which meant "intersect" would not
#    match "intersections" and "small group" would not match "small groups". People say
#    the plural. A pattern meant to match a prefix does not get a closing \b.
GRAMMAR = [
    (r"\b(help|what can (i|you)|options|commands)\b", "help"),
    # anything shaped like "why ... not ..." is asking about the rejection
    (r"\b(why not\b|why\b.{0,30}\bnot\b|reject|refus|did ?n.t ship|not ship)", "why"),
    (r"\b(ci gate|what is (this|a gate)|continuous integration|how does the gate)",
     "gate"),
    (r"\b(intersect|cross|both attributes|race and sex)", "intersection"),
    (r"\b(fix|mitigat|solve|repair|close the gap)", "fix"),
    (r"\b(threshold|cut.?off|sweep)", "threshold"),
    (r"\b(certain|confiden|interval|noise|sure)", "certainty"),
    (r"\b(small group|suppress|hidden|withheld|hatch)", "small"),
    (r"\b(protected|see race|use race|unawareness|remove race|delete race)",
     "protected"),
    (r"\b(data|census|source|survey|where.*(from|come))", "data"),
    (r"\b(how many|people|overlooked|missed|room)", "people"),
    (r"\b(recall|true positive|tpr)\b", "recall"),
    (r"\b(accuracy|accurate)", "accuracy"),
    (r"\b(status|passing|failing|build|red|green|verdict)", "status"),
    (r"\b(find|found|finding|result|gap|disparity|problem|wrong)", "finding"),
]


def _answers(result: dict, tables: dict[str, pd.DataFrame]) -> dict:
    """Every spoken line, written out of the audit rather than typed by hand."""
    test = tables["test"]
    race = next(s for s in result["fairness"]["test"] if s["attribute"] == "RAC1P")
    sex = next(s for s in result["fairness"]["test"] if s["attribute"] == "SEX")
    xsex = next(
        (s for s in result["fairness"]["test"] if s["attribute"] == "RACExSEX"), {}
    )
    mit = (result.get("mitigation") or {}).get("RAC1P") or {}
    base = mit.get("baseline") or {}
    per = mit.get("per_group") or {}
    best = mit.get("best_global") or {}

    cost = human_cost(test)
    hurt, spared = cost.iloc[0], cost.iloc[-1]
    overlooked = round(float(cost["overlooked"].sum()))

    sub = test[(test["attribute"] == "RAC1P") & test["reportable"]]
    acc_lo, acc_hi = float(sub["accuracy"].min()), float(sub["accuracy"].max())

    xcells = test[test["attribute"] == "RACExSEX"]
    suppressed = int((~xcells["reportable"]).sum())

    counts = {"pass": 0, "warn": 0, "fail": 0}
    for c in result["checks"]:
        counts[c["status"]] += 1
    verdict = result["policy_verdict"]

    ci = ((result.get("uncertainty") or {}).get("RAC1P") or {}).get("tpr_gap") or {}
    ci_text = (
        " The ninety five percent interval runs from "
        f"{ci['lo']:.3f} to {ci['hi']:.3f}, so it is not a quirk of the sample."
        if ci else ""
    )

    n = result["splits"]["test"]["n"]
    design = result["design"]
    unaware = (result.get("unaware") or {}).get("comparison") or {}

    return {
        "gate": (
            "Continuous integration is a robot that runs a list of checks every time "
            "somebody changes the code. A gate is a check with the authority to refuse. "
            "Teams already block code that fails a test, but almost nobody blocks a "
            "model that fails a fairness check, because that check usually lives in a "
            "document nobody opens again. So here the fairness limits sit in a file "
            "next to the code, and if the model crosses one, the build turns red like a "
            f"broken test. Right now the gate is at {verdict}, with "
            f"{counts['pass']} checks passing, {counts['warn']} warning and "
            f"{counts['fail']} failing."
        ),
        "finding": (
            "Out of every hundred people who genuinely earn above the threshold, this "
            f"model finds about {round(100 - spared['per_100'])} in the "
            f"{spared['group']} group and about {round(100 - hurt['per_100'])} in the "
            f"{hurt['group']} group. That is a recall gap of "
            f"{race['tpr_gap']:.3f}.{ci_text} And the accuracy is nearly identical "
            f"across those groups, between {acc_lo:.2f} and {acc_hi:.2f}. A model can "
            "be equally accurate everywhere and still put its mistakes on the same "
            "people every time."
        ),
        "people": (
            f"{overlooked:,} qualifying people are overlooked across every reported "
            f"group, in a single survey year, out of {n:,} tested. If a real decision "
            "hung on this model, that is the size of the room."
        ),
        "recall": (
            "Recall is the share of people who genuinely qualify that the model "
            "actually finds. If a hundred people really do earn above the line and it "
            "flags fifty four of them, recall is fifty four percent, and the other "
            "forty six are overlooked. It is the measure that tells you who gets "
            "missed, which is the question accuracy cannot answer."
        ),
        "accuracy": (
            f"Accuracy sits between {acc_lo:.2f} and {acc_hi:.2f} across the racial "
            "groups, which is almost flat. That is exactly the trap. Accuracy is the "
            "number most projects report, and by that number this model looks even "
            "handed. It is not. It is equally accurate everywhere while placing its "
            "errors on the same people."
        ),
        "intersection": (
            "Audit race on its own and the false positive gap is "
            f"{race['fpr_gap']:.3f}. Audit sex on its own and it is "
            f"{sex['fpr_gap']:.3f}. Cross the two together and it is "
            f"{xsex.get('fpr_gap', 0):.3f}. Two acceptable averages with a much worse "
            "cell hiding underneath them. Auditing one attribute at a time will never "
            "find that."
        ),
        "fix": (
            "Yes, partly. Giving each group its own cut-off takes the recall gap from "
            f"{base.get('tpr_gap', 0):.4f} down to {per.get('tpr_gap', 0):.4f}, and it "
            f"costs {mit.get('accuracy_cost', 0):.4f} of accuracy. By the number most "
            "projects report, that is nearly free. I built it, measured it, and did not "
            "ship it. Ask me why not."
        ),
        "why": (
            "Two reasons. First, it needs to know the person's race at the moment it "
            "decides, because it has to look up which cut-off applies to you. In hiring "
            "or lending that is either illegal or is the exact harm you were trying to "
            "prevent. Second, it moves the unfairness rather than removing it. The "
            f"false positive gap goes up, from {base.get('fpr_gap', 0):.3f} to "
            f"{per.get('fpr_gap', 0):.3f}. When groups genuinely differ in how often "
            "the outcome happens, equalising one kind of error forces the other kind to "
            "become unequal. That is proved, not a bug in my code. You are always "
            "choosing which unfairness to keep."
        ),
        "threshold": (
            f"The threshold is the cut-off, currently {design['threshold']:.4f}, chosen "
            f"on the {design['val_year']} data and never on the test year. The model "
            "gives a probability, and the threshold is the line above which it says "
            "yes. Moving it trades one kind of mistake for the other. I swept it end to "
            "end, and the best any single cut-off can do for the gap is "
            f"{best.get('tpr_gap', 0):.3f}, at an accuracy of "
            f"{best.get('accuracy', 0):.3f}, where the model says yes to almost "
            "everybody. That is not a fix, that is turning the model off."
        ),
        "certainty": (
            "A gap measured on a few hundred people and one measured on a hundred "
            "thousand are not the same claim. So every rate carries a confidence "
            "interval, and a check only fails when the whole interval is past the "
            "limit. If the estimate is over the line but the interval still straddles "
            "it, that drops to a warning and says so in words. Uncertainty can soften a "
            "failure. It can never invent one. A build that goes red on luck is a build "
            "people learn to rerun until it passes."
        ),
        "small": (
            "Any group under five hundred people gets measured and then withheld. "
            f"{suppressed} of the {len(xcells)} race by sex cells fall below that line. "
            "They are drawn hatched rather than deleted, because the absence is part of "
            "the finding. The groups most likely to be harmed are the ones a national "
            "survey is least likely to contain enough of to say anything about."
        ),
        "data": (
            "United States census microdata, the American Community Survey, through the "
            f"folktables package. Trained on {design['train_year']}, tuned on "
            f"{design['val_year']}, tested on {design['test_year']}, which is "
            f"{n:,} real people. Split by time rather than at random, because that is "
            "how a model actually gets used: built on the past, applied to the present. "
            "Four states are held out entirely and never trained on."
        ),
        "protected": (
            "It does see them, and I tested what happens when it cannot. Training the "
            "same model with race and sex deleted takes the recall gap from "
            f"{unaware.get('tpr_gap_aware', 0):.4f} to "
            f"{unaware.get('tpr_gap_unaware', 0):.4f}. That is "
            f"{unaware.get('share_remaining', 0):.0%} of the gap surviving, for "
            f"{unaware.get('accuracy_cost', 0):.4f} of accuracy. Deleting the column "
            "does not delete the problem, it deletes your ability to see it. The model "
            "rebuilds almost all of the signal from occupation, education, hours and "
            "birthplace, because those carry the history of who got which opportunities."
        ),
        "status": (
            f"The gate is at {verdict}. {counts['pass']} checks pass, "
            f"{counts['warn']} are warnings and {counts['fail']} fail. A warning means "
            "the model is past the standard I want but still inside the limit I "
            "tolerate. Both numbers are written down on purpose, so the distance "
            "between what is tolerated and what is wanted stays visible."
        ),
        "help": (
            "Ask me what a CI gate is, what the audit found, how many people are "
            "overlooked, what recall means, whether it can be fixed and why I did not "
            "ship the fix, what happens at the intersections, how sure I am, or where "
            "the data comes from. You can also say take me to the checks, or any other "
            "section, and I will scroll you there."
        ),
        "unknown": (
            "I only answer from what the audit actually measured, so I would rather say "
            "I do not know than guess. Try asking what the finding is, whether it can "
            "be fixed, what a CI gate is, or just say help."
        ),
    }


MARKUP = """
<button type="button" class="vx-fab" id="vx-open" aria-expanded="false"
        aria-controls="vx" aria-label="Ask about this audit">
  <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
    <path d="M12 3a3 3 0 0 1 3 3v6a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3z"/>
    <path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>
  </svg>
  <span>Ask</span>
</button>

<aside class="vx" id="vx" hidden aria-live="polite">
  <header>
    <span class="vx-title">Ask the audit</span>
    <button type="button" class="vx-x" id="vx-close" aria-label="Close">&#215;</button>
  </header>
  <div class="vx-log" id="vx-log"></div>
  <p class="vx-note" id="vx-note"></p>
  <div class="vx-bar">
    <button type="button" class="vx-mic" id="vx-mic">Hold to speak</button>
    <form class="vx-form" id="vx-form">
      <input type="text" id="vx-text" autocomplete="off"
             placeholder="or type a question" aria-label="Type a question">
      <button type="submit" aria-label="Send">&#8594;</button>
    </form>
  </div>
</aside>
"""

SCRIPT = r"""
(function () {
  var DATA = __DATA__;
  var panel = document.getElementById('vx');
  var fab = document.getElementById('vx-open');
  var log = document.getElementById('vx-log');
  var note = document.getElementById('vx-note');
  var mic = document.getElementById('vx-mic');
  var text = document.getElementById('vx-text');

  // The patterns come from the grammar in src/voice.py, so what the agent can say and
  // what the page shows cannot drift apart.
  var rules = DATA.grammar.map(function (r) {
    return { re: new RegExp(r[0], 'i'), key: r[1] };
  });

  function add(who, what) {
    var p = document.createElement('p');
    p.className = 'vx-' + who;
    p.textContent = what;
    log.appendChild(p);
    log.scrollTop = log.scrollHeight;
  }

  // Every answer already has an MP3 rendered at build time by scripts/voice_audio.py,
  // in a Microsoft neural voice. That is what plays. No key ships with the page, no
  // request leaves it, and the clip is on the CDN before anybody asks.
  //
  // A clip is only played if its recorded hash still matches the answer on this page.
  // If an answer changed and nobody re-rendered, the clip is stale, and audio confidently
  // reading a number that is no longer true is worse than a synthetic voice reading the
  // right one. So a stale clip is skipped and the browser voice takes over.
  var AUDIO = DATA.audio || {};
  var player = null;

  function clipFor(key) {
    var clip = AUDIO[key];
    if (!clip || !clip.file) { return null; }
    // The hash is of the answer this page is actually showing.
    return clip.sha === DATA.hashes[key] ? clip.file : null;
  }

  // Fallback only. Picking the voice by hand, because the browser default is whatever
  // the machine's locale happens to be. On a Windows box set to China that meant a
  // Chinese voice reading American census figures, which sounds broken and, on this
  // page of all pages, lands badly. Setting utterance.lang is not enough: it is a
  // request, and the engine ignores it if the chosen voice cannot honour it.
  var VOICE_RANK = [
    function (v) { return /en(-|_)US/i.test(v.lang) && /natural|neural/i.test(v.name); },
    function (v) { return /^en/i.test(v.lang) && /natural|neural/i.test(v.name); },
    function (v) { return /^Google US English/i.test(v.name); },
    function (v) { return /en(-|_)US/i.test(v.lang) && /^(Samantha|Ava|Allison)/i.test(v.name); },
    function (v) { return /en(-|_)(US|GB)/i.test(v.lang); },
    function (v) { return /^en/i.test(v.lang); }
  ];
  var picked = null;

  function pickVoice() {
    if (!('speechSynthesis' in window)) { return null; }
    var all = window.speechSynthesis.getVoices() || [];
    for (var r = 0; r < VOICE_RANK.length; r++) {
      for (var i = 0; i < all.length; i++) {
        if (VOICE_RANK[r](all[i])) { return all[i]; }
      }
    }
    return null;
  }

  // getVoices() is empty until the list loads, so ask again when it arrives.
  if ('speechSynthesis' in window) {
    picked = pickVoice();
    window.speechSynthesis.onvoiceschanged = function () { picked = pickVoice(); };
  }

  function hush() {
    if (player) { player.pause(); player = null; }
    if ('speechSynthesis' in window) {
      try { window.speechSynthesis.cancel(); } catch (e) {}
    }
  }

  // Speaking is a nicety. If neither path works, the answer is already on screen.
  function synthesise(what) {
    if (!('speechSynthesis' in window)) { return; }
    try {
      if (!picked) { picked = pickVoice(); }
      // No English voice installed at all: stay quiet rather than read English
      // numbers through a voice that cannot pronounce them.
      if (!picked) { return; }
      var u = new SpeechSynthesisUtterance(what);
      u.voice = picked;
      u.lang = picked.lang || 'en-US';
      u.rate = 0.98;
      window.speechSynthesis.speak(u);
    } catch (e) { /* no voice available */ }
  }

  function speak(key, what) {
    hush();
    var file = key ? clipFor(key) : null;
    if (!file) { synthesise(what); return; }
    try {
      player = new Audio(file);
      // The clip may be missing, blocked, or the browser may refuse to autoplay it.
      // Any of those and the synthetic voice picks it up, rather than silence.
      player.onerror = function () { player = null; synthesise(what); };
      var started = player.play();
      if (started && started.catch) {
        started.catch(function () { player = null; synthesise(what); });
      }
    } catch (e) {
      player = null;
      synthesise(what);
    }
  }

  function reply(key) {
    add('a', DATA.answers[key]);
    speak(key, DATA.answers[key]);
  }

  function navigate(said) {
    if (!/\b(take me|go to|show me|scroll|jump|open)\b/i.test(said)) { return false; }
    var lower = said.toLowerCase();
    var keys = Object.keys(DATA.sections);
    for (var i = 0; i < keys.length; i++) {
      if (lower.indexOf(keys[i]) === -1) { continue; }
      var target = DATA.sections[keys[i]];
      var parts = target.split('#');
      var here = (location.pathname.split('/').pop() || 'index.html');
      // Already on that page: scroll. On another page: follow the link.
      if (parts[0] === here) {
        var el = document.getElementById(parts[1]);
        if (!el) { continue; }
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        add('a', 'Scrolling there now.');
        speak(null, 'Scrolling there now.');
      } else {
        add('a', 'That is on another page. Taking you there.');
        speak(null, 'That is on another page. Taking you there.');
        setTimeout(function () { location.href = target; }, 700);
      }
      return true;
    }
    return false;
  }

  function answer(said) {
    if (!said) { return; }
    add('q', said);
    if (navigate(said)) { return; }
    for (var i = 0; i < rules.length; i++) {
      if (rules[i].re.test(said)) { reply(rules[i].key); return; }
    }
    reply('unknown');
  }

  var SR = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SR) {
    // Firefox and Safari have no recognition. Typing gets identical answers, so the
    // feature is not hidden here, it just loses the microphone.
    mic.hidden = true;
    note.textContent = 'This browser cannot do speech recognition, so type instead. '
      + 'The answers are the same.';
  } else {
    note.textContent = 'Your browser sends the audio to its own speech service to '
      + 'transcribe it. Nothing else on this page leaves your machine.';
    var rec = new SR();
    rec.lang = 'en-US';
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    rec.onresult = function (e) { answer(e.results[0][0].transcript); };
    rec.onerror = function (e) {
      mic.classList.remove('is-live');
      note.textContent = e.error === 'not-allowed'
        ? 'Microphone permission was declined, so type your question instead.'
        : 'That did not come through. Try again, or type it.';
    };
    rec.onend = function () { mic.classList.remove('is-live'); };

    var start = function (ev) {
      ev.preventDefault();
      mic.classList.add('is-live');
      try { rec.start(); } catch (e) { /* already listening */ }
    };
    var stop = function () { try { rec.stop(); } catch (e) {} };
    mic.addEventListener('mousedown', start);
    mic.addEventListener('touchstart', start, { passive: false });
    mic.addEventListener('mouseup', stop);
    mic.addEventListener('mouseleave', stop);
    mic.addEventListener('touchend', stop);
  }

  document.getElementById('vx-form').addEventListener('submit', function (e) {
    e.preventDefault();
    var said = text.value.trim();
    text.value = '';
    answer(said);
  });

  function open() {
    panel.hidden = false;
    fab.setAttribute('aria-expanded', 'true');
    if (!log.childElementCount) { add('a', DATA.answers.help); }
    text.focus();
  }
  function close() {
    panel.hidden = true;
    fab.setAttribute('aria-expanded', 'false');
    hush();
  }
  fab.addEventListener('click', function () {
    if (panel.hidden) { open(); } else { close(); }
  });
  document.getElementById('vx-close').addEventListener('click', close);
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && !panel.hidden) { close(); }
  });
})();
"""


AUDIO_MANIFEST = REPO_ROOT / "docs" / "audio" / "manifest.json"


def _audio() -> dict:
    """The pre-rendered clips, if any have been made. None is a normal state."""
    if not AUDIO_MANIFEST.exists():
        return {}
    return json.loads(AUDIO_MANIFEST.read_text(encoding="utf-8")).get("clips", {})


def build(result: dict, tables: dict[str, pd.DataFrame]) -> str:
    """The panel and its script, with every answer already written from the audit."""
    answers = _answers(result, tables)
    # The page carries the hash of the text it is actually showing. The player compares
    # it to the hash the clip was rendered from and refuses a clip that no longer
    # matches, so stale audio can never read out a number the page has moved past.
    hashes = {
        key: hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        for key, text in answers.items()
    }
    payload = json.dumps(
        {
            "answers": answers,
            "hashes": hashes,
            "audio": _audio(),
            "grammar": [list(rule) for rule in GRAMMAR],
            "sections": SECTIONS,
        },
        ensure_ascii=False,
    )
    return MARKUP + "<script>" + SCRIPT.replace("__DATA__", payload) + "</script>\n"
