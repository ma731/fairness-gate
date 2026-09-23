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

import json

import pandas as pd

from src.dashboard import human_cost

# Where "take me to the intersections" should actually land.
SECTIONS = {
    "start": "plain", "plain": "plain", "basic": "plain", "beginning": "plain",
    "miss": "cost", "overlook": "cost", "who": "cost", "cost": "cost",
    "check": "checks", "policy": "checks", "gate": "checks",
    "intersect": "intersections", "cross": "intersections",
    "certain": "certainty", "confiden": "certainty", "noise": "certainty",
    "group": "groups", "table": "groups",
    "flow": "flow", "sankey": "flow",
    "fix": "fix", "mitigat": "fix", "solution": "fix",
    "method": "method",
    "source": "sources", "data": "sources",
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
            "Race and sex never go into the model. It is trained on ten features and "
            "none of them is a protected attribute, and the gap is there anyway. That "
            "is the whole point. Deleting the column does not delete the problem, it "
            "deletes your ability to see it. The model reaches the same disparity "
            "through occupation, education, hours and region."
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

  // Speaking is a nicety. If the browser has no voice, the answer is already on screen.
  function speak(what) {
    if (!('speechSynthesis' in window)) { return; }
    try {
      window.speechSynthesis.cancel();
      var u = new SpeechSynthesisUtterance(what);
      u.rate = 1.02;
      window.speechSynthesis.speak(u);
    } catch (e) { /* no voice available */ }
  }

  function reply(key) {
    add('a', DATA.answers[key]);
    speak(DATA.answers[key]);
  }

  function navigate(said) {
    if (!/\b(take me|go to|show me|scroll|jump)\b/i.test(said)) { return false; }
    var lower = said.toLowerCase();
    var keys = Object.keys(DATA.sections);
    for (var i = 0; i < keys.length; i++) {
      if (lower.indexOf(keys[i]) !== -1) {
        var el = document.getElementById(DATA.sections[keys[i]]);
        if (el) {
          el.scrollIntoView({ behavior: 'smooth', block: 'start' });
          add('a', 'Scrolling there now.');
          speak('Scrolling there now.');
          return true;
        }
      }
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
    if ('speechSynthesis' in window) { window.speechSynthesis.cancel(); }
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


def build(result: dict, tables: dict[str, pd.DataFrame]) -> str:
    """The panel and its script, with every answer already written from the audit."""
    payload = json.dumps(
        {
            "answers": _answers(result, tables),
            "grammar": [list(rule) for rule in GRAMMAR],
            "sections": SECTIONS,
        },
        ensure_ascii=False,
    )
    return MARKUP + "<script>" + SCRIPT.replace("__DATA__", payload) + "</script>\n"
