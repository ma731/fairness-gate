# fairness-gate

I built a machine learning model, then built something that refuses to let me ship it
quietly when it treats people unequally.

> **Not a real system.** This predicts income from census answers, which is a standard
> benchmark task. No decision about any actual person should be made with it. The model
> is here so the safety machinery around it has something real to hold.

---

## First, in plain language

If you already know this, skip to [What I found](#what-i-found).

**What a CI gate is.** CI stands for continuous integration. It is a robot that wakes up
every time someone changes the code, runs a list of checks, and either says "fine" or
refuses. Most teams use it to catch broken code: does it still compile, do the tests
still pass. A *gate* is a check that can say no and stop everything.

My idea was simple. Teams already refuse to ship code that fails a test. Almost nobody
refuses to ship a model that fails a fairness check, because that check usually lives in
a PDF someone wrote once and never opened again. So I put the fairness limits in a file
next to the code, and wired them to the same robot. If the model starts treating one
group meaningfully worse than another, the build turns red, exactly like a broken test.
You cannot merge past it without someone editing the limits, and editing them leaves a
mark in the history that a reviewer has to approve.

**What the model is doing.** It looks at ten facts about a person from a government
survey (age, education, hours worked, occupation and so on) and guesses whether they earn
more than $50,000. I know the real answer for every person, because the survey recorded
it. So I can check not just whether the model is right overall, but *who it is wrong
about*.

**A few words I use a lot:**

- **Recall** is the share of people who genuinely qualify that the model actually finds.
  If a hundred people really do earn above the threshold and the model flags 54 of them,
  recall is 54%. The other 46 are overlooked.
- **False positive rate** is the opposite mistake: people flagged who should not have
  been.
- **Threshold** is the cut-off. The model outputs a probability, and I have to pick a line
  above which it says yes. Moving the line trades one kind of mistake for the other.
- **The gap** is just the difference between the best-treated group and the worst-treated
  one on whichever of those measures we are talking about.

---

## What I found

On 600,551 people in the year I held back for testing:

| | |
|---|---|
| Recall gap between racial groups | **0.312** (95% confidence interval 0.300 to 0.324) |
| Accuracy across those same groups | 0.77 to 0.84, almost identical |
| Qualifying people the model overlooks | **42,517** in a single survey year |
| False positive gap, race on its own | 0.145 |
| False positive gap, **race and sex together** | **0.221** |
| Policy result | 16 checks: 7 pass, 9 warn, 0 fail |

Here is the sentence that made me keep going. **Out of every hundred people who genuinely
earn above the threshold, the model finds 85 in one group and 54 in another.** And its
accuracy is nearly the same for both.

That is the trap. Accuracy is the number everyone reports, and by that number this model
looks even-handed. It is not. It is equally accurate everywhere and it puts its mistakes
on the same people every time. If you only ever check one number, this is invisible to
you.

It gets worse when you stop looking at one thing at a time. Audit race on its own and the
false positive gap is 0.145. Audit sex on its own and it is 0.080. Audit the two crossed
together and it is **0.221**. Two acceptable averages, one much worse cell underneath
them.

## Can it be fixed?

I tried, and I published the bill instead of guessing at it.

| | recall gap | false positive gap | accuracy |
|---|---:|---:|---:|
| One threshold for everyone | 0.3119 | 0.1447 | 0.8021 |
| One threshold per group | 0.1143 | 0.1956 | 0.7979 |

Giving each group its own cut-off closes two thirds of the recall gap and costs four
tenths of a point of accuracy. By the number most projects report, it is nearly free.

**I still rejected it**, for two reasons that matter more than the accuracy:

1. It needs to know the person's race **at the moment it decides**. The system has to
   look at your race to pick which cut-off applies to you. In hiring or lending, that is
   either illegal or is the exact harm you were trying to prevent.
2. It moves the unfairness rather than removing it. The false positive gap goes *up*,
   0.145 to 0.196. When groups genuinely differ in how often the outcome occurs, making
   one type of error equal across groups forces the other type to become unequal. That is
   a proved mathematical result, not a bug I could code my way out of. There is no
   setting where everything is fair at once. You are choosing which unfairness to keep.

I also tried the obvious thing first: just move the one shared cut-off up and down until
the gap closes. It does close, at a threshold of 0.05, where the model says yes to almost
everybody and accuracy collapses to 0.602. That is not a fix. That is switching the model
off and calling it fair.

Recording a fix that works and explaining why I would not ship it felt more honest than
quietly not trying.

---

## How the gate works

```bash
python scripts/check_policy.py      # a few seconds, needs no data. This is what CI runs.
python scripts/run_audit.py --gate  # the full run, ~3 GB of data, exits non-zero on a breach
```

`policy.yaml` is the whole point. It is a plain file listing every limit, in language you
can argue with. Each metric has two levels: `fail` breaks the build, and `warn` is the
standard I actually want to hit.

The model currently sits *between* those two levels on the race gaps, and that is
deliberate. If I set the failing limit to wherever the model happens to land today, the
gate never fires and means nothing. If I set it to where I wish the model were, the build
is permanently red and everyone learns to ignore it. Writing both numbers down keeps the
distance between "tolerated" and "wanted" visible instead of quietly collapsing it.

On every commit the gate checks three things:

1. **The committed results still pass the policy**, recalculated rather than trusted,
   because the saved verdict was produced under whatever the limits said at the time.
2. **Every document still matches the numbers.** The model card, the EU AI Act
   documentation, the privacy assessment and this whole website are regenerated from the
   audit and compared character by character. If I hand-edit a number in any of them, the
   build fails. That is what stops a document slowly becoming fiction.
3. **The results match the code that produced them**, by hashing the pipeline files. If I
   change the model and forget to re-run, the old results stop counting as evidence.

**The gate will not fire on noise.** This one took me a while to get right. A gap measured
on 180 people and one measured on 178,000 are not the same claim, and a build that goes
red because of random chance is a build people learn to re-run until it passes. So a check
only fails when the *entire* confidence interval is past the limit. If the estimate is
over the line but the interval still straddles it, that is a warning that says so in
words. Uncertainty can soften a failure. It can never invent one.

## What runs on its own

Three jobs, none of which anyone has to remember.

**On every commit and every pull request**, the gate: lint, the tests on Python 3.11 and
3.12, and `check_policy.py`. Seconds, no data needed.

**On every pull request**, a reviewer that comments with what actually moved. This one
matters more than it sounds. A green build tells you nothing crossed a limit. It does
not tell you what moved, which direction, or which group absorbed it, and a commit can
keep every check inside its limit while taking four points of recall off one group. So
`scripts/review_pr.py` diffs the audit on main against the audit on the branch and
writes the difference: checks that changed status, numbers that moved more than the
noise floor, and which gap is behind it. One comment per pull request, edited in place.

It never fails the build. Blocking is the gate's job, and a bot that both nags and
blocks gets muted. It also stays quiet about moves under 0.002, because on 600,000
people that is rounding and a comment that cries wolf is a comment people skim.
Deleting a check is the quietest way to turn a gate green, so removals get named.

**Monthly**, the full audit: it downloads the data, retrains, re-audits, commits the
new numbers only when they actually move, and opens an issue if a limit breaks.

---

## How I measured it, and why that way

**I split the data by time, not at random.** Trained on 2015, tuned on 2016, tested on
2018. That is how a model like this really gets used: built on the past, applied to the
present. Shuffling one year and splitting it randomly would have looked better and told
me nothing, because it hides the fact that the world moves.

**I held four whole states out.** Never trained on them at all. Performance drops there,
and I report the drop instead of pretending it does not happen.

**The model does see race and sex, and I tested what happens when it cannot.** They are
part of the standard feature set for this task, so the shipped model uses them. Then I
trained the whole thing a second time with both columns deleted, which is the fix
everybody suggests first, and measured it:

| | recall gap | false positive gap | accuracy |
|---|---:|---:|---:|
| sees race and sex | 0.3119 | 0.1447 | 0.8021 |
| both columns deleted | 0.2883 | 0.1409 | 0.7965 |

**92% of the gap survives.** Deleting the attributes buys you 0.024 of a 0.312 gap and
costs accuracy. The model reconstructs almost all of the signal from occupation,
education, hours and birthplace, because those carry the history of who got which
opportunities.

So an attribute is protected whether or not the model is allowed to look at it. Removing
it does not remove the disparity, it removes your ability to see the disparity, while
leaving you feeling like you did something. That is the mistake I most wanted to avoid,
and now there is a number for it instead of an opinion.

**I report three fairness measures, not one.** They contradict each other on purpose, and
they cannot all be satisfied when groups genuinely differ in outcomes, which here they do.
Picking one number to report would have been choosing which unfairness to hide.

**Groups under 500 people are measured and then withheld.** A percentage from a few
hundred people is mostly noise, and publishing it as a finding would be its own kind of
harm. Six of the eighteen race-by-sex cells fall below that line and are drawn hatched
rather than deleted, because the absence is the point: the groups most likely to be
harmed are the ones a survey is least likely to have enough of to say anything about.

---

## What is in here

```
policy.yaml              the limits, in a file a non-engineer can read and argue with
src/data.py              loading the census data, the four splits, attributes kept apart
src/model.py             the model, the threshold, and calibrating its probabilities
src/fairness.py          per-group metrics and the gaps between them
src/uncertainty.py       confidence intervals; the gate runs on these, not point estimates
src/mitigation.py        what fixing it would cost, and why I did not ship the fix
src/distributions.py     the binned data the charts need
src/policy.py            compares measured numbers to declared ones. Nothing else.
src/report.py            model card, EU AI Act Annex IV, privacy assessment: all generated
src/dashboard.py         the website
src/charts_extra.py      the charts that need more than a summary row
src/pointfield.py        the animated hero: one dot per person, no libraries
src/voice.py             ask it out loud; every answer written from the audit
src/glossary.py          plain English for every check name, in one place
src/unaware.py           the same model without race and sex, and what that costs
src/compare.py           a logistic regression, to ask if it is the algorithm
src/narrator.py          the generated summary, and the verifier that gates it
evals/                   drafts with planted errors, and what should be caught
scripts/review_pr.py     the pull request reviewer
scripts/voice_audio.py   renders every spoken answer to a file, once
scripts/run_audit.py     one command, every number
scripts/check_policy.py  the gate
docs/decisions/          why things are the way they are, including what I rejected
```

## Running it yourself

```bash
pip install -r requirements-dev.txt
python scripts/download_data.py   # ~3 GB of census files, cached outside the repo
python scripts/run_audit.py       # regenerates every number and every document
python -m pytest tests/ -q        # 45 tests, no data needed
```

One trap if you contribute: CI runs Python 3.11 as well as 3.12, and some f-string syntax
added in 3.12 parses fine locally while breaking the older job. Check against 3.11 before
pushing. I learned that the slow way.

## The website

The site is five pages, in the order the argument runs: the finding, the evidence, the
method, the fix I did not ship, and why I built it. It used to be one endless scroll,
which asked far too much of a reader. Every page is a real file, so links go straight to
a section and nothing needs JavaScript to show you content.

`docs/index.html` is the front of it. Fifteen kinds of chart, and each one is there because
it answers something a table cannot: a flow diagram because every person has to come out
somewhere, a dot grid because the denominator is people, a tradeoff curve because the
shape of the curve *is* the argument. I deliberately left out a couple of charts that
would have looked impressive, like a chord diagram, because this data has nothing for
them to show and a chart that encodes nothing is decoration pretending to be evidence.

## Is it the algorithm, or the data?

The obvious objection to everything above is that it all comes from one gradient boosted
tree. Boosting is very good at finding interactions, and an interaction between
occupation, region and hours is exactly the shape of thing that could rebuild race
without being told. So maybe this is a fact about the algorithm.

I trained a logistic regression on the identical splits and the identical features.
Linear, additive, about as different from a boosted forest as you can get while still
predicting the same thing.

| | recall gap | AUC | accuracy |
|---|---:|---:|---:|
| gradient boosted trees | 0.3119 | 0.888 | 0.8021 |
| logistic regression | 0.3154 | 0.875 | 0.7897 |

The gap is **slightly worse**, not better. And the rank correlation between the two is
**0.93**: rank the groups by recall under each model and you get almost the same order,
so the two families do not merely fail equally hard, they fail the same people.

That moves the finding from "this algorithm has a problem" to "this data encodes an
inequality and a model fitted to it will inherit it". The second is much harder to fix
and much more useful to know, and it is why picking a different library is not a plan.

## The generated summary, and why it needs a verifier

Everything else here builds its prose from templates, which cannot be wrong and also
cannot say anything the template author did not think of. A language model writes a
better summary, and will, given the chance, state a number that is not in the data. On a
page arguing that you should check what your model is doing, publishing unverified
generated text would be the joke writing itself.

So the model never publishes. It proposes, and `src/narrator.py` decides. Every number
in the draft has to trace to a measured value **at the precision it was written**.
Superlatives have to name the right group. Suppressed groups stay suppressed. The
mitigation is never described as adopted. Nothing may claim the model is fair. Nothing
may make a finding about an attribute this audit never measured. A draft with any
violation is rejected, the model is told exactly what was wrong, and it tries again. If
nothing verifies, **nothing is published**. There is no least-bad fallback, because a
guardrail with one is just a delay.

`evals/` holds twelve drafts with known planted errors and scores both directions:
misses, which are dangerous, and false alarms, which are how a check gets switched off by
whoever has to live with it. It runs in CI, needs no API key and no data.

Building it caught three bugs, two of them mine. My first fact set walked every number in
the audit including a 41 point sweep, so the allowed set was so large that three planted
errors matched something and passed. The second version still let through "accuracy
between 0.79 and 0.84", because 0.79 is a rounding of an unrelated AUC sitting elsewhere
in the file. The number existed. The claim was false: the measured range starts at 0.77.
That figure was in this README. Membership in the audit turns out to be necessary and not
sufficient, so claims with exactly one correct answer are now checked against that answer.

## The voice

There is a button in the corner that lets you **ask the page questions out loud**.
Press it, say "what did you find" or "why did you not fix it", and it answers. The
interesting part is what it cannot do: every sentence it speaks is written in
`src/voice.py` out of `results/audit.json` at build time. No model, no API call, no
generated prose. It can quote the recall gap because the recall gap is in the audit, and
it cannot say anything else, which is why there is a test asserting that every number it
is capable of speaking traces back to a measured one. On a page arguing that you should
check what your model is doing, bolting on a chatbot nobody could verify would have been
a bad joke. It falls back to a text box in browsers without speech recognition, and it
tells you up front that Chrome sends your audio to Google to transcribe it.

It speaks in a real voice rather than the browser's. The obvious way to do that is a
hosted speech API, and the obvious problem with one on a static site is that an API key
in a public page is a published API key. There is no server here to hide it behind. But
the set of sentences this page can say is finite and known before anybody visits, because
every answer is written in Python from the audit. So `scripts/voice_audio.py` renders
them all to MP3 once and commits them: 16 clips, 2.5 MB, Microsoft neural voices through
`edge-tts`, which needs no key and no account.

Each clip records the hash of the text it was made from, and the page carries the hash of
the text it is actually showing. If an answer changes and nobody re-renders, the hashes
differ, the clip is refused, and the browser voice takes over. Audio confidently reading
a figure the page has moved past is worse than a synthetic voice reading the right one,
and it is exactly the failure this whole project exists to complain about. A test fails
if any committed clip goes stale.

`docs/about.html` is the only page not generated from the data, and it says so on the
page. It is me explaining why I think this matters.

## Licence

MIT. See [LICENSE](LICENSE).
