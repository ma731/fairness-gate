# fairness-gate

I built a machine learning model, then built something that refuses to let me ship it
quietly when it treats people unequally.

**[See the live site](https://ma731.github.io/fairness-gate/)**

![A tour of the site: the finding, who the model misses, the checks, the charts, the fix
it refused, and the voice panel](docs/assets/tour.gif)

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
| Policy result | 24 checks: 15 pass, 9 warn, 0 fail (8 compare against a recorded baseline) |

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

`policy.yaml` is where the limits live: a plain file listing every one, in language you
can argue with. Each metric has two levels: `fail` breaks the build, and `warn` is the
standard I actually want to hit.

The model currently sits *between* those two levels on the race gaps, and that is
deliberate. If I set the failing limit to wherever the model happens to land today, the
gate never fires and means nothing. If I set it to where I wish the model were, the build
is permanently red and everyone learns to ignore it. Writing both numbers down keeps the
distance between "tolerated" and "wanted" visible instead of quietly collapsing it.

On every commit the gate checks four things:

1. **The committed results still pass the policy**, recalculated rather than trusted,
   because the saved verdict was produced under whatever the limits said at the time.
2. **Every document still matches the numbers.** The model card, the EU AI Act
   documentation, the privacy assessment and this whole website are regenerated from the
   audit and compared character by character. If I hand-edit a number in any of them, the
   build fails. That is what stops a document slowly becoming fiction.
3. **The results match the code that produced them**, by hashing the pipeline files. If I
   change the model and forget to re-run, the old results stop counting as evidence.
4. **Nothing got worse since the recorded baseline.** Any gap that widens by more than
   0.020 fails, even if it's still inside its limit. The limits were set after I'd seen
   the results, so this is what stops them being quietly used up. See
   [decision 0007](docs/decisions/0007-how-the-limits-were-chosen.md).

**The gate will not fire on noise.** This one took me a while to get right. A gap measured
on 180 people and one measured on 178,000 are not the same claim, and a build that goes
red because of random chance is a build people learn to re-run until it passes. So a check
only fails when the *entire* confidence interval is past the limit. If the estimate is
over the line but the interval still straddles it, that is a warning that says so in
words. Uncertainty can soften a failure. It can never invent one.

## What runs on its own

**Every commit and pull request:** lint, the tests on Python 3.11 and 3.12, the narrator's
evals, and `check_policy.py`. Seconds, no data needed.

**Every pull request:** a bot comments with what actually moved. A green build only says
nothing crossed a limit, and a change can stay inside every limit while taking four
points of recall off one group. `scripts/review_pr.py` compares the audit on main with
the one on the branch and lists status changes, numbers that moved more than 0.002, and
any check that was deleted. It never blocks the merge; that's the gate's job.

**Monthly:** the full audit reruns from scratch, commits the numbers if they moved, and
opens an issue if a limit breaks.

---

## How I measured it

**Split by time, not at random.** Trained on 2015, tuned on 2016, tested on 2018, the
way a model like this is really used. A random split of one year would score better and
hide the fact that the world moves.

**Four states held out entirely.** Never trained on. Performance drops there, and the
drop is reported.

**The model does see race and sex, so I tested removing them.** That's the fix most
people suggest first.

| | recall gap | false positive gap | accuracy |
|---|---:|---:|---:|
| sees race and sex | 0.3119 | 0.1447 | 0.8021 |
| both columns deleted | 0.2883 | 0.1409 | 0.7965 |

92% of the gap survives. The model rebuilds the signal from occupation, education,
hours and birthplace. Deleting the column doesn't delete the problem, it just stops you
seeing it.

**A second model family.** Everything above comes from gradient boosted trees, which
are good at finding exactly the kind of interactions that could rebuild race. So I
trained a logistic regression on the same splits and features.

| | recall gap | AUC | accuracy |
|---|---:|---:|---:|
| gradient boosted trees | 0.3119 | 0.888 | 0.8021 |
| logistic regression | 0.3154 | 0.875 | 0.7897 |

The gap is slightly worse, and the rank correlation between the two is 0.93: they fail
the same groups. The problem is in the data, not the algorithm.

**Three fairness measures, not one.** They can't all hold when groups differ in how
often the outcome happens, which here they do. Reporting only one would be choosing
which unfairness to hide.

**Groups under 500 people are withheld.** A rate from a few hundred people is mostly
noise. Six of the eighteen race-by-sex cells fall below that line and are shown hatched
rather than deleted, because they're the groups a survey covers worst.

---

## The generated summary

The front page has a short summary written by a language model. The model can't publish
it: `src/narrator.py` checks every draft first. Every number has to match a measured
value at the precision written, any group it calls best or worst has to really be,
withheld groups stay withheld, the fix is never described as shipped, the model is never
called fair, and nothing is said about attributes this audit didn't measure. A failed
draft goes back with its violations. If nothing passes, nothing is published.

To run it yourself, copy `.env.example` to `.env`, paste in a key (an Azure OpenAI
deployment, Claude on Azure, or the Claude API), and run
`python scripts/narrate.py --check`, then `python scripts/narrate.py`. The `.env` file is
git-ignored. Any model works, a cheap one included: the checker rejects a bad draft
whatever wrote it, so a weaker model only costs extra retries.

`evals/` holds twelve drafts: ten with planted errors, and two clean ones that must
pass. Building it caught a wrong number in this README: it said accuracy ran from 0.79,
and the real figure is 0.77.

## The voice

The button in the corner answers questions out loud, like "what did you find" or "why
didn't you fix it". Every answer is written from `results/audit.json` at build time,
with no model and no API call, so it can't say a number the audit doesn't contain.

The audio is pre-rendered by `scripts/voice_audio.py` with Microsoft's neural voices
through `edge-tts`: 17 clips, 2.8 MB, no API key needed, which matters because a key in
a public page is a published key. Each clip stores a hash of its text. If an answer
changes and the clip isn't re-rendered, the page falls back to the browser's voice
instead of reading an old number.

---

## The website

Five pages: the finding, the evidence, the method, the fix I didn't ship, and why I
built it. Fifteen kinds of chart, each chosen because it shows something a table
can't. `docs/about.html` is the only page I wrote by hand.

## What is in here

```
policy.yaml              the limits, readable by a non-engineer
src/data.py              loading the census data and building the splits
src/model.py             the model, the threshold, and calibration
src/fairness.py          per-group metrics and the gaps between them
src/uncertainty.py       confidence intervals; the gate runs on these
src/mitigation.py        what fixing it would cost
src/unaware.py           the same model without race and sex
src/compare.py           a logistic regression, to test if it's the algorithm
src/policy.py            compares measured numbers to the declared limits
src/report.py            model card, EU AI Act Annex IV, privacy assessment
src/dashboard.py         the website
src/charts_extra.py      charts that need more than a summary row
src/pointfield.py        the animated hero: one dot per person
src/glossary.py          plain English for every check name
src/narrator.py          the generated summary and its verifier
src/voice.py             the ask-out-loud panel
evals/                   drafts with planted errors for the verifier
scripts/run_audit.py     one command, every number
scripts/check_policy.py  the gate
scripts/review_pr.py     the pull request reviewer
scripts/voice_audio.py   renders the spoken answers to audio
scripts/narrate.py       writes the summary with any model, then checks it
scripts/record_tour.py   records the GIF above from the built site
docs/decisions/          why things are the way they are, including what I rejected
```

## Running it yourself

```bash
pip install -r requirements-dev.txt
python scripts/download_data.py   # ~3 GB of census files, cached outside the repo
python scripts/run_audit.py       # regenerates every number and every document
python -m pytest tests/ -q        # 129 tests, no data needed
python scripts/run_evals.py       # scores the summary verifier
```

CI runs Python 3.11 as well as 3.12, and some newer f-string syntax parses locally but
breaks on 3.11. Check against 3.11 before pushing.

## Licence

MIT. See [LICENSE](LICENSE).
