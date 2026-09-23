# 0007. How the limits in policy.yaml were chosen

**Status:** accepted

## What I had to decide

The gate is only as good as the numbers in `policy.yaml`. The first question anyone
should ask is who decided the recall gap limit is 0.35, and whether that number means
anything.

## The honest answer first

I set them, and I set them after I'd seen the model's results. The first commit already
reported every group's recall. `policy.yaml` came about three and a half hours later, and
the race-by-sex limits the next morning, in the same commit as that audit.

There's no legal threshold for a recall gap to borrow. The closest real standard, the US
four-fifths rule for hiring, compares selection rates as a ratio, and it isn't what this
gate measures.

A limit chosen with the answer in view can be chosen to pass. Several of mine sit close to
what the model scored: a limit of 0.35 against a measured 0.312, a target of 0.08 against
a measured 0.080. I'm not going to pretend that's a coincidence.

## How each number was picked

- **Two levels per check.** `warn` is where I'd want a model like this to be, roughly half
  the current race recall gap. `fail` sits above where the model landed, so the gate
  catches it getting worse without being red from day one. A limit set at the aspiration
  would be permanently red, and people stop reading a build that's always red.
- **Crossed attributes get looser limits**, because comparing more, smaller cells widens a
  max-minus-min gap on its own.
- **Selection rate is held loosely on purpose.** The groups genuinely differ by 0.270 in
  how often the outcome happens, so demanding equal selection would mean being wrong about
  individuals to look equal on average.
- **Performance floors** (AUC 0.85, calibration error 0.05) exist so fairness isn't being
  audited on a model nobody should use.

## What makes that less bad

- **The regression guard.** Every commit is compared with a recorded baseline, and any gap
  that widens by more than 0.020 fails the build, even if it's still inside its absolute
  limit. So a generous limit can't be quietly used up.

  This guard was in `policy.yaml` from the first version and never ran once. The baseline
  file was never recorded, so the check returned nothing and nobody noticed. I found it
  while writing this record. It now runs on every commit, and a missing baseline shows up
  as a warning instead of silence.
- **Every change is a visible diff.** Loosening a limit means editing `policy.yaml` in a
  pull request someone has to approve, and the review bot names any check that disappears.

## What a real deployment should do differently

The limits should be set by whoever is accountable for the decision (the model owner,
with legal and the people affected), written down before the model is trained, and
committed before the first audit. The gate then enforces a standard it didn't help
choose.

Here I was the builder and the standard-setter at once. That's the weakest part of this
project, and the first thing I'd change in a real one.
