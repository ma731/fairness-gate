# 0002. Build the fix that works, then refuse to ship it

**Status:** accepted

## What I had to decide

The audit found that the model finds 85 out of every 100 qualifying people in one racial
group and 54 in another. Proving that is the easy half. The question is what I do about
it.

There is a known technique for exactly this. Instead of one cut-off for everyone, you
give each group its own, tuned so that every group ends up with the same recall. It is
called equal opportunity, it is a published method, and it is not hard to implement.

## What I picked

I built it, measured it on real data, and did not ship it. Both halves matter: I did not
want to wave it away as "too complex", and I did not want to ship it because the numbers
looked good.

| | recall gap | false positive gap | accuracy |
|---|---:|---:|---:|
| One cut-off for everyone | 0.3119 | 0.1447 | 0.8021 |
| One cut-off per group | 0.1143 | 0.1956 | 0.7979 |

It closes about two thirds of the recall gap for four tenths of a point of accuracy. By
the number most projects report, it is basically free.

Two reasons I still said no:

**1. It needs the person's race at the moment it decides.** Not while training. Not while
auditing. At the instant it produces an answer about you, it has to look up your race to
know which cut-off applies. In hiring, in lending, in almost every setting where anyone
would want a model like this, that is either flatly illegal or is itself the harm I was
trying to prevent. A fix that only functions by doing the thing you were avoiding is not
a fix.

**2. It moves the unfairness, it does not remove it.** Look at the false positive column.
It goes *up*, 0.145 to 0.196. That is not a bug in my code. When two groups genuinely
differ in how often the outcome occurs (and here the base rates differ by 0.270), making
one kind of error equal across groups mathematically forces the other kind to become
unequal. This is proved, not observed. There is no configuration where everything is fair
at once. You are always choosing which unfairness you are willing to keep, and anyone who
tells you they removed bias from a model has simply not shown you the column that got
worse.

I also checked the lazy option first, which is to just move the one shared cut-off around
until the gap closes. It does close, at a threshold of 0.05, where the model says yes to
nearly everybody and accuracy falls to 0.602. That is not fairness, that is turning the
model off.

## What it costs me

The gap is still there. The repo ships with a measured, unresolved disparity in it, and
the headline finding is a problem rather than a solution.

I would rather that. The thing I actually wanted to demonstrate is that you can tell the
difference between a mitigation that helps and a mitigation that relocates the damage, and
you can only tell by publishing both columns. All of this is in the model card and the
Annex IV documentation as a rejected option with the reasoning attached, and there is a
test that fails if someone quietly removes it.
