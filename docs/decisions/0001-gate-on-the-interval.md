# 0001. Fail on the interval, not on the estimate

**Status:** accepted

## What I had to decide

The gate compares a measured number to a limit I wrote down. Say the limit is 0.30 and I
measure 0.31. Does the build go red?

The naive answer is yes, obviously. But a measured number is not a fact, it is an
estimate with wobble in it, and how much wobble depends entirely on how many people it
was calculated from. Some of my groups have 178,000 people in them. Some have 600. A gap
of 0.31 measured on the big group is a real finding. The same 0.31 measured on the small
one could easily be 0.24 or 0.38 if I re-ran on a different sample of the same
population.

## What I picked

Every rate gets a confidence interval, which is the range the true value is plausibly in
given how much data I had. A check only **fails** when the whole interval is past the
limit. If the estimate is over the line but the interval still overlaps it, the check
drops to a **warning** and says so in words:

> point estimate breaches the limit but the 95% interval [0.287, 0.334] does not clear
> it, so this is not yet distinguishable from noise

Uncertainty can turn a failure into a warning. It can never turn a pass into a failure.
It only ever makes the gate more reluctant to stop you.

Intervals come from a parametric bootstrap over the counts (Wilson for single rates), and
when I need the interval on a *gap* I resample both groups together on the same draw,
because the difference between two numbers has its own spread that you cannot get by
subtracting their two intervals.

## What it costs me

A real problem in a small group can sit at warning for a while instead of failing
immediately. I decided that is the right way round. The alternative failure mode is much
worse: a gate that fires on randomness trains everyone to re-run the build until it goes
green, and once people are doing that the gate has stopped existing. It is still there in
the config, it just does not mean anything any more. A check nobody believes is worse
than no check, because it also buys you false comfort.

Six of my race-by-sex cells are too small to report at all, and they surface as an
explicit "not enough data" rather than as a number I would have to caveat in every
conversation.
