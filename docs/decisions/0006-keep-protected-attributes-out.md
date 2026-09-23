# 0006. The model never sees race or sex, and that does not make it fair

**Status:** accepted

## What I had to decide

Should race and sex be features the model can learn from?

The instinctive answer from most people, including most engineers, is "obviously not,
delete them, then it cannot discriminate." That intuition has a name, fairness through
unawareness, and it is wrong. It is probably the single most common mistake in this area.

## What I picked

Race and sex are loaded, held in a separate structure, and used only to evaluate results.
The model is trained on ten features and none of them is a protected attribute. Race
crossed with sex is constructed for auditing and cannot reach the feature matrix, because
of how the loading code is structured rather than because I remembered not to.

And the gap is 0.312 anyway.

That is the entire point of doing it this way. The model has no idea what race anyone is,
and it still finds 85 out of 100 qualifying people in one group and 54 in another. It gets
there through occupation, education, hours, region: variables that carry the history of
who got which opportunities, and correlate with race strongly enough to reconstruct most
of the signal without ever being told.

So removing the column does not remove the disparity. It removes your **ability to see**
the disparity, while leaving you feeling like you did something. A team that deletes race
and ships is in a worse position than a team that keeps it, because they now have a
problem they have made unmeasurable, and they believe they solved it.

An attribute is protected whether or not the model is allowed to look at it. Which is why
the protected columns stay in the repo, stay out of the features, and are the only reason
any of this could be measured at all.

## What it costs me

Slightly awkward plumbing: the data loader has to carry two parallel structures everywhere
and keep them aligned across four splits. That is a small price.

The bigger cost is that this repo is now a demonstration that the comfortable answer does
not work, and it does not have a comfortable answer to replace it with. See
[0002](0002-reject-per-group-thresholds.md) for the fix that does close the gap and why I
would not ship that either.
