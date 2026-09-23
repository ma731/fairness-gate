# 0006. Let the model see race and sex, then measure what removing them does

**Status:** accepted
**Supersedes:** an earlier version of this file that claimed the model never saw these
attributes. That was wrong: `run_audit.py` calls `load_splits()` with the default, and
the default includes them. The claim was corrected and then tested properly, which is
how this decision ended up with numbers attached instead of an argument.

## What I had to decide

Should race and sex be features the model can learn from?

The instinctive answer from most people, including engineers, is "obviously not, delete
them, then it cannot discriminate." That intuition has a name, fairness through
unawareness. It is the single most common first move in this area, and I did not want
to dismiss it with a citation. I wanted to run it.

## What I picked

The shipped model uses them, because they are part of the standard ACSIncome feature set
and because leaving them in makes the interesting question askable. Then `src/unaware.py`
trains the identical pipeline a second time with both columns dropped, picks its own
threshold on validation, and scores it on the same test year.

| | recall gap | false positive gap | accuracy |
|---|---:|---:|---:|
| sees race and sex | 0.3119 | 0.1447 | 0.8021 |
| both columns deleted | 0.2883 | 0.1409 | 0.7965 |

**92.4% of the gap survives.** Deleting the two columns closes 0.024 of a 0.312 gap and
costs 0.0056 of accuracy.

The model has no idea what race anyone is and it still finds roughly 83 out of every 100
qualifying people in one group and 56 in another. It gets there through occupation,
education, hours worked and place of birth: variables that carry the history of who got
which opportunities, and correlate with race strongly enough to rebuild almost all of the
signal without ever being told.

The second model gets its own threshold rather than inheriting the first one. The two
produce differently shaped score distributions, so reusing one cut-off would have meant
comparing two different operating points and calling the difference an effect.

## What it costs me

The comfortable answer is gone and I do not have one to replace it with. A team that
deletes race and ships is in a worse position than a team that keeps it, because they now
have a problem they have made unmeasurable and they believe they solved it. That is a
harder thing to tell someone than "remove the column".

It also costs a second full training run in every audit, which is a few minutes. Worth it:
this is the one result that changes what a reader does on Monday.

See [0002](0002-reject-per-group-thresholds.md) for the mitigation that does close the
gap, and why I would not ship that either.
