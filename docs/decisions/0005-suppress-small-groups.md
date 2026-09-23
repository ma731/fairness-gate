# 0005. Any group under 500 people gets measured and then withheld

**Status:** accepted

## What I had to decide

When I cross race with sex I get eighteen cells. Some of them are large. Some of them
have a couple of hundred people in the whole test year.

A percentage calculated from 200 people is mostly noise. If I publish "this group has a
false positive rate of 0.31" and that number could just as easily have been 0.19 or 0.44,
I have not reported a finding, I have reported a coin flip in a serious voice. And the
groups where this happens are almost always small minority populations, which means the
noisiest, least trustworthy numbers on the page would be attached to exactly the people
who get hurt when someone quotes a statistic about them that is not real.

There is also a privacy angle. Rates over very small cells start leaking information
about individuals, which is why statistical agencies have suppression rules in the first
place.

## What I picked

A floor of 500 people. Below that, the group is still measured, still stored in the
results, still excluded from every reported table, chart, gap calculation and gate check.

The important part: **the cell is drawn, hatched, labelled "below reporting floor". It is
not deleted.**

Six of the eighteen cells fall below the line. If I dropped those rows, the page would
show twelve tidy cells and the reader would assume that was everybody. Drawing the hole
makes the absence itself part of the finding, and the finding is uncomfortable: the groups
most likely to be harmed by a model are the ones a national survey is least likely to
contain enough of to say anything about. That is not a gap in my data, that is a gap in
how the data gets collected, and it is inherited by every model built on it.

## What it costs me

Real disparities inside those six cells are invisible to the gate. It cannot fail on a
group it refuses to measure, so this is a genuine blind spot rather than a solved problem.

I think the alternative is worse. Reporting a rate I do not believe, about a population
that has historically been on the receiving end of exactly that, is not more rigorous than
saying "I do not have enough data here." Saying it out loud, on the page, where a reader
can see the shape of what is missing, is the most honest version I could find.
