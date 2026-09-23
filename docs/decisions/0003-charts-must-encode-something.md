# 0003. Every chart has to encode a number, or it is decoration

**Status:** accepted

## What I had to decide

I wanted the site to be genuinely good looking, not a Jupyter notebook screenshot. So I
went looking at chart catalogues for forms worth using, and the temptation there is
obvious: a chord diagram looks incredible. So does a transit map. Put either on a page
and people stop scrolling.

The problem is that a chord diagram shows flows *between* categories, and a transit map
shows a network with real topology. My data has neither. I have groups, and rates within
those groups. There is nothing flowing from one racial group to another. If I drew a
chord diagram here, the ribbons would have to be connected to something I made up.

## What I picked

Fifteen chart forms, each one chosen because it answers a question a table cannot, and
two impressive ones cut.

What survived, and why each earns its place:

- **Sankey**, because every one of the 600,551 people has to come out somewhere, and the
  ribbon widths force the four outcomes to add back up to the population. You cannot hide
  a group in a Sankey.
- **Dot grid**, because when the answer is "42,517 people were overlooked" the reader
  should be looking at people, not at a bar whose height is 42,517.
- **Tradeoff curve**, because the *shape* is the argument. You can see the line refuse to
  dip. No table makes "there is no good threshold" as immediately obvious as a curve that
  visibly does not go where you want it to.
- **Intersection matrix**, because the whole finding is that the bad cell only appears
  when you cross two attributes, and a matrix is literally the shape of crossing two
  attributes.
- **Ridgeline**, because the groups have differently shaped score distributions and a
  single mean per group hides that completely.
- **Faceted reliability curves**, because I first drew all eight on one axis and it was
  unreadable. Small multiples, one per group, same scale, so you compare by position
  rather than by untangling colours.

The rule I ended up with: if I can delete the underlying number and the chart still looks
the same, the chart was not showing me the number.

## What it costs me

Two of the most eye-catching options in the catalogue, in a project where part of the
point was proving I can build something that looks good.

I think that trade is correct and I think it is actually the stronger signal. A chart that
encodes nothing is decoration pretending to be evidence, and on a page whose entire
argument is "look at the numbers properly", that would undercut everything else on it.
