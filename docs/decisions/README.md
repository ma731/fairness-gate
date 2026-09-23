# Decisions

Every real choice in this project, written down at the point I made it, including the
ones where I decided *not* to do something.

The reason these exist: a repo shows you what someone built, and almost never shows you
what they considered and rejected. In fairness work the rejected options are usually the
interesting part, because the wrong fix and the right fix produce very similar looking
dashboards. If I only published the end state, you would have no way to tell whether I
thought about a problem and ruled it out, or just never noticed it.

Each file is short and follows the same shape: what I had to decide, what I picked, and
what it costs me.

| | decision | short version |
|---|---|---|
| [0001](0001-gate-on-the-interval.md) | Fail on the interval, not the estimate | A build that goes red on luck is a build people ignore |
| [0002](0002-reject-per-group-thresholds.md) | Do not ship the fix that works | It needs the person's race at decision time |
| [0003](0003-charts-must-encode-something.md) | No chart without a number behind it | Two impressive charts cut because the data had nothing for them |
| [0004](0004-webgl-without-a-library.md) | Raw WebGL, no Three.js | 4 KB against 600 KB, for one effect |
| [0005](0005-suppress-small-groups.md) | Hide any group under 500 people | And draw the hole, do not delete the row |
| [0006](0006-keep-protected-attributes-out.md) | Model never sees race or sex | Deleting the column does not delete the problem |
