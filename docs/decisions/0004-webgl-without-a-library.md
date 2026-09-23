# 0004. Raw WebGL for the opening animation, no Three.js

**Status:** accepted

## What I had to decide

The top of the page is a slowly rotating cloud of points. Every point is one person from
the test set, coloured by what the model did to them, and the proportions of each colour
are the real confusion matrix. It is not an abstract graphic. The mass of points glowing
in one colour is the group that gets overlooked.

The normal way to build this is Three.js. It is the default choice for anything 3D in a
browser and I know how to use it.

## What I picked

Raw WebGL2, written directly. About 4 KB of JavaScript, one vertex shader, one fragment
shader, 26,000 points positioned on a seeded Fibonacci sphere.

Three.js is roughly 600 KB. I am using it for one effect: put points in space, spin them
slowly, colour them. That is a matrix multiply and two small shaders. Pulling in a whole
scene graph, a material system, a loader stack and a renderer abstraction to do that is
paying for a hundred features to use one.

There is a second reason that matters more to me. The project's entire argument is that
you should be able to trace every number on the page back to the data that produced it.
Loading 600 KB of third-party code into the hero section, where the visual *is* a claim
about the confusion matrix, sits badly with that. At 4 KB you can read the whole thing and
check that the colour proportions are the numbers I say they are.

The seed is fixed, so the layout is identical on every build. That keeps the
byte-identical regeneration check working, which would break if point positions were
random per load.

## What it costs me

I wrote the matrix maths myself, which is the part a library would have handled. If this
ever needs real 3D (lighting, loaded models, depth sorting) then this decision is wrong
and I should switch. It does not, and I will not pre-build for a requirement I do not
have.

If WebGL2 is unavailable the canvas is simply not drawn and the page is unaffected. The
animation is the opening image, never the evidence. Everything it hints at is stated
numerically further down.
