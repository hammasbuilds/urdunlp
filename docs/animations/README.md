# Isometric animations

Two views of each project: an **overview pipeline** (`*-to-*.html`, `*-readers.html`)
showing the path an input takes through the stages, and an **architecture** view
(`*-architecture.html`, `token-by-token.html`) whose inset band shows what happens to
each token of that input inside a stage.

Every value on these pages was read out of the code while it ran - the decode problem,
the coercion verdict, the confirmation text, the comparison counts - by the generator
that writes them:

    D:/launchpad/animations/gen/urdunlp.py      writes the spec
    D:/launchpad/animations/build.py                 writes the page
    D:/launchpad/animations/build.py --render        writes the .mp4 and the poster .png

Each page names the module and function behind every stage, so a claim on the picture
can be checked against the file. Open any `.html` straight from disk; the `.mp4` is
1080x1350, which is the shape LinkedIn gives the most room to.

One thing they deliberately do not draw: an embedding or a vector store. The decoder is built from counts: letter emission tables, a word-level vocabulary prior and a bigram language model. There is no neural network and nothing to download.
