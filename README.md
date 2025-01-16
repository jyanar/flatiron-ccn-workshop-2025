# ccn-software-jan-2025

Materials for CCN software workshop at Flatiron, Jan 2025

We have a slack channel for communicating with attendees, if you haven't received an invitation, please send us a note!

Note that the rest of this README is for contributors to the workshop.

## Building the site locally

To build the site locally, clone this repo and install it in a fresh python 3.11 environment (`pip install -e .`). Then run `make -C docs html O="-T"` and open `docs/build/html/index.html` in your browser.

## strip_text.sh

- This script creates one copy of each file found at `docs/source/full/*/*md`, the copy in `docs/source/users/*/*-stripped.md`.
- In general, `strip_text.sh` leaves the code alone and removes all markdown except blocks in colon fences (`:::`, e.g., admonitions) and headers.
- The title should be on a line by itself, use `#` (e.g., `# My awesome title`) and be the first such line (so no comments above it).
- Any text wrapped in `<div class='render-strip'></div>` will be visible in the stripped version of the notebook, but not the original, while text wrapped in `<div class='render-both'></div>` will be visible in both.
    
## binder

See [nemos Feb 2024 workshop](https://github.com/flatironinstitute/nemos-workshop-feb-2024) for details on how to set up the Binder
