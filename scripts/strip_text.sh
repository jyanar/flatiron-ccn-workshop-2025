#!/usr/bin/env bash

set -euo pipefail

HEADER="---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: python3
  language: python
  name: python3
---
"
STRIP_TEXT_EXPLAIN="
This notebook has had all its explanatory text removed and has not been run.
 It is intended to be downloaded and run locally (or on the provided binder)
 while listening to the presenter's explanation. In order to see the fully
 rendered of this notebook, go [here]" # and then we add the link below

# \-escape the newlines for use in Sed.
HEADER=${HEADER//$'\n'/\\$'\n'}
STRIP_TEXT_EXPLAIN=${STRIP_TEXT_EXPLAIN//$'\n'/\\$'\n'}

for f in docs/source/full/*/*md; do

    # don't do this on index or if we've already removed text
    if [[ "$f" =~ "index" || "$f" =~ "stripped" || "$f" =~ "glossary" ]]; then
        continue
    fi

    out_f="${f/.md/-stripped.md}"
    out_f="${out_f/full/users}"
    mkdir -p "$(dirname $out_f)"
    md="$(basename $f)"
    dir="$(basename $(dirname $f))"
    ipynb="${md/.md/.ipynb}"
    stripped_ipynb="${ipynb/.ipynb/-stripped.ipynb}"
    TITLE="$(grep '^# ' $f | head -1)"

    # Process file with sed
    sed -E -n -e '/^#/p' -e '/\{code-cell\}|:::/,/```|:::/p' -e '/<div class=.render/,/div>/p' "$f" > "$out_f"
    sed -i '' 's/.*div.*/\n/g' "$out_f"   # Note the '' for macOS sed
    sed -i '' "s/${ipynb}/${stripped_ipynb}/g" "$out_f"
    sed -i '' "s/${TITLE}/${TITLE}, Text Removed\n${STRIP_TEXT_EXPLAIN}(..\/..\/full\/${dir}\/${md})\n/g" "$out_f"
    sed -i '' "1s/^/${HEADER}/g" "$out_f"

    cp "$out_f" tmp.md
    uniq tmp.md > "$out_f"
    rm tmp.md

done
