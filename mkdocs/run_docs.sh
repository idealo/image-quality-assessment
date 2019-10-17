#!/usr/bin/env bash

cp ../README.md docs/index.md
cp -r ../readme_figures docs/
cp ../CONTRIBUTING.md docs/CONTRIBUTING.md
cp ../LICENSE docs/LICENSE.md
python autogen.py
mkdocs serve