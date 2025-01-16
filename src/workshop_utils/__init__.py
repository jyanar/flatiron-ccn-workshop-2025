#!/usr/bin/env python3

from .fetch import DOWNLOADABLE_FILES
from .plotting import *

import os
import pathlib

repo_dir = pathlib.Path(__file__).parent.parent.parent
os.environ["NEMOS_DATA_DIR"] = os.environ.get("NEMOS_DATA_DIR", str(repo_dir / "data"))
