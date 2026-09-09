"""Read the Week 3 notebook and expose its validated scalar pricing functions.

The adapter is intentionally read-only: it does not modify Week 3.  Week 4
uses it for regression checks before applying the vectorized implementation.
"""

from __future__ import annotations

import json
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np


REQUIRED = {
    "normal_cdf",
    "validate_model_inputs",
    "calculate_d1_d2",
    "bsm_call_price",
    "bsm_put_price",
    "chooser_choice_boundary",
    "simple_chooser_price",
}


def load_week3_functions(project_root: Path):
    notebook_path = Path(project_root) / "Week 3" / "Week3.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace = {"erf": erf, "exp": exp, "log": log, "sqrt": sqrt, "np": np}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if any(f"def {name}(" in source for name in REQUIRED):
            definitions = []
            current = []
            for line in source.splitlines():
                if line.startswith("def ") and current:
                    definitions.append("\n".join(current))
                    current = [line]
                elif current:
                    if line and not line.startswith((" ", "\t")):
                        definitions.append("\n".join(current))
                        current = []
                    else:
                        current.append(line)
                elif line.startswith("def "):
                    current = [line]
            if current:
                definitions.append("\n".join(current))
            for definition in definitions:
                if definition.startswith("def "):
                    exec(definition, namespace)
    missing = sorted(REQUIRED - namespace.keys())
    if missing:
        raise RuntimeError(f"Week 3 functions not found: {missing}")
    return {name: namespace[name] for name in REQUIRED}
