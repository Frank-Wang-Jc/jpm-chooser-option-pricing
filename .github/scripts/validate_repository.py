from __future__ import annotations

import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
week_dirs = [ROOT / "Week1", ROOT / "Week2"] + [ROOT / f"Week {week}" for week in range(3, 9)]

for directory in week_dirs:
    if not directory.is_dir():
        raise AssertionError(f"Missing project directory: {directory.name}")

notebooks = [path for directory in week_dirs for path in directory.glob("*.ipynb")]
if len(notebooks) != 9:
    raise AssertionError(f"Expected 9 notebooks, found {len(notebooks)}")

code_cells = 0
for path in notebooks:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code" or not "".join(cell.get("source", [])).strip():
            continue
        code_cells += 1
        if cell.get("execution_count") is None:
            raise AssertionError(f"Unexecuted code cell: {path.name} cell {index}")
        errors = [output for output in cell.get("outputs", []) if output.get("output_type") == "error"]
        if errors:
            raise AssertionError(f"Saved error output: {path.name} cell {index}")

required_files = [
    ROOT / "Week2" / "processed_data" / "market_data_processed.csv",
    ROOT / "Week 3" / "parameter_config.json",
    ROOT / "Week 4" / "Week4_Baseline_Evaluation.ipynb",
    ROOT / "Week 5" / "Week5_ML_Design_Implementation.ipynb",
    ROOT / "Week 6" / "trained_models" / "best_volatility_model.pkl",
    ROOT / "Week 6" / "trained_models" / "best_direct_pricing_model.pkl",
    ROOT / "Week 7" / "app.py",
    ROOT / "Week 8" / "pricing_tool" / "app.py",
    ROOT / "Reports" / "Week8 Report" / "Final Project Report.docx",
    ROOT / "Reports" / "Week8 Report" / "Final Presentation.pptx",
]
for path in required_files:
    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError(f"Missing or empty deliverable: {path.relative_to(ROOT)}")

for path in required_files[-2:]:
    with zipfile.ZipFile(path) as archive:
        broken = archive.testzip()
        if broken:
            raise AssertionError(f"Corrupt Office package {path.name}: {broken}")

pptx = required_files[-1]
with zipfile.ZipFile(pptx) as archive:
    names = archive.namelist()
slides = [name for name in names if name.startswith("ppt/slides/slide") and name.endswith(".xml")]
notes = [name for name in names if name.startswith("ppt/notesSlides/notesSlide") and name.endswith(".xml")]
if len(slides) != 12 or len(notes) != 12:
    raise AssertionError(f"Expected 12 slides and notes; found {len(slides)} slides and {len(notes)} notes")

print({
    "status": "PASS",
    "notebooks": len(notebooks),
    "code_cells": code_cells,
    "required_deliverables": len(required_files),
    "presentation_slides": len(slides),
})
