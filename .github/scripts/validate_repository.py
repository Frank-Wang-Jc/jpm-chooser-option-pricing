from __future__ import annotations

import json
import subprocess
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
]
report_paths = [
    "Reports/Project Closure Report.pdf",
    "Reports/Week1 Report/Data Specification Document.pdf",
    "Reports/Week4 Report/Model Validation Report.pdf",
    "Reports/Week4 Report/Performance Benchmark Documentation.pdf",
    "Reports/Week6 Report/Model Performance Report.pdf",
    "Reports/Week7 Report/Comprehensive Sensitivity Analysis Report.pdf",
    "Reports/Week8 Report/Final Project Report.pdf",
]
report_paths.extend(
    f"Reports/Week{week} Report/Week {week} Weekly Report.pdf"
    for week in range(1, 9)
)
required_files.extend(ROOT / path for path in report_paths)
for path in required_files:
    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError(f"Missing or empty deliverable: {path.relative_to(ROOT)}")

for relative_path in report_paths:
    path = ROOT / relative_path
    content = path.read_bytes()
    if not content.startswith(b"%PDF-") or b"%%EOF" not in content[-4096:]:
        raise AssertionError(f"Invalid or incomplete PDF: {relative_path}")

# Inspect the index so local-only documents do not affect this publication check.
tracked_reports = set(subprocess.check_output(
    ["git", "ls-files", "-z", "--", "Reports"], cwd=ROOT,
).decode("utf-8").rstrip("\0").split("\0"))
if tracked_reports != set(report_paths):
    raise AssertionError(
        f"Report publication mismatch: extra={sorted(tracked_reports - set(report_paths))}; "
        f"missing={sorted(set(report_paths) - tracked_reports)}"
    )

print({
    "status": "PASS",
    "notebooks": len(notebooks),
    "code_cells": code_cells,
    "required_deliverables": len(required_files),
    "published_pdf_reports": len(report_paths),
})
