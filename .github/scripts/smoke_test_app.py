from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "Week 8" / "pricing_tool"
sys.path.insert(0, str(APP_DIR))

from pricing_engine import price_contract  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402


result = price_contract(231.07756)
if not result["ml_available"]:
    raise AssertionError(result.get("ml_warning"))
if not (result["bsm_price"] > 0 and result["best_ml_price"] > 0):
    raise AssertionError("Pricing engine returned non-positive values")
if not (0 <= result["error_margin_90"][0] <= result["error_margin_90"][1]):
    raise AssertionError("Invalid empirical error interval")

app = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=30)
app.run()
if app.exception:
    raise AssertionError([exception.value for exception in app.exception])
expected_tabs = ["Dual pricing", "Sensitivity", "Model performance", "Methods & limits"]
if [tab.label for tab in app.tabs] != expected_tabs:
    raise AssertionError("Unexpected Streamlit tab structure")
if len(app.metric) != 7:
    raise AssertionError(f"Expected 7 metrics, found {len(app.metric)}")

print({
    "status": "PASS",
    "bsm_price": result["bsm_price"],
    "best_ml_price": result["best_ml_price"],
    "tabs": len(app.tabs),
    "metrics": len(app.metric),
})
