$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $here
$projectPython = Join-Path $here '..\..\.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $projectPython) {
    & $projectPython -m streamlit run app.py
} else {
    python -m streamlit run app.py
}
