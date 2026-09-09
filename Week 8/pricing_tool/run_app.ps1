$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $here
python -m streamlit run app.py
