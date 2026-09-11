# Deployment checklist

- [x] Self-contained pricing engine, models, data cache and requirements
- [x] Local FRED credentials configured securely; public CSV fallback available
- [x] Full feature refresh with bounded retries and independent source caches
- [x] Explicit online/cached data status
- [x] Local health test
- [x] Original project pushed to GitHub
- [ ] Export revised reports to PDF, then publish the current code/report revision
- [ ] Connect the repository to Streamlit Community Cloud
- [ ] Record the resulting public URL in the final submission
- [ ] Re-test the public URL in a private browser window

The current revision is validated locally. Its new daily refresh workflow will become
active after the pending GitHub publication. Public app hosting is separate from the
local prototype and still requires a connected hosting account.
