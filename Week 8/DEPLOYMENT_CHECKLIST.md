# Deployment checklist

- [x] Self-contained pricing engine, models, data cache and requirements
- [x] Local FRED credentials configured securely; public CSV fallback available
- [x] Full feature refresh with bounded retries and independent source caches
- [x] Explicit online/cached data status
- [x] Local health test
- [x] Original project pushed to GitHub
- [x] Export revised reports to PDF and publish the current code/report revision
- [x] Verify repository checks, application/refresh tests, weekly preprocessing and market refresh on GitHub Actions
- [ ] Connect the repository to Streamlit Community Cloud
- [ ] Record the resulting public URL in the final submission
- [ ] Re-test the public URL in a private browser window

The code and all 15 publication PDFs were synchronized on 2026-09-11. Project validation,
weekly preprocessing and the new market refresh workflow passed on GitHub Actions.
The market refresh schedule is now configured on the default branch. Public app hosting is separate from the
local prototype and still requires a connected hosting account.
