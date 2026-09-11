# Deployment readiness and submission checklist

- [x] Self-contained pricing engine, models, data cache and requirements
- [x] Local FRED credentials configured securely; public CSV fallback available
- [x] Full feature refresh with bounded retries and independent source caches
- [x] Explicit online/cached data status
- [x] Local health test
- [x] Original project pushed to GitHub
- [x] Export revised reports to PDF and publish the current code/report revision
- [x] Verify repository checks, application/refresh tests, weekly preprocessing and market refresh on GitHub Actions
- [ ] Verify that the reviewer can access the GitHub repository and README
- [ ] Record the required 5–10 minute narrated demo using the local application
- [ ] Submit the repository, final report PDF, demo video and presentation deck

The code and all 15 publication PDFs were synchronized on 2026-09-11. Project validation,
weekly preprocessing and the new market refresh workflow passed on GitHub Actions.
The market refresh schedule is now configured on the default branch. The project brief
requires a fully deployable pricing tool with a GitHub repository and README. It does
not require a public application URL or cloud hosting. The bundled app supports local
execution and demo recording. Public hosting is an optional extension, not an outstanding deliverable.
