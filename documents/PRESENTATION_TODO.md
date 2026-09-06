# Presentation — to finalize

The assignment requires a use-case presentation (PPT exported to PDF) in this
`documents/` folder, with output screenshots of the running app.

**Status:** content is ready to assemble; screenshots are still needed.

## What's needed

Run the app (`docker-compose up`, or `streamlit run app/streamlit_app.py`)
on the **real** Kaggle data and capture one screenshot of each tab:

1. **EDA** — dataset summary metrics + the insight charts
2. **Risk Prediction** — a scored applicant showing probability + risk band
3. **Explainability** — the SHAP per-applicant factor list + global importance
4. **Business Rules** — the surrogate-tree rules + threshold-bin table
5. **Chatbot** — one canonical question answered (SQL + result table shown)

Drop the images in `documents/screenshots/` (or send them over) and they get
assembled into `documents/project_presentation.pdf` covering: use case &
business context, architecture, EDA highlights, model + calibration results
(ROC-AUC 0.763), explainability, rules, talk-to-data design, limitations.
