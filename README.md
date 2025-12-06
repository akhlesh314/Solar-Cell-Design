# Solar-Cell-Design


This is a Streamlit application that helps you analyze CSV datasets using an LLM-driven, reproducible data-science pipeline.  
All LLM prompts follow the **P.E.A.C.E. Prompt Framework** (Purpose, Expectations, Actions, Constraints, Evaluation).

## Features
- Upload CSV datasets
- Quick dataset preview (`head`, `info`)
- Automated dataset summary (missing values, stats, correlations)
- LLM-driven agentic analysis that returns:
  - Data cleaning steps
  - Feature engineering suggestions
  - Model selection rationale (classification vs. regression)
  - Full Python training code (pandas + scikit-learn + matplotlib)
  - Evaluation metric explanations
- Downloadable agent report and code snippets

## Files
- `app.py` — main Streamlit app
- `requirements.txt` — Python dependencies
- `README.md` — this file
- `train.py` — helper to run a generated code snippet locally (optional)

## Deployment (Render)
1. Add repository to Render and create a new Web Service.
2. Build command:
