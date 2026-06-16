# Pakistan Bus Delay Predictor

This repo now includes a Streamlit app built from the semester project notebook.

## Files

- `streamlit_app.py`: Streamlit UI
- `app_core.py`: data loading, analytics, feature engineering, modeling, and prediction logic
- `requirements.txt`: deployment dependencies
- `PROB&STATS_SEMESTER_PROJECT/PROB&STATS_SEMESTER_PROJECT/Pakistan_Bus_Delay_Dataset.xlsx`: source dataset

## Local run

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\streamlit run streamlit_app.py
```

## Streamlit deployment

1. Push this folder to a GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Create a new app and point it to `streamlit_app.py`.
4. Streamlit will install `requirements.txt` automatically.

## Notes

- The original notebook remains unchanged.
- The deployed app evaluates multiple models and uses the lowest-RMSE winner from the current environment.
- The dataset is loaded directly from the bundled Excel file, so no extra configuration is required.
