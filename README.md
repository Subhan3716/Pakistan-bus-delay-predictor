# Pakistan Bus Delay Predictor

Pakistan Bus Delay Predictor is a Streamlit dashboard built from the semester project notebook. It explores intercity bus delay patterns in Pakistan, compares multiple regression models, and lets users run live delay predictions from the trained app bundle.

## Highlights

- Interactive Streamlit UI with filters, charts, and model diagnostics
- Exploratory analysis for weather, day, route, and delay categories
- Probability and hypothesis-testing views for the dataset
- Model comparison across Ridge, Lasso, Gradient Boosting, and XGBoost
- Live delay prediction with uncertainty intervals
- Built-in dataset loading from the bundled Excel file

## Project Structure

- `streamlit_app.py` - Streamlit user interface
- `app_core.py` - data loading, feature engineering, model training, and plotting helpers
- `requirements.txt` - pinned Python dependencies
- `PROB&STATS_SEMESTER_PROJECT/PROB&STATS_SEMESTER_PROJECT/Pakistan_Bus_Delay_Dataset.xlsx` - source dataset
- `PROB&STATS_SEMESTER_PROJECT/PROB&STATS_SEMESTER_PROJECT/Prob&Stats_SemesterProject.ipynb` - original semester notebook
- `.streamlit/config.toml` - Streamlit theme and server settings

## Requirements

- Python 3.11 or newer recommended
- Internet connection on first run if Streamlit needs to fetch fonts

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the App

```powershell
streamlit run streamlit_app.py
```

The first launch may take a little longer because the app loads the dataset, builds engineered features, and trains the candidate models before selecting the best one.

## What the App Does

1. Loads and cleans the bus delay dataset.
2. Creates engineered features for weather, route, traffic, and trip context.
3. Compares multiple regression models and keeps the best performer by RMSE.
4. Shows summary statistics, hypothesis tests, and visual explorations.
5. Predicts delay for user-defined trip scenarios with a confidence interval.

## Deployment Notes

- This repository is ready for Streamlit Community Cloud.
- Set the app entry point to `streamlit_app.py`.
- Streamlit will install dependencies from `requirements.txt`.
- The theme and server config are already included in `.streamlit/config.toml`.

## Data Notes

- The dataset is stored locally in the repository, so the app does not need an external database.
- If you replace the dataset, keep the same sheet name and columns expected by `app_core.py`.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
