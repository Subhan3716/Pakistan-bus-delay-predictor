from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, ttest_ind
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = (
    PROJECT_ROOT
    / "PROB&STATS_SEMESTER_PROJECT"
    / "PROB&STATS_SEMESTER_PROJECT"
    / "Pakistan_Bus_Delay_Dataset.xlsx"
)
SHEET_NAME = "Bus_Delay_Data"

COL = {
    "delay": "Delay_Minutes",
    "weather": "Weather",
    "day": "Day",
    "distance": "Distance_KM",
    "departure": "Departure_Hour",
    "event": "Special_Event",
    "route": "Route",
    "bus_type": "Bus_Type",
    "road_type": "Road_Type",
    "traffic": "Traffic_Level",
}
TARGET = COL["delay"]
LATE_THRESHOLD = 15
DELAY_BINS = [-np.inf, 0, 5, 15, 30, 60, np.inf]
DELAY_LABELS = [
    "On time",
    "Slight (1-5)",
    "Minor (6-15)",
    "Moderate (16-30)",
    "Severe (31-60)",
    "Extreme (60+)",
]
BASE_NUMERIC_FEATURES = [
    "Distance_KM",
    "Avg_Speed",
    "Fuel_Stops",
    "Traffic_Score",
    "Weather_Impact",
    "Stops",
    "Maintenance_Score",
    "Peak_Hour",
    "Driver_Experience_Years",
    "Road_Quality_Index",
    "Special_Event",
    "Road_Condition",
    "Bus_Age_Years",
    "Passenger_Load",
    "Departure_Hour",
    "Weekend",
]
PROTECTED_FEATURES = {
    "Distance_KM",
    "Avg_Speed",
    "Fuel_Stops",
    "Traffic_Score",
    "Weather_Impact",
}
SCALED_MODELS = ("Ridge Regression", "Lasso Regression")

sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    }
)


@dataclass
class AppBundle:
    data_path: Path
    df: pd.DataFrame
    feature_columns: list[str]
    metrics: pd.DataFrame
    best_model_name: str
    best_model: Any
    scaler: StandardScaler
    residuals: np.ndarray
    y_test: np.ndarray
    best_predictions: np.ndarray
    feature_scores: pd.Series
    numeric_defaults: dict[str, float]
    numeric_ranges: dict[str, tuple[float, float]]
    weather_impact_defaults: dict[str, float]
    traffic_score_defaults: dict[str, float]
    distribution_fits: pd.DataFrame
    conditional_late_table: pd.DataFrame
    delay_frequency: pd.DataFrame
    pivot_weather_day: pd.DataFrame
    route_stats: pd.DataFrame
    summary_stats: dict[str, float]
    hypothesis_tests: dict[str, Any]
    train_rows: int
    validation_rows: int
    test_rows: int
    late_threshold: int = LATE_THRESHOLD


def load_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(data_path, sheet_name=SHEET_NAME, engine="openpyxl")
    for key, column in COL.items():
        if column not in df.columns:
            raise KeyError(f"Missing expected column for {key!r}: {column}")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] >= 0]
    df = df.drop_duplicates().reset_index(drop=True)
    df["Late_15_Min"] = (df[TARGET] > LATE_THRESHOLD).astype(int)
    df["Delay_Category"] = pd.cut(
        df[TARGET], bins=DELAY_BINS, labels=DELAY_LABELS
    )
    return df


def _compute_vif_table(features: pd.DataFrame) -> pd.DataFrame:
    values = features.values.astype(float)
    if values.shape[1] == 1:
        return pd.DataFrame({"feature": features.columns, "VIF": [1.0]})

    scores: list[float] = []
    for index in range(values.shape[1]):
        try:
            score = float(variance_inflation_factor(values, index))
        except Exception:
            score = float("inf")
        scores.append(score)
    return pd.DataFrame({"feature": features.columns, "VIF": scores})


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    for column in BASE_NUMERIC_FEATURES:
        if column in df.columns:
            numeric_col = pd.to_numeric(df[column], errors="coerce")
            features[column] = numeric_col.fillna(numeric_col.median())

    for key in ("weather", "day", "bus_type", "road_type", "traffic"):
        column = COL[key]
        if column in df.columns and df[column].dtype == object:
            dummies = pd.get_dummies(df[column], prefix=key, drop_first=True, dtype=float)
            features = pd.concat([features, dummies], axis=1)

    features = features.fillna(0)
    features = features.loc[:, features.std() > 0]

    vif_table = _compute_vif_table(features)
    keepers = vif_table[
        (vif_table["VIF"] <= 10) | (vif_table["feature"].isin(PROTECTED_FEATURES))
    ]["feature"].tolist()
    features = features[keepers].copy()

    rain_col = next((col for col in features.columns if "weather_Rain" in col), None)
    storm_col = next((col for col in features.columns if "weather_Storm" in col), None)

    if rain_col and "Peak_Hour" in features.columns:
        features["rain_x_peak"] = features[rain_col] * features["Peak_Hour"]
    if storm_col and "Peak_Hour" in features.columns:
        features["storm_x_peak"] = features[storm_col] * features["Peak_Hour"]
    if {"Distance_KM", "Weather_Impact"}.issubset(features.columns):
        features["dist_x_weather"] = (
            features["Distance_KM"] * features["Weather_Impact"]
        ) / 1000
    if {"Stops", "Traffic_Score"}.issubset(features.columns):
        features["stops_x_traffic"] = features["Stops"] * features["Traffic_Score"]
    if {"Bus_Age_Years", "Maintenance_Score"}.issubset(features.columns):
        features["age_x_maintenance"] = (
            features["Bus_Age_Years"] * (10 - features["Maintenance_Score"])
        ) / 10
    if {"Fuel_Stops", "Distance_KM"}.issubset(features.columns):
        features["fuel_x_dist"] = (
            features["Fuel_Stops"] * features["Distance_KM"]
        ) / 1000

    return features


def _fit_models(features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    x_matrix = features.values.astype(float)
    y_values = target.values.astype(float)

    x_train, x_test, y_train, y_test = train_test_split(
        x_matrix, y_values, test_size=0.15, random_state=42
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.176, random_state=42
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    model_specs = [
        (
            "Ridge Regression",
            RidgeCV(alphas=np.logspace(-3, 3, 50), cv=kfold),
            x_train_scaled,
            x_test_scaled,
            True,
        ),
        (
            "Lasso Regression",
            LassoCV(alphas=np.logspace(-3, 1, 50), cv=kfold, max_iter=5000),
            x_train_scaled,
            x_test_scaled,
            True,
        ),
        (
            "Gradient Boosting",
            GradientBoostingRegressor(
                n_estimators=600,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                min_samples_leaf=15,
                max_features="sqrt",
                random_state=42,
            ),
            x_train,
            x_test,
            False,
        ),
        (
            "XGBoost",
            XGBRegressor(
                n_estimators=700,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.7,
                colsample_bylevel=0.7,
                reg_alpha=0.05,
                reg_lambda=1.0,
                min_child_weight=5,
                gamma=0.1,
                random_state=42,
                verbosity=0,
                n_jobs=1,
            ),
            x_train,
            x_test,
            False,
        ),
    ]

    metrics_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    predictions_by_model: dict[str, np.ndarray] = {}

    for name, model, x_fit, x_holdout, scaled in model_specs:
        model.fit(x_fit, y_train)
        predictions = model.predict(x_holdout)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        mae = float(mean_absolute_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))
        cv_rmse = float(
            -cross_val_score(
                model,
                x_fit,
                y_train,
                cv=kfold,
                scoring="neg_root_mean_squared_error",
            ).mean()
        )
        metrics_rows.append(
            {
                "Model": name,
                "Scaled": scaled,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
                "CV_RMSE": cv_rmse,
            }
        )
        fitted_models[name] = model
        predictions_by_model[name] = predictions

    metrics = pd.DataFrame(metrics_rows).sort_values("RMSE").reset_index(drop=True)
    best_model_name = str(metrics.iloc[0]["Model"])
    best_model = fitted_models[best_model_name]
    best_predictions = predictions_by_model[best_model_name]
    residuals = y_test - best_predictions

    if hasattr(best_model, "feature_importances_"):
        feature_scores = pd.Series(
            best_model.feature_importances_, index=features.columns, name="score"
        )
    else:
        feature_scores = pd.Series(
            np.abs(best_model.coef_), index=features.columns, name="score"
        )

    return {
        "metrics": metrics,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "scaler": scaler,
        "residuals": residuals,
        "y_test": y_test,
        "best_predictions": best_predictions,
        "feature_scores": feature_scores.sort_values(ascending=False),
        "train_rows": len(x_train),
        "validation_rows": len(x_val),
        "test_rows": len(x_test),
    }


def _distribution_fit_table(delay: pd.Series) -> pd.DataFrame:
    positive_delay = delay[delay > 0]
    sample = positive_delay.sample(min(3000, len(positive_delay)), random_state=42)
    candidates = {
        "Normal": ("norm", stats.norm.fit(sample)),
        "Lognormal": ("lognorm", stats.lognorm.fit(sample, floc=0)),
        "Gamma": ("gamma", stats.gamma.fit(sample, floc=0)),
        "Exponential": ("expon", stats.expon.fit(sample)),
    }

    rows: list[dict[str, Any]] = []
    for label, (dist_name, params) in candidates.items():
        ks_stat, p_value = stats.kstest(sample, dist_name, args=params)
        rows.append(
            {
                "Distribution": label,
                "KS_Statistic": float(ks_stat),
                "P_Value": float(p_value),
                "Parameters": ", ".join(f"{value:.4f}" for value in params),
            }
        )

    return pd.DataFrame(rows).sort_values("KS_Statistic").reset_index(drop=True)


def _conditional_late_table(df: pd.DataFrame) -> pd.DataFrame:
    table = (
        df.groupby(COL["weather"])["Late_15_Min"]
        .agg(Late_Probability="mean", Trips="count", Avg_Delay="mean")
        .reset_index()
    )
    table["Late_Probability"] = table["Late_Probability"].round(4)
    table["Avg_Delay"] = table["Avg_Delay"].round(2)
    return table.sort_values("Late_Probability", ascending=False).reset_index(drop=True)


def _delay_frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    frequency = df["Delay_Category"].value_counts().reindex(DELAY_LABELS)
    frequency_pct = (frequency / frequency.sum() * 100).round(2)
    cumulative_pct = frequency_pct.cumsum().round(2)
    return pd.DataFrame(
        {"Count": frequency, "Percent": frequency_pct, "Cumulative": cumulative_pct}
    )


def _pivot_weather_day(df: pd.DataFrame) -> pd.DataFrame:
    return pd.pivot_table(
        df,
        values=TARGET,
        index=COL["weather"],
        columns=COL["day"],
        aggfunc="mean",
    ).round(1)


def _route_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(COL["route"])[TARGET]
        .agg(
            trips="count",
            mean_delay="mean",
            median_delay="median",
            std_delay="std",
            p90_delay=lambda values: np.percentile(values.values, 90),
        )
        .round(2)
        .sort_values("mean_delay", ascending=False)
    )


def make_weather_day_pivot(df: pd.DataFrame) -> pd.DataFrame:
    return _pivot_weather_day(df)


def make_route_stats(df: pd.DataFrame) -> pd.DataFrame:
    return _route_stats(df)


def _summary_stats(df: pd.DataFrame) -> dict[str, float]:
    delay = df[TARGET]
    return {
        "rows": float(len(df)),
        "routes": float(df[COL["route"]].nunique()),
        "mean_delay": float(delay.mean()),
        "median_delay": float(delay.median()),
        "on_time_rate": float((delay <= 5).mean()),
        "late_rate": float((delay > LATE_THRESHOLD).mean()),
        "p95_delay": float(np.percentile(delay, 95)),
        "avg_distance": float(df[COL["distance"]].mean()),
    }


def _hypothesis_tests(df: pd.DataFrame) -> dict[str, Any]:
    results: dict[str, Any] = {}

    weather_groups = [
        group[TARGET].values for _, group in df.groupby(COL["weather"]) if len(group) > 10
    ]
    if len(weather_groups) == 2:
        t_stat, p_value = ttest_ind(*weather_groups, equal_var=False)
        results["weather_test"] = {
            "name": "Welch t-test by weather",
            "statistic": float(t_stat),
            "p_value": float(p_value),
        }
    elif len(weather_groups) > 2:
        f_stat, p_value = f_oneway(*weather_groups)
        results["weather_test"] = {
            "name": "ANOVA by weather",
            "statistic": float(f_stat),
            "p_value": float(p_value),
        }

    day_groups = [group[TARGET].values for _, group in df.groupby(COL["day"]) if len(group) > 10]
    if len(day_groups) > 2:
        f_stat, p_value = f_oneway(*day_groups)
        results["day_test"] = {
            "name": "ANOVA by day",
            "statistic": float(f_stat),
            "p_value": float(p_value),
        }

    contingency = pd.crosstab(df[COL["weather"]], df["Late_15_Min"])
    chi2_stat, p_value, dof, _ = chi2_contingency(contingency)
    results["chi_square"] = {
        "name": "Chi-square weather x late",
        "statistic": float(chi2_stat),
        "p_value": float(p_value),
        "dof": int(dof),
    }

    std_error = df[TARGET].std() / np.sqrt(len(df))
    t_critical = stats.t.ppf(0.975, df=len(df) - 1)
    results["mean_delay_ci"] = (
        float(df[TARGET].mean() - t_critical * std_error),
        float(df[TARGET].mean() + t_critical * std_error),
    )
    return results


def _numeric_defaults(df: pd.DataFrame) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    defaults: dict[str, float] = {}
    ranges: dict[str, tuple[float, float]] = {}
    for column in BASE_NUMERIC_FEATURES:
        if column in df.columns:
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            defaults[column] = float(series.median())
            ranges[column] = (float(series.min()), float(series.max()))
    return defaults, ranges


def build_app_bundle() -> AppBundle:
    df = load_dataset()
    features = build_feature_frame(df)
    model_outputs = _fit_models(features, df[TARGET])
    numeric_defaults, numeric_ranges = _numeric_defaults(df)

    weather_defaults = (
        df.groupby(COL["weather"])["Weather_Impact"].median().round(2).to_dict()
    )
    traffic_defaults = (
        df.groupby(COL["traffic"])["Traffic_Score"].median().round(2).to_dict()
    )

    return AppBundle(
        data_path=DATA_PATH,
        df=df,
        feature_columns=features.columns.tolist(),
        metrics=model_outputs["metrics"],
        best_model_name=model_outputs["best_model_name"],
        best_model=model_outputs["best_model"],
        scaler=model_outputs["scaler"],
        residuals=model_outputs["residuals"],
        y_test=model_outputs["y_test"],
        best_predictions=model_outputs["best_predictions"],
        feature_scores=model_outputs["feature_scores"],
        numeric_defaults=numeric_defaults,
        numeric_ranges=numeric_ranges,
        weather_impact_defaults=weather_defaults,
        traffic_score_defaults=traffic_defaults,
        distribution_fits=_distribution_fit_table(df[TARGET]),
        conditional_late_table=_conditional_late_table(df),
        delay_frequency=_delay_frequency_table(df),
        pivot_weather_day=_pivot_weather_day(df),
        route_stats=_route_stats(df),
        summary_stats=_summary_stats(df),
        hypothesis_tests=_hypothesis_tests(df),
        train_rows=model_outputs["train_rows"],
        validation_rows=model_outputs["validation_rows"],
        test_rows=model_outputs["test_rows"],
    )


def apply_filters(
    df: pd.DataFrame,
    weather: list[str] | None = None,
    days: list[str] | None = None,
    bus_types: list[str] | None = None,
    road_types: list[str] | None = None,
    routes: list[str] | None = None,
    delay_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    filtered = df.copy()
    if weather:
        filtered = filtered[filtered[COL["weather"]].isin(weather)]
    if days:
        filtered = filtered[filtered[COL["day"]].isin(days)]
    if bus_types:
        filtered = filtered[filtered[COL["bus_type"]].isin(bus_types)]
    if road_types:
        filtered = filtered[filtered[COL["road_type"]].isin(road_types)]
    if routes:
        filtered = filtered[filtered[COL["route"]].isin(routes)]
    if delay_range:
        low, high = delay_range
        filtered = filtered[filtered[TARGET].between(low, high)]
    return filtered.reset_index(drop=True)


def predict_delay(bundle: AppBundle, user_inputs: dict[str, float]) -> dict[str, float | tuple[float, float]]:
    row = {column: 0.0 for column in bundle.feature_columns}
    for key, value in user_inputs.items():
        if key in row:
            row[key] = float(value)

    if "dist_x_weather" in row:
        row["dist_x_weather"] = (row.get("Distance_KM", 0.0) * row.get("Weather_Impact", 0.0)) / 1000
    if "stops_x_traffic" in row:
        row["stops_x_traffic"] = row.get("Stops", 0.0) * row.get("Traffic_Score", 0.0)
    if "age_x_maintenance" in row:
        row["age_x_maintenance"] = (
            row.get("Bus_Age_Years", 0.0) * (10 - row.get("Maintenance_Score", 0.0))
        ) / 10
    if "fuel_x_dist" in row:
        row["fuel_x_dist"] = (row.get("Fuel_Stops", 0.0) * row.get("Distance_KM", 0.0)) / 1000
    if "rain_x_peak" in row:
        row["rain_x_peak"] = row.get("weather_Rain", 0.0) * row.get("Peak_Hour", 0.0)
    if "storm_x_peak" in row:
        row["storm_x_peak"] = row.get("weather_Storm", 0.0) * row.get("Peak_Hour", 0.0)

    x_pred = np.array([[row[column] for column in bundle.feature_columns]])
    if bundle.best_model_name in SCALED_MODELS:
        x_pred = bundle.scaler.transform(x_pred)
    prediction = float(bundle.best_model.predict(x_pred)[0])

    rng = np.random.default_rng(42)
    simulated = prediction + rng.choice(bundle.residuals, size=4000, replace=True)
    interval_low, interval_high = np.percentile(simulated, [2.5, 97.5])

    return {
        "predicted_delay_min": round(prediction, 1),
        "95_PI": (round(max(0.0, float(interval_low)), 1), round(float(interval_high), 1)),
        "p_more_than_15min_late": round(float((simulated > 15).mean()), 4),
        "p_on_time": round(float((simulated <= 5).mean()), 4),
    }


def contextual_history(df: pd.DataFrame, weather: str, day: str) -> dict[str, float]:
    subset = df[(df[COL["weather"]] == weather) & (df[COL["day"]] == day)]
    if subset.empty:
        return {"trips": 0.0, "avg_delay": float("nan"), "late_rate": float("nan")}
    return {
        "trips": float(len(subset)),
        "avg_delay": float(subset[TARGET].mean()),
        "late_rate": float((subset[TARGET] > LATE_THRESHOLD).mean()),
    }


def plot_delay_distribution(df: pd.DataFrame) -> plt.Figure:
    delay = df[TARGET]
    q1, q3 = delay.quantile([0.25, 0.75])
    iqr = q3 - q1
    iqr_mask = (delay < q1 - 1.5 * iqr) | (delay > q3 + 1.5 * iqr)

    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    figure.suptitle("Delay Distribution Profile", fontsize=14, fontweight="bold")

    axes[0, 0].hist(delay, bins=min(50, max(5, len(delay) // 4)), density=True, color="#5B8DB8", alpha=0.85, edgecolor="white")
    if len(delay) > 2 and delay.nunique() > 1:
        grid = np.linspace(delay.min(), delay.max(), 300)
        axes[0, 0].plot(grid, stats.gaussian_kde(delay)(grid), color="#C7683D", lw=2)
    axes[0, 0].axvline(delay.mean(), color="#A23E48", ls="--", lw=1.5, label=f"Mean {delay.mean():.1f}")
    axes[0, 0].axvline(delay.median(), color="#2F6F57", ls=":", lw=1.8, label=f"Median {delay.median():.1f}")
    axes[0, 0].set_title("Histogram and KDE")
    axes[0, 0].set_xlabel("Delay (minutes)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend(fontsize=9)

    log_delay = np.log1p(delay)
    axes[0, 1].hist(log_delay, bins=min(45, max(5, len(log_delay) // 4)), density=True, color="#D2A24C", alpha=0.85, edgecolor="white")
    if len(log_delay) > 2 and log_delay.nunique() > 1:
        log_grid = np.linspace(log_delay.min(), log_delay.max(), 300)
        axes[0, 1].plot(log_grid, stats.gaussian_kde(log_delay)(log_grid), color="#345B8C", lw=2)
    axes[0, 1].set_title("log(1 + delay)")
    axes[0, 1].set_xlabel("Transformed delay")
    axes[0, 1].set_ylabel("Density")

    axes[1, 0].boxplot(
        delay,
        vert=True,
        patch_artist=True,
        boxprops={"facecolor": "#8DC6B8", "alpha": 0.7},
        medianprops={"color": "#A23E48", "lw": 2},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.35},
    )
    axes[1, 0].set_title(f"Boxplot with {int(iqr_mask.sum())} IQR outliers")
    axes[1, 0].set_ylabel("Delay (minutes)")

    sorted_delay = np.sort(delay.values)
    ecdf = np.arange(1, len(sorted_delay) + 1) / len(sorted_delay)
    axes[1, 1].plot(sorted_delay, ecdf, color="#345B8C", lw=1.8)
    axes[1, 1].set_title("Empirical CDF")
    axes[1, 1].set_xlabel("Delay (minutes)")
    axes[1, 1].set_ylabel("Cumulative probability")

    figure.tight_layout()
    return figure


def plot_category_delays(df: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    weather_order = (
        df.groupby(COL["weather"])[TARGET].median().sort_values(ascending=False).index
    )
    day_order = df.groupby(COL["day"])[TARGET].median().sort_values(ascending=False).index

    sns.boxplot(
        data=df,
        x=COL["weather"],
        y=TARGET,
        hue=COL["weather"],
        order=weather_order,
        ax=axes[0],
        palette="crest",
        dodge=False,
        legend=False,
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.35},
    )
    axes[0].set_title("Delay by Weather")
    axes[0].tick_params(axis="x", rotation=25)

    sns.boxplot(
        data=df,
        x=COL["day"],
        y=TARGET,
        hue=COL["day"],
        order=day_order,
        ax=axes[1],
        palette="rocket",
        dodge=False,
        legend=False,
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.35},
    )
    axes[1].set_title("Delay by Day")
    axes[1].tick_params(axis="x", rotation=25)

    figure.tight_layout()
    return figure


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_columns].corr().round(2)

    figure, axis = plt.subplots(figsize=(10, 7))
    if corr.empty or corr.isna().all().all():
        axis.text(0.5, 0.5, "Not enough variation for a correlation heatmap.", ha="center", va="center")
        axis.axis("off")
    else:
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            linewidths=0.5,
            ax=axis,
            annot_kws={"size": 8},
        )
        axis.set_title("Correlation Heatmap")
    figure.tight_layout()
    return figure


def plot_weather_day_pivot(pivot_table: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9, 4.5))
    if pivot_table.empty or pivot_table.isna().all().all():
        axis.text(0.5, 0.5, "No weather/day combinations available for this view.", ha="center", va="center")
        axis.axis("off")
    else:
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".1f",
            cmap="YlOrBr",
            linewidths=0.4,
            ax=axis,
        )
        axis.set_title("Mean Delay by Weather and Day")
    figure.tight_layout()
    return figure


def plot_route_profile(route_stats: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    if route_stats.empty:
        for axis in axes:
            axis.text(0.5, 0.5, "No route statistics available for this view.", ha="center", va="center")
            axis.axis("off")
    else:
        top_routes = route_stats.head(12).sort_values("mean_delay")
        axes[0].barh(top_routes.index, top_routes["mean_delay"], color="#C7683D")
        axes[0].set_title("Top 12 Routes by Mean Delay")
        axes[0].set_xlabel("Mean delay (minutes)")

        axes[1].barh(top_routes.index, top_routes["std_delay"], color="#4F7CAC")
        axes[1].set_title("Delay Volatility on the Same Routes")
        axes[1].set_xlabel("Standard deviation")

    figure.tight_layout()
    return figure


def plot_model_diagnostics(bundle: AppBundle) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    residuals = bundle.y_test - bundle.best_predictions

    axes[0].scatter(bundle.y_test, bundle.best_predictions, alpha=0.35, s=16, color="#345B8C")
    axes[0].plot(
        [bundle.y_test.min(), bundle.y_test.max()],
        [bundle.y_test.min(), bundle.y_test.max()],
        color="#A23E48",
        ls="--",
        lw=1.5,
    )
    axes[0].set_title(f"Predicted vs Actual ({bundle.best_model_name})")
    axes[0].set_xlabel("Actual delay")
    axes[0].set_ylabel("Predicted delay")

    axes[1].scatter(bundle.best_predictions, residuals, alpha=0.35, s=16, color="#C7683D")
    axes[1].axhline(0, color="#A23E48", ls="--", lw=1.5)
    axes[1].set_title("Residuals vs Fitted")
    axes[1].set_xlabel("Predicted delay")
    axes[1].set_ylabel("Residual")

    figure.tight_layout()
    return figure


def plot_feature_scores(bundle: AppBundle) -> plt.Figure:
    scores = bundle.feature_scores.sort_values(ascending=True).tail(12)
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.barh(scores.index, scores.values, color="#2F6F57")
    axis.set_title(f"Top Feature Signals ({bundle.best_model_name})")
    axis.set_xlabel("Magnitude")
    figure.tight_layout()
    return figure
