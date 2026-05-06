from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from app_core import (
    AppBundle,
    COL,
    TARGET,
    apply_filters,
    build_app_bundle,
    contextual_history,
    make_route_stats,
    make_weather_day_pivot,
    plot_category_delays,
    plot_correlation_heatmap,
    plot_delay_distribution,
    plot_feature_scores,
    plot_model_diagnostics,
    plot_route_profile,
    plot_weather_day_pivot,
    predict_delay,
)


st.set_page_config(
    page_title="Pakistan Bus Delay Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Preparing data, statistics, and trained models...")
def get_bundle() -> AppBundle:
    return build_app_bundle()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
            --sand: #f5efe6;
            --paper: rgba(255, 255, 255, 0.82);
            --ink: #17324d;
            --muted: #5f6b7a;
            --accent: #b85c38;
            --accent-soft: #f0d6c7;
            --forest: #2f6f57;
            --line: rgba(23, 50, 77, 0.12);
        }

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
            color: var(--ink);
        }

        h1, h2, h3, h4 {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: -0.03em;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(240, 214, 199, 0.95) 0%, rgba(245, 239, 230, 0.98) 36%, rgba(229, 238, 234, 0.98) 100%);
        }

        [data-testid="stSidebar"] {
            background: rgba(248, 245, 240, 0.92);
            border-right: 1px solid var(--line);
        }

        .hero {
            padding: 1.5rem 1.7rem;
            border-radius: 24px;
            background:
                linear-gradient(135deg, rgba(23, 50, 77, 0.96) 0%, rgba(52, 91, 140, 0.92) 45%, rgba(47, 111, 87, 0.88) 100%);
            color: #f8f4ef;
            box-shadow: 0 18px 45px rgba(23, 50, 77, 0.18);
            margin-bottom: 1rem;
        }

        .hero p {
            margin: 0.35rem 0 0 0;
            color: rgba(248, 244, 239, 0.88);
            font-size: 1rem;
        }

        .metric-card {
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            box-shadow: 0 10px 26px rgba(23, 50, 77, 0.08);
        }

        .metric-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 0.35rem;
        }

        .metric-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.7rem;
            line-height: 1.1;
            color: var(--ink);
        }

        .metric-note {
            margin-top: 0.3rem;
            color: var(--muted);
            font-size: 0.9rem;
        }

        .section-card {
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem 1.1rem 0.2rem 1.1rem;
            box-shadow: 0 10px 26px rgba(23, 50, 77, 0.06);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.56);
            border-radius: 999px;
            padding: 0.4rem 1rem;
            border: 1px solid var(--line);
        }

        .stTabs [aria-selected="true"] {
            background: rgba(184, 92, 56, 0.12);
            color: var(--accent);
            border-color: rgba(184, 92, 56, 0.22);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(items: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, note) in zip(columns, items):
        column.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def human_pct(value: float) -> str:
    if math.isnan(value):
        return "N/A"
    return f"{value * 100:.1f}%"


def sidebar_filters(bundle: AppBundle) -> pd.DataFrame:
    df = bundle.df
    st.sidebar.markdown("## Filters")

    weather = st.sidebar.multiselect(
        "Weather",
        options=sorted(df[COL["weather"]].dropna().unique().tolist()),
    )
    days = st.sidebar.multiselect(
        "Day",
        options=sorted(df[COL["day"]].dropna().unique().tolist()),
    )
    bus_types = st.sidebar.multiselect(
        "Bus Type",
        options=sorted(df[COL["bus_type"]].dropna().unique().tolist()),
    )
    road_types = st.sidebar.multiselect(
        "Road Type",
        options=sorted(df[COL["road_type"]].dropna().unique().tolist()),
    )
    routes = st.sidebar.multiselect(
        "Routes",
        options=sorted(df[COL["route"]].dropna().unique().tolist()),
    )
    delay_range = st.sidebar.slider(
        "Delay range (minutes)",
        min_value=float(df[TARGET].min()),
        max_value=float(df[TARGET].max()),
        value=(float(df[TARGET].min()), float(df[TARGET].max())),
        step=1.0,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Dataset: `{bundle.data_path.name}`")
    st.sidebar.caption(f"Deployed model: `{bundle.best_model_name}`")

    return apply_filters(
        df,
        weather=weather,
        days=days,
        bus_types=bus_types,
        road_types=road_types,
        routes=routes,
        delay_range=delay_range,
    )


def overview_tab(bundle: AppBundle, filtered_df: pd.DataFrame) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    render_metric_cards(
        [
            ("Trips in view", f"{len(filtered_df):,}", "Filtered dataset rows"),
            ("Average delay", f"{filtered_df[TARGET].mean():.1f} min", "Mean observed delay"),
            ("On-time rate", human_pct((filtered_df[TARGET] <= 5).mean()), "Trips delayed 5 minutes or less"),
            ("Routes", f"{filtered_df[COL['route']].nunique():,}", "Distinct intercity routes"),
        ]
    )

    col1, col2 = st.columns([1.15, 0.85])
    with col1:
        st.subheader("Dataset Snapshot")
        st.dataframe(filtered_df.head(25), width="stretch", height=360)
    with col2:
        st.subheader("Delay Category Mix")
        freq = filtered_df["Delay_Category"].value_counts(normalize=True).reindex(bundle.delay_frequency.index)
        chart_df = (
            freq.fillna(0)
            .rename("share")
            .mul(100)
            .round(2)
            .reset_index()
            .rename(columns={"Delay_Category": "Delay Category"})
        )
        st.bar_chart(chart_df, x="Delay Category", y="share", height=360)

    st.download_button(
        "Download filtered data as CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_bus_delay_data.csv",
        mime="text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)


def exploration_tab(bundle: AppBundle, filtered_df: pd.DataFrame) -> None:
    filtered_pivot = make_weather_day_pivot(filtered_df)
    filtered_route_stats = make_route_stats(filtered_df)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Distribution and Group Patterns")
    st.pyplot(plot_delay_distribution(filtered_df), clear_figure=True, width="stretch")
    st.pyplot(plot_category_delays(filtered_df), clear_figure=True, width="stretch")
    st.pyplot(plot_correlation_heatmap(filtered_df), clear_figure=True, width="stretch")
    st.pyplot(plot_weather_day_pivot(filtered_pivot), clear_figure=True, width="stretch")
    st.pyplot(plot_route_profile(filtered_route_stats), clear_figure=True, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def probability_tab(bundle: AppBundle) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    ci_low, ci_high = bundle.hypothesis_tests["mean_delay_ci"]
    render_metric_cards(
        [
            ("Mean delay", f"{bundle.summary_stats['mean_delay']:.1f} min", "Across all 10,000 trips"),
            ("95th percentile", f"{bundle.summary_stats['p95_delay']:.1f} min", "Tail risk threshold"),
            ("Late probability", human_pct(bundle.summary_stats["late_rate"]), "Delay greater than 15 minutes"),
            ("95% CI", f"{ci_low:.1f} to {ci_high:.1f}", "Confidence interval for mean delay"),
        ]
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution Fit Leaderboard")
        st.dataframe(bundle.distribution_fits, width="stretch", hide_index=True)

        st.subheader("Conditional Probability by Weather")
        st.dataframe(bundle.conditional_late_table, width="stretch", hide_index=True)

    with col2:
        st.subheader("Delay Frequency Table")
        st.dataframe(bundle.delay_frequency, width="stretch")

        st.subheader("Hypothesis Tests")
        test_rows = []
        for key in ("weather_test", "day_test", "chi_square"):
            item = bundle.hypothesis_tests.get(key)
            if not item:
                continue
            test_rows.append(
                {
                    "Test": item["name"],
                    "Statistic": round(item["statistic"], 4),
                    "P Value": round(item["p_value"], 6),
                    "Interpretation": "Significant" if item["p_value"] < 0.05 else "Not significant",
                }
            )
        st.dataframe(pd.DataFrame(test_rows), width="stretch", hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def model_tab(bundle: AppBundle) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    render_metric_cards(
        [
            ("Winning model", bundle.best_model_name, "Lowest RMSE on holdout data"),
            ("Train rows", f"{bundle.train_rows:,}", "Model fitting split"),
            ("Validation rows", f"{bundle.validation_rows:,}", "Model tuning split"),
            ("Test rows", f"{bundle.test_rows:,}", "Final evaluation split"),
        ]
    )

    st.subheader("Model Leaderboard")
    st.dataframe(bundle.metrics, width="stretch", hide_index=True)

    st.subheader("Diagnostics")
    st.pyplot(plot_model_diagnostics(bundle), clear_figure=True, width="stretch")
    st.pyplot(plot_feature_scores(bundle), clear_figure=True, width="stretch")

    with st.expander("Feature set used by the deployed model"):
        st.write(bundle.feature_columns)
    st.markdown("</div>", unsafe_allow_html=True)


def predictor_tab(bundle: AppBundle) -> None:
    df = bundle.df
    weather_options = sorted(df[COL["weather"]].dropna().unique().tolist())
    day_options = sorted(df[COL["day"]].dropna().unique().tolist())
    traffic_options = sorted(df[COL["traffic"]].dropna().unique().tolist())

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Interactive Delay Predictor")
    st.caption(
        "The deployed model keeps the strongest engineered numerical signals after VIF filtering. "
        "Weather, day, and traffic selections below are used for context and for sensible default values."
    )

    left, right = st.columns([1.05, 0.95])
    with left:
        with st.form("prediction_form"):
            selected_weather = st.selectbox("Weather context", weather_options)
            selected_day = st.selectbox("Day context", day_options)
            selected_traffic = st.selectbox("Traffic level context", traffic_options)

            distance = st.slider(
                "Distance (KM)",
                min_value=float(bundle.numeric_ranges["Distance_KM"][0]),
                max_value=float(bundle.numeric_ranges["Distance_KM"][1]),
                value=float(bundle.numeric_defaults["Distance_KM"]),
                step=1.0,
            )
            avg_speed = st.slider(
                "Average speed",
                min_value=float(bundle.numeric_ranges["Avg_Speed"][0]),
                max_value=float(bundle.numeric_ranges["Avg_Speed"][1]),
                value=float(bundle.numeric_defaults["Avg_Speed"]),
                step=0.1,
            )
            departure_hour = st.slider(
                "Departure hour",
                min_value=0,
                max_value=23,
                value=int(round(bundle.numeric_defaults["Departure_Hour"])),
                step=1,
            )
            stops = st.slider(
                "Stops",
                min_value=int(bundle.numeric_ranges["Stops"][0]),
                max_value=int(bundle.numeric_ranges["Stops"][1]),
                value=int(round(bundle.numeric_defaults["Stops"])),
                step=1,
            )
            fuel_stops = st.slider(
                "Fuel stops",
                min_value=int(bundle.numeric_ranges["Fuel_Stops"][0]),
                max_value=int(bundle.numeric_ranges["Fuel_Stops"][1]),
                value=int(round(bundle.numeric_defaults["Fuel_Stops"])),
                step=1,
            )
            traffic_score = st.slider(
                "Traffic score",
                min_value=float(bundle.numeric_ranges["Traffic_Score"][0]),
                max_value=float(bundle.numeric_ranges["Traffic_Score"][1]),
                value=float(bundle.traffic_score_defaults[selected_traffic]),
                step=0.1,
            )
            weather_impact = st.slider(
                "Weather impact",
                min_value=float(bundle.numeric_ranges["Weather_Impact"][0]),
                max_value=float(bundle.numeric_ranges["Weather_Impact"][1]),
                value=float(bundle.weather_impact_defaults[selected_weather]),
                step=0.1,
            )
            maintenance = st.slider(
                "Maintenance score",
                min_value=float(bundle.numeric_ranges["Maintenance_Score"][0]),
                max_value=float(bundle.numeric_ranges["Maintenance_Score"][1]),
                value=float(bundle.numeric_defaults["Maintenance_Score"]),
                step=0.1,
            )
            driver_exp = st.slider(
                "Driver experience (years)",
                min_value=float(bundle.numeric_ranges["Driver_Experience_Years"][0]),
                max_value=float(bundle.numeric_ranges["Driver_Experience_Years"][1]),
                value=float(bundle.numeric_defaults["Driver_Experience_Years"]),
                step=1.0,
            )
            bus_age = st.slider(
                "Bus age (years)",
                min_value=float(bundle.numeric_ranges["Bus_Age_Years"][0]),
                max_value=float(bundle.numeric_ranges["Bus_Age_Years"][1]),
                value=float(bundle.numeric_defaults["Bus_Age_Years"]),
                step=1.0,
            )
            passenger_load = st.slider(
                "Passenger load",
                min_value=float(bundle.numeric_ranges["Passenger_Load"][0]),
                max_value=float(bundle.numeric_ranges["Passenger_Load"][1]),
                value=float(bundle.numeric_defaults["Passenger_Load"]),
                step=1.0,
            )
            peak_hour = st.toggle("Peak hour departure", value=bool(round(bundle.numeric_defaults["Peak_Hour"])))
            special_event = st.toggle("Special event nearby", value=False)

            submit = st.form_submit_button("Predict delay")

    with right:
        st.subheader("Context Snapshot")
        context = contextual_history(df, selected_weather, selected_day)
        render_metric_cards(
            [
                ("Weather trips", f"{int(context['trips']):,}", f"{selected_weather} on {selected_day}"),
                (
                    "Historical avg delay",
                    "N/A" if math.isnan(context["avg_delay"]) else f"{context['avg_delay']:.1f} min",
                    "Observed mean for the selected context",
                ),
                (
                    "Historical late rate",
                    human_pct(context["late_rate"]),
                    "Delay above 15 minutes",
                ),
                ("Suggested weekend flag", str(int(selected_day in {'Sat', 'Sun'})), "Derived automatically from day"),
            ]
        )

    if submit:
        weekend = 1.0 if selected_day in {"Sat", "Sun"} else 0.0
        inputs = {
            "Distance_KM": distance,
            "Avg_Speed": avg_speed,
            "Fuel_Stops": fuel_stops,
            "Traffic_Score": traffic_score,
            "Weather_Impact": weather_impact,
            "Stops": stops,
            "Maintenance_Score": maintenance,
            "Peak_Hour": float(peak_hour),
            "Driver_Experience_Years": driver_exp,
            "Special_Event": float(special_event),
            "Bus_Age_Years": bus_age,
            "Passenger_Load": passenger_load,
            "Departure_Hour": float(departure_hour),
            "Weekend": weekend,
        }
        result = predict_delay(bundle, inputs)
        context = contextual_history(df, selected_weather, selected_day)

        render_metric_cards(
            [
                ("Predicted delay", f"{result['predicted_delay_min']:.1f} min", "Primary model output"),
                ("95% interval", f"{result['95_PI'][0]:.1f} to {result['95_PI'][1]:.1f}", "Residual-based uncertainty band"),
                (">15 min late", human_pct(result["p_more_than_15min_late"]), "Scenario probability"),
                ("On-time chance", human_pct(result["p_on_time"]), "Delay of 5 minutes or less"),
            ]
        )

        st.markdown("### Scenario vs History")
        render_metric_cards(
            [
                ("Context trips", f"{int(context['trips']):,}", f"{selected_weather} on {selected_day}"),
                (
                    "Historical avg delay",
                    "N/A" if math.isnan(context["avg_delay"]) else f"{context['avg_delay']:.1f} min",
                    "Observed mean under same context",
                ),
                (
                    "Historical late rate",
                    human_pct(context["late_rate"]),
                    "Trips with delay above 15 minutes",
                ),
                ("Weekend flag", str(int(weekend)), "Derived from selected day"),
            ]
        )

    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    inject_styles()
    bundle = get_bundle()
    filtered_df = sidebar_filters(bundle)

    st.markdown(
        """
        <div class="hero">
            <h1>Pakistan Intercity Bus Delay Predictor</h1>
            <p>Streamlit deployment of the semester project notebook: cleaned data, probability analysis, model comparison, and live delay prediction in one app.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered_df.empty:
        st.warning("The current filter combination returned no rows. Reset one or more sidebar filters.")
        return

    tabs = st.tabs(["Overview", "Exploration", "Probability", "Model", "Predictor"])
    with tabs[0]:
        overview_tab(bundle, filtered_df)
    with tabs[1]:
        exploration_tab(bundle, filtered_df)
    with tabs[2]:
        probability_tab(bundle)
    with tabs[3]:
        model_tab(bundle)
    with tabs[4]:
        predictor_tab(bundle)


if __name__ == "__main__":
    main()
