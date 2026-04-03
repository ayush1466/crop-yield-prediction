from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from agri_core import (
    NUMERICAL_FEATURES,
    benchmark_profile,
    compare_scenarios,
    find_similar_records,
    generate_management_tips,
    list_options,
    load_dataset,
    predict_yield,
    rank_crops_for_conditions,
    train_models,
)


CHART_BG = "#ffffff"
TEXT_COLOR = "#17212b"
GRID_COLOR = "#d7dee8"
PRIMARY_COLOR = "#1f6aa5"
ACCENT_GREEN = "#3d8b63"
ACCENT_ORANGE = "#d97706"
ACCENT_RED = "#c2410c"
ACCENT_GOLD = "#b58900"
ACCENT_TEAL = "#0f766e"


st.set_page_config(
    page_title="Agri Yield Intelligence Suite",
    page_icon="AG",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return load_dataset()


@st.cache_resource(show_spinner=False)
def get_model_bundle(dataframe: pd.DataFrame) -> dict:
    return train_models(dataframe)


def build_profile(prefix: str, options: dict, dataframe: pd.DataFrame, include_crop: bool) -> dict:
    profile = {}
    if include_crop:
        profile["crop"] = st.selectbox(
            "Crop",
            options["crop"],
            index=options["crop"].index("Rice") if "Rice" in options["crop"] else 0,
            key=f"{prefix}_crop",
        )
    profile["state"] = st.selectbox("State", options["state"], key=f"{prefix}_state")
    profile["season"] = st.selectbox("Season", options["season"], key=f"{prefix}_season")
    profile["soil_type"] = st.selectbox("Soil type", options["soil_type"], key=f"{prefix}_soil")
    profile["rainfall_mm"] = st.slider(
        "Rainfall (mm)",
        min_value=int(dataframe["rainfall_mm"].min()),
        max_value=int(dataframe["rainfall_mm"].max()),
        value=int(dataframe["rainfall_mm"].median()),
        step=25,
        key=f"{prefix}_rainfall",
    )
    profile["temperature_c"] = st.slider(
        "Temperature (C)",
        min_value=int(dataframe["temperature_c"].min()),
        max_value=int(dataframe["temperature_c"].max()),
        value=int(round(dataframe["temperature_c"].median())),
        key=f"{prefix}_temperature",
    )
    profile["humidity_pct"] = st.slider(
        "Humidity (%)",
        min_value=int(dataframe["humidity_pct"].min()),
        max_value=int(dataframe["humidity_pct"].max()),
        value=int(round(dataframe["humidity_pct"].median())),
        key=f"{prefix}_humidity",
    )
    profile["fertilizer_kg_ha"] = st.slider(
        "Fertilizer (kg/ha)",
        min_value=int(dataframe["fertilizer_kg_ha"].min()),
        max_value=int(dataframe["fertilizer_kg_ha"].max()),
        value=int(round(dataframe["fertilizer_kg_ha"].median())),
        step=5,
        key=f"{prefix}_fertilizer",
    )
    profile["area_hectares"] = st.number_input(
        "Area (hectares)",
        min_value=0.5,
        max_value=float(dataframe["area_hectares"].max()),
        value=float(round(dataframe["area_hectares"].median(), 2)),
        step=0.5,
        key=f"{prefix}_area",
    )
    return profile


def render_metric_card(title: str, value: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_axis(ax) -> None:
    ax.set_facecolor(CHART_BG)
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#b8c4d1")
    ax.spines["bottom"].set_color("#b8c4d1")


dataframe = get_dataset()
bundle = get_model_bundle(dataframe)
options = list_options(dataframe)
leaderboard = bundle["leaderboard"]
best_model_name = bundle["best_model_name"]
best_model = bundle["best_model"]
best_rmse = bundle["best_rmse"]

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(123, 176, 232, 0.18), transparent 28%),
                linear-gradient(180deg, #f8fbff 0%, #eef4fb 100%);
            color: #17212b;
            font-family: "Trebuchet MS", "Gill Sans", sans-serif;
        }
        .stApp [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #eaf2fb 0%, #dfeaf7 100%);
            border-right: 1px solid #c4d5e8;
        }
        .hero {
            padding: 1.75rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #174c7c, #2f7eb6);
            color: #f8fbff;
            box-shadow: 0 16px 40px rgba(34, 80, 120, 0.18);
            margin-bottom: 1rem;
        }
        .hero h1 {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 2.3rem;
            margin-bottom: 0.35rem;
            color: #ffffff;
        }
        .hero p {
            font-size: 1rem;
            line-height: 1.55;
            margin-bottom: 0;
            color: #eef6fd;
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid #c9d8e6;
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 24px rgba(50, 84, 118, 0.08);
            min-height: 126px;
        }
        .metric-title {
            font-size: 0.88rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #47627d;
            margin-bottom: 0.45rem;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #17324d;
            line-height: 1.1;
            margin-bottom: 0.35rem;
        }
        .metric-subtitle {
            color: #4f6478;
            font-size: 0.9rem;
        }
        .section-note {
            background: #fff8ea;
            border-left: 4px solid #d97706;
            padding: 0.9rem 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: #e6eef8;
            border-radius: 999px;
            color: #21415f;
            padding: 0.55rem 1rem;
            border: 1px solid #c8d8ea;
        }
        .stTabs [aria-selected="true"] {
            background: #1f6aa5 !important;
            color: #ffffff !important;
            border-color: #1f6aa5 !important;
        }
        .stSelectbox label,
        .stMultiSelect label,
        .stNumberInput label,
        .stSlider label {
            color: #17212b !important;
            font-weight: 600;
        }
        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div {
            background: #ffffff !important;
            color: #17212b !important;
            border: 1px solid #bfd0e2 !important;
            box-shadow: none !important;
        }
        .stSelectbox [data-baseweb="select"] input,
        .stMultiSelect [data-baseweb="select"] input {
            color: #17212b !important;
            -webkit-text-fill-color: #17212b !important;
        }
        .stSelectbox [data-baseweb="select"] span,
        .stMultiSelect [data-baseweb="select"] span,
        .stSelectbox [data-baseweb="select"] svg,
        .stMultiSelect [data-baseweb="select"] svg {
            color: #17212b !important;
            fill: #17212b !important;
        }
        div[data-baseweb="popover"] ul {
            background: #ffffff !important;
            border: 1px solid #bfd0e2 !important;
        }
        div[data-baseweb="popover"] li {
            background: #ffffff !important;
            color: #17212b !important;
        }
        div[data-baseweb="popover"] li:hover {
            background: #e8f1fb !important;
        }
        .stMultiSelect [data-baseweb="tag"] {
            background: #dbeafe !important;
            border-radius: 999px !important;
        }
        .stMultiSelect [data-baseweb="tag"] span,
        .stMultiSelect [data-baseweb="tag"] svg {
            color: #17324d !important;
            fill: #17324d !important;
        }
        .stNumberInput input {
            background: #ffffff !important;
            color: #17212b !important;
            border: 1px solid #bfd0e2 !important;
        }
        .stDataFrame, div[data-testid="stMetric"] {
            background: #ffffff;
            border-radius: 12px;
        }
        h1, h2, h3, h4, h5, h6, label, p, li, span, div {
            color: #17212b;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Agri Yield Intelligence Suite</h1>
        <p>
            A project-scale platform for crop yield forecasting, crop planning, scenario analysis,
            model comparison, and historical data exploration. This version uses the real dataset in the repository
            and spreads the work across multiple product areas instead of a single predictor form.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Workspace Summary")
    st.write(f"Rows loaded: {len(dataframe):,}")
    st.write(f"Best model: {best_model_name}")
    st.write(f"Average yield: {dataframe['yield_kg_per_ha'].mean():.1f} kg/ha")
    st.write(f"Tracked crops: {dataframe['crop'].nunique()}")
    selected_states = st.multiselect(
        "Overview filter: states",
        options["state"],
        default=options["state"],
    )
    selected_seasons = st.multiselect(
        "Overview filter: seasons",
        options["season"],
        default=options["season"],
    )

overview_df = dataframe[
    dataframe["state"].isin(selected_states) & dataframe["season"].isin(selected_seasons)
].copy()
if overview_df.empty:
    overview_df = dataframe.copy()

tabs = st.tabs(
    [
        "Executive Dashboard",
        "Prediction Studio",
        "Crop Planner",
        "Scenario Lab",
        "Data Explorer",
        "Model Lab",
    ]
)

with tabs[0]:
    top_crop = overview_df.groupby("crop")["yield_kg_per_ha"].mean().sort_values(ascending=False).index[0]
    top_state = overview_df.groupby("state")["yield_kg_per_ha"].mean().sort_values(ascending=False).index[0]
    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Yield average", f"{overview_df['yield_kg_per_ha'].mean():.0f} kg/ha")
    with metric_cols[1]:
        render_metric_card("Top crop", top_crop, "Highest mean yield in the current view")
    with metric_cols[2]:
        render_metric_card("Top state", top_state, "Best average output in the current view")
    with metric_cols[3]:
        render_metric_card("Best model R2", f"{leaderboard.iloc[0]['r2']:.3f}", best_model_name)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.subheader("Average Yield by Crop")
        fig, ax = plt.subplots(figsize=(7, 4))
        (
            overview_df.groupby("crop")["yield_kg_per_ha"]
            .mean()
            .sort_values()
            .plot(kind="barh", color=PRIMARY_COLOR, ax=ax)
        )
        ax.set_xlabel("Yield (kg/ha)")
        ax.set_ylabel("")
        style_axis(ax)
        fig.tight_layout()
        st.pyplot(fig)

    with chart_cols[1]:
        st.subheader("Average Yield by State")
        fig, ax = plt.subplots(figsize=(7, 4))
        (
            overview_df.groupby("state")["yield_kg_per_ha"]
            .mean()
            .sort_values(ascending=False)
            .plot(kind="bar", color=ACCENT_ORANGE, ax=ax)
        )
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Yield (kg/ha)")
        style_axis(ax)
        fig.tight_layout()
        st.pyplot(fig)

    lower_cols = st.columns([1.2, 1])
    with lower_cols[0]:
        st.subheader("Season and Soil Performance")
        pivot_table = overview_df.pivot_table(
            values="yield_kg_per_ha",
            index="season",
            columns="soil_type",
            aggfunc="mean",
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(
            pivot_table,
            annot=True,
            cmap="Blues",
            fmt=".0f",
            linewidths=0.6,
            linecolor="#d9e5f2",
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(colors=TEXT_COLOR)
        fig.tight_layout()
        st.pyplot(fig)

    with lower_cols[1]:
        st.subheader("Top Crop-State Combinations")
        combo_table = (
            overview_df.groupby(["crop", "state"])["yield_kg_per_ha"]
            .mean()
            .reset_index()
            .sort_values("yield_kg_per_ha", ascending=False)
            .head(10)
            .rename(columns={"yield_kg_per_ha": "avg_yield_kg_per_ha"})
        )
        st.dataframe(combo_table, use_container_width=True, hide_index=True)

with tabs[1]:
    st.markdown(
        '<div class="section-note">Forecast a specific field, benchmark it against similar historical records, and export a report.</div>',
        unsafe_allow_html=True,
    )
    form_col, output_col = st.columns([1, 1.15], gap="large")

    with form_col:
        selected_model_name = st.selectbox(
            "Prediction model",
            leaderboard["model"].tolist(),
            index=leaderboard["model"].tolist().index(best_model_name),
        )
        prediction_profile = build_profile("predict", options, dataframe, include_crop=True)
        run_prediction = st.button("Run forecast", use_container_width=True)

    with output_col:
        if run_prediction:
            selected_model = bundle["production_models"][selected_model_name]
            predicted_yield = predict_yield(selected_model, prediction_profile)
            projected_production = predicted_yield * prediction_profile["area_hectares"]
            benchmark = benchmark_profile(dataframe, prediction_profile)
            yield_delta = predicted_yield - benchmark["avg_yield"]
            confidence_low = predicted_yield - best_rmse
            confidence_high = predicted_yield + best_rmse
            tips = generate_management_tips(dataframe, prediction_profile)
            similar_records = find_similar_records(dataframe, prediction_profile)

            summary_cols = st.columns(3)
            summary_cols[0].metric("Predicted yield", f"{predicted_yield:,.0f} kg/ha")
            summary_cols[1].metric("Projected production", f"{projected_production:,.0f} kg")
            summary_cols[2].metric("Vs benchmark", f"{yield_delta:+,.0f} kg/ha", benchmark["benchmark_level"])

            st.info(
                f"Expected operating range from the best-model RMSE: {confidence_low:,.0f} to {confidence_high:,.0f} kg/ha."
            )

            st.subheader("Management Notes")
            for tip in tips:
                st.write(f"- {tip}")

            st.subheader("Closest Historical Records")
            st.dataframe(similar_records.round(2), use_container_width=True, hide_index=True)

            report_frame = pd.DataFrame(
                [
                    {
                        "model": selected_model_name,
                        **prediction_profile,
                        "predicted_yield_kg_per_ha": round(predicted_yield, 2),
                        "projected_production_kg": round(projected_production, 2),
                        "benchmark_avg_yield_kg_per_ha": round(benchmark["avg_yield"], 2),
                    }
                ]
            )
            st.download_button(
                "Download prediction report",
                data=report_frame.to_csv(index=False).encode("utf-8"),
                file_name="prediction_report.csv",
                mime="text/csv",
            )
        else:
            st.write("Run a forecast to see benchmark data, similar records, and export options.")

with tabs[2]:
    st.markdown(
        '<div class="section-note">Rank crops for one set of field conditions so the project supports planning decisions, not only predictions.</div>',
        unsafe_allow_html=True,
    )
    planner_left, planner_right = st.columns([1, 1.1], gap="large")
    with planner_left:
        planner_context = build_profile("planner", options, dataframe, include_crop=False)
        run_planner = st.button("Generate crop plan", use_container_width=True)
    with planner_right:
        if run_planner:
            crop_plan = rank_crops_for_conditions(best_model, dataframe, planner_context, top_n=7)
            st.subheader("Recommended crops")
            st.dataframe(crop_plan.round(2), use_container_width=True, hide_index=True)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(
                crop_plan["crop"][::-1],
                crop_plan["predicted_yield_kg_per_ha"][::-1],
                color=ACCENT_GREEN,
            )
            ax.set_xlabel("Predicted yield (kg/ha)")
            style_axis(ax)
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.write("Generate a crop plan to rank crops for the selected field conditions.")

with tabs[3]:
    st.markdown(
        '<div class="section-note">Compare a baseline plan against an adjusted one to quantify the effect of changing field assumptions.</div>',
        unsafe_allow_html=True,
    )
    baseline_col, candidate_col = st.columns(2, gap="large")
    with baseline_col:
        st.subheader("Baseline")
        baseline_profile = build_profile("baseline", options, dataframe, include_crop=True)
    with candidate_col:
        st.subheader("Candidate")
        candidate_profile = build_profile("candidate", options, dataframe, include_crop=True)

    if st.button("Compare scenarios", use_container_width=True):
        scenario_result = compare_scenarios(best_model, baseline_profile, candidate_profile)
        summary_cols = st.columns(3)
        summary_cols[0].metric("Yield change", f"{scenario_result['delta_yield']:+,.0f} kg/ha")
        summary_cols[1].metric("Production change", f"{scenario_result['delta_production']:+,.0f} kg")
        summary_cols[2].metric("Candidate yield", f"{scenario_result['candidate_yield']:,.0f} kg/ha")

        compare_frame = pd.DataFrame(
            {
                "Scenario": ["Baseline", "Candidate"],
                "Yield (kg/ha)": [
                    scenario_result["baseline_yield"],
                    scenario_result["candidate_yield"],
                ],
                "Production (kg)": [
                    scenario_result["baseline_production"],
                    scenario_result["candidate_production"],
                ],
            }
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        compare_frame.set_index("Scenario")["Yield (kg/ha)"].plot(
            kind="bar", ax=ax, color=[ACCENT_ORANGE, ACCENT_GREEN]
        )
        ax.set_ylabel("Yield (kg/ha)")
        style_axis(ax)
        fig.tight_layout()
        st.pyplot(fig)

with tabs[4]:
    st.subheader("Interactive Data Explorer")
    filter_cols = st.columns(4)
    explorer_crops = filter_cols[0].multiselect("Crop", options["crop"], default=options["crop"])
    explorer_states = filter_cols[1].multiselect("State", options["state"], default=options["state"])
    explorer_seasons = filter_cols[2].multiselect("Season", options["season"], default=options["season"])
    explorer_soils = filter_cols[3].multiselect("Soil", options["soil_type"], default=options["soil_type"])

    explorer_df = dataframe[
        dataframe["crop"].isin(explorer_crops)
        & dataframe["state"].isin(explorer_states)
        & dataframe["season"].isin(explorer_seasons)
        & dataframe["soil_type"].isin(explorer_soils)
    ].copy()
    if explorer_df.empty:
        explorer_df = dataframe.copy()

    explorer_metrics = st.columns(4)
    explorer_metrics[0].metric("Rows", f"{len(explorer_df):,}")
    explorer_metrics[1].metric("Avg yield", f"{explorer_df['yield_kg_per_ha'].mean():.0f} kg/ha")
    explorer_metrics[2].metric("Avg rainfall", f"{explorer_df['rainfall_mm'].mean():.0f} mm")
    explorer_metrics[3].metric("Avg fertilizer", f"{explorer_df['fertilizer_kg_ha'].mean():.0f} kg/ha")

    plot_cols = st.columns(2)
    with plot_cols[0]:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(
            data=explorer_df,
            x="rainfall_mm",
            y="yield_kg_per_ha",
            hue="crop",
            alpha=0.65,
            palette="tab10",
            ax=ax,
        )
        style_axis(ax)
        fig.tight_layout()
        st.pyplot(fig)

    with plot_cols[1]:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(
            data=explorer_df,
            x="season",
            y="yield_kg_per_ha",
            palette=["#1f6aa5", "#3d8b63", "#d97706"],
            ax=ax,
        )
        style_axis(ax)
        fig.tight_layout()
        st.pyplot(fig)

    st.subheader("Correlation Map")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        explorer_df[NUMERICAL_FEATURES + ["yield_kg_per_ha"]].corr(),
        annot=True,
        cmap="RdYlBu_r",
        linewidths=0.5,
        linecolor="#d9e5f2",
        ax=ax,
    )
    ax.tick_params(colors=TEXT_COLOR)
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Filtered records")
    st.dataframe(explorer_df.round(2), use_container_width=True, hide_index=True)

with tabs[5]:
    st.subheader("Model leaderboard")
    metric_view = leaderboard.copy()
    for column in ["r2", "mae", "rmse", "cv_r2_mean", "cv_r2_std"]:
        metric_view[column] = metric_view[column].map(lambda value: round(float(value), 4))
    st.dataframe(metric_view, use_container_width=True, hide_index=True)

    model_cols = st.columns(2)
    with model_cols[0]:
        st.subheader("Permutation importance")
        fig, ax = plt.subplots(figsize=(7, 4))
        importance_frame = bundle["importance"].sort_values("importance_mean", ascending=True)
        ax.barh(importance_frame["feature"], importance_frame["importance_mean"], color=ACCENT_TEAL)
        ax.set_xlabel("Mean importance drop")
        style_axis(ax)
        fig.tight_layout()
        st.pyplot(fig)

    with model_cols[1]:
        st.subheader("Residual spread")
        fig, ax = plt.subplots(figsize=(7, 4))
        residuals = bundle["residuals"]
        ax.scatter(residuals["actual"], residuals["residual"], alpha=0.55, color=ACCENT_RED)
        ax.axhline(0, color=PRIMARY_COLOR, linestyle="--", linewidth=1)
        ax.set_xlabel("Actual yield (kg/ha)")
        ax.set_ylabel("Residual")
        style_axis(ax)
        fig.tight_layout()
        st.pyplot(fig)

    st.caption("The best model is refit on the full dataset for production predictions after holdout evaluation.")
