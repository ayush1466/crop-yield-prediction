from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATASET_PATH = Path(__file__).resolve().parent / "crop_yield_dataset.csv"
CATEGORICAL_FEATURES = ["crop", "state", "season", "soil_type"]
NUMERICAL_FEATURES = [
    "rainfall_mm",
    "temperature_c",
    "humidity_pct",
    "fertilizer_kg_ha",
    "area_hectares",
]
TARGET_COLUMN = "yield_kg_per_ha"
MODEL_RANDOM_STATE = 42


def load_dataset(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    dataframe.columns = dataframe.columns.str.strip().str.lower()
    return dataframe


def get_feature_frame(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    return dataframe[feature_columns].copy(), dataframe[TARGET_COLUMN].copy()


def list_options(dataframe: pd.DataFrame) -> Dict[str, List[str]]:
    return {
        column: sorted(dataframe[column].dropna().astype(str).unique().tolist())
        for column in CATEGORICAL_FEATURES
    }


def _make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", numeric_transformer, NUMERICAL_FEATURES),
        ]
    )


def build_model_registry() -> Dict[str, Pipeline]:
    return {
        "Linear Regression": Pipeline(
            steps=[
                ("preprocessor", _make_preprocessor(scale_numeric=True)),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", _make_preprocessor(scale_numeric=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        random_state=MODEL_RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocessor", _make_preprocessor(scale_numeric=False)),
                ("model", GradientBoostingRegressor(random_state=MODEL_RANDOM_STATE)),
            ]
        ),
    }


def train_models(dataframe: pd.DataFrame) -> dict:
    features, target = get_feature_frame(dataframe)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=MODEL_RANDOM_STATE
    )

    evaluation_models = {}
    production_models = {}
    rows = []

    for name, pipeline in build_model_registry().items():
        fitted_pipeline = pipeline.fit(X_train, y_train)
        predictions = fitted_pipeline.predict(X_test)
        cv_scores = cross_val_score(
            pipeline, features, target, cv=5, scoring="r2", n_jobs=1
        )

        rows.append(
            {
                "model": name,
                "r2": r2_score(y_test, predictions),
                "mae": mean_absolute_error(y_test, predictions),
                "rmse": mean_squared_error(y_test, predictions, squared=False),
                "cv_r2_mean": cv_scores.mean(),
                "cv_r2_std": cv_scores.std(),
            }
        )
        evaluation_models[name] = fitted_pipeline
        production_models[name] = pipeline.fit(features, target)

    leaderboard = pd.DataFrame(rows).sort_values(
        by=["r2", "cv_r2_mean"], ascending=False
    )
    best_model_name = leaderboard.iloc[0]["model"]
    best_eval_model = evaluation_models[best_model_name]
    best_production_model = production_models[best_model_name]
    best_predictions = best_eval_model.predict(X_test)

    importance = permutation_importance(
        best_eval_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=MODEL_RANDOM_STATE,
        scoring="r2",
    )
    importance_frame = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    residual_frame = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": best_predictions,
            "residual": y_test.to_numpy() - best_predictions,
        }
    )

    return {
        "leaderboard": leaderboard.reset_index(drop=True),
        "evaluation_models": evaluation_models,
        "production_models": production_models,
        "best_model_name": best_model_name,
        "best_model": best_production_model,
        "best_rmse": float(
            leaderboard.loc[leaderboard["model"] == best_model_name, "rmse"].iloc[0]
        ),
        "importance": importance_frame,
        "residuals": residual_frame,
    }


def make_profile_frame(profile: dict) -> pd.DataFrame:
    ordered = {
        column: [profile[column]]
        for column in CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    }
    return pd.DataFrame(ordered)


def predict_yield(model: Pipeline, profile: dict) -> float:
    prediction = model.predict(make_profile_frame(profile))[0]
    return float(prediction)


def benchmark_profile(dataframe: pd.DataFrame, profile: dict) -> dict:
    candidates = [
        (
            "Exact crop-state-season match",
            dataframe[
                (dataframe["crop"] == profile["crop"])
                & (dataframe["state"] == profile["state"])
                & (dataframe["season"] == profile["season"])
            ],
        ),
        (
            "Crop and state match",
            dataframe[
                (dataframe["crop"] == profile["crop"])
                & (dataframe["state"] == profile["state"])
            ],
        ),
        (
            "Crop and season match",
            dataframe[
                (dataframe["crop"] == profile["crop"])
                & (dataframe["season"] == profile["season"])
            ],
        ),
        ("Season match", dataframe[dataframe["season"] == profile["season"]]),
    ]

    for label, subset in candidates:
        if len(subset) >= 8:
            return {
                "benchmark_level": label,
                "sample_size": int(len(subset)),
                "avg_yield": float(subset[TARGET_COLUMN].mean()),
                "p75_yield": float(subset[TARGET_COLUMN].quantile(0.75)),
                "avg_area": float(subset["area_hectares"].mean()),
            }

    return {
        "benchmark_level": "Overall dataset",
        "sample_size": int(len(dataframe)),
        "avg_yield": float(dataframe[TARGET_COLUMN].mean()),
        "p75_yield": float(dataframe[TARGET_COLUMN].quantile(0.75)),
        "avg_area": float(dataframe["area_hectares"].mean()),
    }


def find_similar_records(
    dataframe: pd.DataFrame, profile: dict, limit: int = 5
) -> pd.DataFrame:
    subset = dataframe[
        (dataframe["state"] == profile["state"])
        & (dataframe["season"] == profile["season"])
    ].copy()
    if len(subset) < limit:
        subset = dataframe.copy()

    numeric = subset[NUMERICAL_FEATURES].copy()
    for column in NUMERICAL_FEATURES:
        std_value = max(float(dataframe[column].std()), 1.0)
        numeric[column] = (numeric[column] - profile[column]) / std_value

    subset["distance_score"] = np.sqrt((numeric**2).sum(axis=1))
    subset = subset.sort_values("distance_score").head(limit)
    columns = [
        "crop",
        "state",
        "season",
        "soil_type",
        "rainfall_mm",
        "temperature_c",
        "humidity_pct",
        "fertilizer_kg_ha",
        "yield_kg_per_ha",
        "distance_score",
    ]
    return subset[columns].reset_index(drop=True)


def rank_crops_for_conditions(
    model: Pipeline, dataframe: pd.DataFrame, context: dict, top_n: int = 5
) -> pd.DataFrame:
    rows = []
    for crop_name in sorted(dataframe["crop"].unique()):
        profile = {
            "crop": crop_name,
            "state": context["state"],
            "season": context["season"],
            "soil_type": context["soil_type"],
            "rainfall_mm": context["rainfall_mm"],
            "temperature_c": context["temperature_c"],
            "humidity_pct": context["humidity_pct"],
            "fertilizer_kg_ha": context["fertilizer_kg_ha"],
            "area_hectares": context["area_hectares"],
        }
        predicted = predict_yield(model, profile)
        rows.append(
            {
                "crop": crop_name,
                "predicted_yield_kg_per_ha": predicted,
                "projected_production_kg": predicted * context["area_hectares"],
                "yield_per_kg_fertilizer": predicted / max(context["fertilizer_kg_ha"], 1.0),
            }
        )

    recommendations = pd.DataFrame(rows).sort_values(
        "predicted_yield_kg_per_ha", ascending=False
    )
    best_value = recommendations["predicted_yield_kg_per_ha"].max()
    recommendations["fit_score"] = (
        100 * recommendations["predicted_yield_kg_per_ha"] / max(best_value, 1.0)
    ).round(1)
    return recommendations.head(top_n).reset_index(drop=True)


def compare_scenarios(model: Pipeline, baseline: dict, candidate: dict) -> dict:
    baseline_yield = predict_yield(model, baseline)
    candidate_yield = predict_yield(model, candidate)
    baseline_production = baseline_yield * baseline["area_hectares"]
    candidate_production = candidate_yield * candidate["area_hectares"]
    return {
        "baseline_yield": baseline_yield,
        "candidate_yield": candidate_yield,
        "delta_yield": candidate_yield - baseline_yield,
        "baseline_production": baseline_production,
        "candidate_production": candidate_production,
        "delta_production": candidate_production - baseline_production,
    }


def generate_management_tips(dataframe: pd.DataFrame, profile: dict) -> List[str]:
    crop_slice = dataframe[
        (dataframe["crop"] == profile["crop"])
        & (dataframe["season"] == profile["season"])
    ].copy()
    if len(crop_slice) < 20:
        crop_slice = dataframe[dataframe["crop"] == profile["crop"]].copy()
    if len(crop_slice) < 20:
        crop_slice = dataframe.copy()

    high_yield_slice = crop_slice[
        crop_slice[TARGET_COLUMN] >= crop_slice[TARGET_COLUMN].quantile(0.75)
    ]
    if high_yield_slice.empty:
        high_yield_slice = crop_slice

    tips = []
    target_rainfall = float(high_yield_slice["rainfall_mm"].median())
    if profile["rainfall_mm"] < target_rainfall * 0.85:
        tips.append(
            "Rainfall is below the high-yield median for similar records, so irrigation or moisture conservation could help."
        )
    elif profile["rainfall_mm"] > target_rainfall * 1.15:
        tips.append(
            "Rainfall is above the typical high-yield range, so drainage and disease monitoring should be part of the plan."
        )

    target_temp = float(high_yield_slice["temperature_c"].median())
    if profile["temperature_c"] > target_temp + 3:
        tips.append(
            "Temperature is higher than the strong-performing benchmark; plan for heat-stress mitigation."
        )

    target_fertilizer = float(high_yield_slice["fertilizer_kg_ha"].median())
    if profile["fertilizer_kg_ha"] < target_fertilizer * 0.8:
        tips.append(
            "Fertilizer application is lower than the top-quartile median, which may be limiting yield potential."
        )
    elif profile["fertilizer_kg_ha"] > target_fertilizer * 1.3:
        tips.append(
            "Fertilizer input is relatively high; review nutrient efficiency so extra cost is justified by output."
        )

    target_humidity = float(high_yield_slice["humidity_pct"].median())
    if profile["humidity_pct"] < target_humidity * 0.85:
        tips.append(
            "Humidity is lower than the high-yield benchmark, which can increase moisture stress during sensitive stages."
        )

    if not tips:
        tips.append(
            "This profile is close to stronger historical conditions, so focus on execution quality and timely monitoring."
        )

    return tips
