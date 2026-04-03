from __future__ import annotations

import warnings

import pandas as pd

from agri_core import (
    benchmark_profile,
    compare_scenarios,
    find_similar_records,
    generate_management_tips,
    load_dataset,
    rank_crops_for_conditions,
    train_models,
)

warnings.filterwarnings("ignore")


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def format_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    for column in ["r2", "mae", "rmse", "cv_r2_mean", "cv_r2_std"]:
        formatted[column] = formatted[column].map(lambda value: f"{value:.4f}")
    return formatted


def main() -> None:
    dataframe = load_dataset()
    bundle = train_models(dataframe)

    print_section("Crop Yield Intelligence Suite")
    print(f"Rows               : {len(dataframe)}")
    print(f"Columns            : {len(dataframe.columns)}")
    print(f"Crops covered      : {dataframe['crop'].nunique()}")
    print(f"States covered     : {dataframe['state'].nunique()}")
    print(f"Average yield      : {dataframe['yield_kg_per_ha'].mean():.1f} kg/ha")
    print(f"Best model         : {bundle['best_model_name']}")

    print_section("Model Leaderboard")
    print(format_metrics(bundle["leaderboard"]).to_string(index=False))

    profile = {
        "crop": "Rice",
        "state": "Punjab",
        "season": "Kharif",
        "soil_type": "Alluvial",
        "rainfall_mm": 1200.0,
        "temperature_c": 28.0,
        "humidity_pct": 72.0,
        "fertilizer_kg_ha": 180.0,
        "area_hectares": 12.0,
    }

    benchmark = benchmark_profile(dataframe, profile)
    similar_records = find_similar_records(dataframe, profile)
    tips = generate_management_tips(dataframe, profile)
    predicted_yield = bundle["best_model"].predict(pd.DataFrame([profile]))[0]

    print_section("Field Forecast")
    print(f"Profile crop       : {profile['crop']}")
    print(f"Predicted yield    : {predicted_yield:.1f} kg/ha")
    print(f"Projected output   : {predicted_yield * profile['area_hectares']:.1f} kg")
    print(
        "Benchmark context  : "
        f"{benchmark['benchmark_level']} "
        f"(n={benchmark['sample_size']})"
    )
    print(f"Benchmark avg yield: {benchmark['avg_yield']:.1f} kg/ha")

    print_section("Recommended Actions")
    for index, tip in enumerate(tips, start=1):
        print(f"{index}. {tip}")

    print_section("Closest Historical Records")
    print(similar_records.round(2).to_string(index=False))

    recommendation_context = {
        "state": "Punjab",
        "season": "Kharif",
        "soil_type": "Alluvial",
        "rainfall_mm": 1200.0,
        "temperature_c": 28.0,
        "humidity_pct": 72.0,
        "fertilizer_kg_ha": 180.0,
        "area_hectares": 12.0,
    }
    crop_rankings = rank_crops_for_conditions(
        bundle["best_model"], dataframe, recommendation_context, top_n=5
    )

    print_section("Crop Planning Recommendations")
    print(crop_rankings.round(2).to_string(index=False))

    improved_profile = profile.copy()
    improved_profile["rainfall_mm"] = 1350.0
    improved_profile["fertilizer_kg_ha"] = 200.0
    improved_profile["humidity_pct"] = 76.0

    scenario = compare_scenarios(bundle["best_model"], profile, improved_profile)

    print_section("Scenario Comparison")
    print(f"Baseline yield     : {scenario['baseline_yield']:.1f} kg/ha")
    print(f"Improved yield     : {scenario['candidate_yield']:.1f} kg/ha")
    print(f"Yield change       : {scenario['delta_yield']:.1f} kg/ha")
    print(f"Production change  : {scenario['delta_production']:.1f} kg")


if __name__ == "__main__":
    main()
