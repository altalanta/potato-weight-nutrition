"""Feature engineering: weekly aggregates, keyword categories, baselines."""

import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np


def build_features(
    input_dir: Path,
    output_dir: Path,
    config: Optional[Path] = None
) -> Dict[str, Any]:
    """Build features from cleaned data files.
    
    Creates:
    - weight_trajectories.csv: Per-subject weekly weights with deltas
    - nutrition_weekly.csv: Weekly nutrition aggregates  
    - food_keywords.csv: Weekly food category flags
    - analysis_df.csv: Merged modeling dataset
    
    Args:
        input_dir: Directory containing cleaned data files
        output_dir: Directory to save feature files
        config: Optional configuration file path
        
    Returns:
        Dictionary with feature engineering statistics
    """
    logger = logging.getLogger("pwn.features")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data files
    tidy_path = input_dir / "Potato_tidy.csv"
    nutrition_path = input_dir / "Potato_nutrition_rows.csv"
    fiber_path = input_dir / "Potato_fiber_daily.csv"
    
    missing_files = []
    if not tidy_path.exists():
        missing_files.append("Potato_tidy.csv")
    if not nutrition_path.exists():
        missing_files.append("Potato_nutrition_rows.csv")
    if not fiber_path.exists():
        missing_files.append("Potato_fiber_daily.csv")
    
    if missing_files:
        logger.warning(f"Missing data files: {missing_files}")
        return {"status": "failed", "missing_files": missing_files}
    
    # Load dataframes
    tidy_df = pd.read_csv(tidy_path)
    nutrition_df = pd.read_csv(nutrition_path)
    fiber_df = pd.read_csv(fiber_path)
    
    logger.info(f"Loaded data: {len(tidy_df)} tidy rows, {len(nutrition_df)} nutrition rows, {len(fiber_df)} fiber rows")
    
    # Build feature datasets
    logger.info("Building weight trajectories")
    weight_trajectories = _build_weight_trajectories(tidy_df)
    weight_path = output_dir / "weight_trajectories.csv"
    weight_trajectories.to_csv(weight_path, index=False)
    
    logger.info("Building weekly nutrition aggregates")
    nutrition_weekly = _build_nutrition_weekly(fiber_df)
    nutrition_weekly_path = output_dir / "nutrition_weekly.csv"
    nutrition_weekly.to_csv(nutrition_weekly_path, index=False)
    
    logger.info("Extracting food keywords")
    food_keywords = _extract_food_keywords(nutrition_df)
    keywords_path = output_dir / "food_keywords.csv"
    food_keywords.to_csv(keywords_path, index=False)
    
    logger.info("Merging analysis dataset")
    analysis_df = _merge_analysis_dataset(weight_trajectories, nutrition_weekly, food_keywords)
    analysis_path = output_dir / "analysis_df.csv"
    analysis_df.to_csv(analysis_path, index=False)
    
    stats = {
        "weight_trajectories_rows": len(weight_trajectories),
        "nutrition_weekly_rows": len(nutrition_weekly),
        "food_keywords_rows": len(food_keywords),
        "analysis_df_rows": len(analysis_df),
        "subjects": len(analysis_df["subject_id"].unique()) if not analysis_df.empty else 0,
        "weeks": len(analysis_df["week"].unique()) if not analysis_df.empty else 0,
        "files_created": [
            "weight_trajectories.csv",
            "nutrition_weekly.csv", 
            "food_keywords.csv",
            "analysis_df.csv"
        ]
    }
    
    logger.info(f"Feature engineering completed: {stats}")
    return stats


def _build_weight_trajectories(tidy_df: pd.DataFrame) -> pd.DataFrame:
    """Build weight trajectories with deltas and percent changes."""
    
    if tidy_df.empty:
        return pd.DataFrame()
    
    # Ensure proper types
    tidy_df["date"] = pd.to_datetime(tidy_df["date"])
    tidy_df = tidy_df.sort_values(["subject_id", "date"])
    
    trajectories = []
    
    for subject_id in tidy_df["subject_id"].unique():
        subject_data = tidy_df[tidy_df["subject_id"] == subject_id].copy()
        
        # Calculate weight changes
        subject_data["weight_kg_prev"] = subject_data["weight_kg"].shift(1)
        subject_data["delta_kg"] = subject_data["weight_kg"] - subject_data["weight_kg_prev"]
        subject_data["delta_kg_pct"] = (subject_data["delta_kg"] / subject_data["weight_kg_prev"]) * 100
        
        # Next week delta (target variable)
        subject_data["delta_kg_next"] = subject_data["delta_kg"].shift(-1)
        
        # Baseline weight (first valid measurement)
        baseline_weight = subject_data["weight_kg"].dropna().iloc[0] if not subject_data["weight_kg"].dropna().empty else np.nan
        subject_data["baseline_weight_kg"] = baseline_weight
        subject_data["delta_from_baseline_kg"] = subject_data["weight_kg"] - baseline_weight
        subject_data["delta_from_baseline_pct"] = (subject_data["delta_from_baseline_kg"] / baseline_weight) * 100
        
        # Weight trajectory slope (OLS fit over available data)
        valid_weights = subject_data.dropna(subset=["weight_kg"])
        if len(valid_weights) >= 3:
            x = np.arange(len(valid_weights))
            y = valid_weights["weight_kg"].values
            slope = np.polyfit(x, y, 1)[0]  # kg per week
        else:
            slope = np.nan
        subject_data["weight_slope_kg_per_week"] = slope
        
        trajectories.append(subject_data)
    
    if not trajectories:
        return pd.DataFrame()
    
    result = pd.concat(trajectories, ignore_index=True)
    
    # Select key columns
    columns = [
        "subject_id", "date", "week", "weight_kg", "energy_1_10", "mood_1_10",
        "delta_kg", "delta_kg_pct", "delta_kg_next",
        "baseline_weight_kg", "delta_from_baseline_kg", "delta_from_baseline_pct",
        "weight_slope_kg_per_week"
    ]
    
    return result[columns]


def _build_nutrition_weekly(fiber_df: pd.DataFrame) -> pd.DataFrame:
    """Build weekly nutrition aggregates from daily fiber data."""
    
    if fiber_df.empty:
        return pd.DataFrame()
    
    # Ensure proper types
    fiber_df["date"] = pd.to_datetime(fiber_df["date"])
    
    # Check if week column exists, otherwise create it from date
    if "week" not in fiber_df.columns:
        # Create week column from date (assuming start date is study week 1)
        fiber_df = fiber_df.sort_values(["subject_id", "date"])
        for subject_id in fiber_df["subject_id"].unique():
            subject_mask = fiber_df["subject_id"] == subject_id
            subject_dates = fiber_df.loc[subject_mask, "date"]
            min_date = subject_dates.min()
            fiber_df.loc[subject_mask, "week"] = ((subject_dates - min_date).dt.days // 7) + 1
    
    # Group by subject and week
    weekly_agg = fiber_df.groupby(["subject_id", "week"]).agg({
        "total_calories": ["mean", "sum", "std", "count"],
        "total_fiber_g": ["mean", "sum", "std", "count"]
    }).reset_index()
    
    # Flatten column names
    weekly_agg.columns = [
        "subject_id", "week",
        "calories_mean", "calories_sum", "calories_std", "calories_days",
        "fiber_g_mean", "fiber_g_sum", "fiber_g_std", "fiber_days"
    ]
    
    # Calculate weekly totals and averages
    weekly_agg["calories_week"] = weekly_agg["calories_sum"]
    weekly_agg["fiber_g_week"] = weekly_agg["fiber_g_sum"]
    weekly_agg["calories_day_avg"] = weekly_agg["calories_mean"]
    weekly_agg["fiber_g_day_avg"] = weekly_agg["fiber_g_mean"]
    
    # Fiber to calorie ratio
    weekly_agg["fiber_per_1000_cal"] = (weekly_agg["fiber_g_week"] / weekly_agg["calories_week"]) * 1000
    weekly_agg["fiber_per_1000_cal"] = weekly_agg["fiber_per_1000_cal"].replace([np.inf, -np.inf], np.nan)
    
    return weekly_agg


def _extract_food_keywords(nutrition_df: pd.DataFrame) -> pd.DataFrame:
    """Extract food category keywords from nutrition data."""
    
    if nutrition_df.empty:
        return pd.DataFrame()
    
    # Food category keywords from README
    food_patterns = {
        "potato": r"\b(potato|potatoes|baked potato|mashed|roasted potato|potato salad)\b",
        "beans": r"\b(beans|bean|chickpea|chickpeas|lentils|lentil|kidney bean|black bean)\b",
        "rice": r"\b(rice|brown rice|white rice|wild rice|jasmine rice|basmati)\b",
        "oats": r"\b(oats|oatmeal|steel cut|overnight oats|porridge)\b",
        "bread": r"\b(bread|toast|bagel|whole wheat|sourdough|baguette)\b",
        "fruit": r"\b(fruit|apple|banana|berries|berry|orange|grape|strawberry)\b",
        "vegetables": r"\b(vegetables|vegetable|carrot|broccoli|spinach|pepper|tomato)\b",
        "dairy": r"\b(dairy|milk|cheese|yogurt|yoghurt|cottage cheese|cream)\b",
        "meat": r"\b(meat|chicken|fish|salmon|beef|pork|turkey|lamb)\b",
        "egg": r"\b(eggs|egg|scrambled|omelet|omelette|hard boiled|fried egg)\b",
        "nuts": r"\b(nuts|nut|almond|walnut|cashew|peanut|pecan|hazelnut)\b"
    }
    
    # Ensure proper types
    nutrition_df["date"] = pd.to_datetime(nutrition_df["date"])
    
    # Check if week column exists, otherwise create it from date
    if "week" not in nutrition_df.columns:
        # Create week column from date (assuming start date is study week 1)
        nutrition_df = nutrition_df.sort_values(["subject_id", "date"])
        for subject_id in nutrition_df["subject_id"].unique():
            subject_mask = nutrition_df["subject_id"] == subject_id
            subject_dates = nutrition_df.loc[subject_mask, "date"]
            if not subject_dates.empty:
                min_date = subject_dates.min()
                nutrition_df.loc[subject_mask, "week"] = ((subject_dates - min_date).dt.days // 7) + 1
    
    results = []
    
    for subject_id in nutrition_df["subject_id"].unique():
        subject_data = nutrition_df[nutrition_df["subject_id"] == subject_id]
        
        for week in subject_data["week"].unique():
            if pd.isna(week):
                continue
                
            week_data = subject_data[subject_data["week"] == week]
            
            # Combine all food items for this subject-week
            food_text = " ".join(week_data["food_item"].astype(str).str.lower())
            
            # Extract keyword flags
            week_result = {
                "subject_id": subject_id,
                "week": int(week)
            }
            
            for category, pattern in food_patterns.items():
                # Binary flag: any mention of category
                week_result[f"{category}_week"] = int(bool(re.search(pattern, food_text, re.IGNORECASE)))
                
                # Count of mentions
                matches = re.findall(pattern, food_text, re.IGNORECASE)
                week_result[f"{category}_mentions"] = len(matches)
            
            # Total food variety (number of unique categories)
            category_flags = [week_result[f"{cat}_week"] for cat in food_patterns.keys()]
            week_result["food_variety_categories"] = sum(category_flags)
            
            # Total food entries this week
            week_result["food_entries_week"] = len(week_data)
            
            results.append(week_result)
    
    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)


def _merge_analysis_dataset(
    weight_trajectories: pd.DataFrame,
    nutrition_weekly: pd.DataFrame, 
    food_keywords: pd.DataFrame
) -> pd.DataFrame:
    """Merge all feature datasets into analysis-ready format."""
    
    if weight_trajectories.empty:
        return pd.DataFrame()
    
    # Start with weight trajectories as base
    analysis_df = weight_trajectories.copy()
    
    # Merge nutrition weekly data
    if not nutrition_weekly.empty:
        analysis_df = analysis_df.merge(
            nutrition_weekly,
            on=["subject_id", "week"],
            how="left"
        )
    
    # Merge food keywords
    if not food_keywords.empty:
        analysis_df = analysis_df.merge(
            food_keywords,
            on=["subject_id", "week"], 
            how="left"
        )
    
    # Fill missing keyword flags with 0 (no mention)
    keyword_columns = [col for col in analysis_df.columns if col.endswith("_week") and col not in ["week"]]
    for col in keyword_columns:
        analysis_df[col] = analysis_df[col].fillna(0)
    
    # Create derived features
    if "calories_week" in analysis_df.columns and "fiber_g_week" in analysis_df.columns:
        # High fiber week flag (above median)
        fiber_median = analysis_df["fiber_g_week"].median()
        analysis_df["high_fiber_week"] = (analysis_df["fiber_g_week"] > fiber_median).astype(int)
        
        # High calorie week flag  
        cal_median = analysis_df["calories_week"].median()
        analysis_df["high_calorie_week"] = (analysis_df["calories_week"] > cal_median).astype(int)
    
    # Remove rows without next week delta (can't be used for modeling)
    analysis_df = analysis_df.dropna(subset=["delta_kg_next"])
    
    return analysis_df