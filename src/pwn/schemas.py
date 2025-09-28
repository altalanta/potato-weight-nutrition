"""Data schemas and validation for potato-weight-nutrition pipeline."""

from datetime import date
from typing import Optional

import pandas as pd
import polars as pl
from pydantic import BaseModel, Field


class TidyDataRecord(BaseModel):
    """Schema for Potato_tidy.csv records (weight/energy/mood measurements)."""
    
    subject_id: int = Field(..., description="Subject identifier")
    date: str = Field(..., description="Measurement date (string format)")
    week: int = Field(..., description="Study week number")
    weight_kg: Optional[float] = Field(None, description="Weight in kilograms")
    energy_1_10: Optional[float] = Field(None, description="Energy level (1-10 scale)")
    mood_1_10: Optional[float] = Field(None, description="Mood level (1-10 scale)")


class NutritionRecord(BaseModel):
    """Schema for Potato_nutrition_rows.csv records (daily food entries)."""
    
    subject_id: int = Field(..., description="Subject identifier")
    date: str = Field(..., description="Entry date (string format)")
    food_item: str = Field(..., description="Food item name")
    category: str = Field(..., description="Food category")
    calories: float = Field(..., description="Calories")
    protein_g: float = Field(..., description="Protein in grams")
    carbs_g: float = Field(..., description="Carbohydrates in grams")
    fat_g: float = Field(..., description="Fat in grams")
    fiber_g: float = Field(..., description="Fiber in grams")


class FiberDailyRecord(BaseModel):
    """Schema for Potato_fiber_daily.csv records (daily fiber/calorie rollups)."""
    
    subject_id: int = Field(..., description="Subject identifier")
    date: str = Field(..., description="Date (string format)")
    week: int = Field(..., description="Study week number")
    total_calories: Optional[float] = Field(None, description="Total daily calories")
    total_fiber_g: Optional[float] = Field(None, description="Total daily fiber in grams")


class WeightTrajectoryRecord(BaseModel):
    """Schema for weight_trajectories.csv (processed weight data)."""
    
    subject_id: int = Field(..., gt=0)
    date: date = Field(...)
    week: int = Field(..., ge=1)
    weight_kg: Optional[float] = Field(None, gt=0)
    energy_1_10: Optional[float] = Field(None, ge=1, le=10)
    mood_1_10: Optional[float] = Field(None, ge=1, le=10)
    delta_kg: Optional[float] = Field(None, description="Weight change from previous week")
    delta_kg_pct: Optional[float] = Field(None, description="Percent weight change")
    delta_kg_next: Optional[float] = Field(None, description="Next week weight change (target)")
    baseline_weight_kg: Optional[float] = Field(None, gt=0)
    delta_from_baseline_kg: Optional[float] = Field(None)
    delta_from_baseline_pct: Optional[float] = Field(None)
    weight_slope_kg_per_week: Optional[float] = Field(None)


class NutritionWeeklyRecord(BaseModel):
    """Schema for nutrition_weekly.csv (weekly nutrition aggregates)."""
    
    subject_id: int = Field(..., gt=0)
    week: int = Field(..., ge=1)
    calories_mean: Optional[float] = Field(None, ge=0)
    calories_sum: Optional[float] = Field(None, ge=0)
    calories_std: Optional[float] = Field(None, ge=0)
    calories_days: Optional[int] = Field(None, ge=0, le=7)
    fiber_g_mean: Optional[float] = Field(None, ge=0)
    fiber_g_sum: Optional[float] = Field(None, ge=0)
    fiber_g_std: Optional[float] = Field(None, ge=0)
    fiber_days: Optional[int] = Field(None, ge=0, le=7)
    calories_week: Optional[float] = Field(None, ge=0)
    fiber_g_week: Optional[float] = Field(None, ge=0)
    calories_day_avg: Optional[float] = Field(None, ge=0)
    fiber_g_day_avg: Optional[float] = Field(None, ge=0)
    fiber_per_1000_cal: Optional[float] = Field(None, ge=0)


class AnalysisRecord(BaseModel):
    """Schema for analysis_df.csv (merged modeling dataset)."""
    
    subject_id: int = Field(..., gt=0)
    week: int = Field(..., ge=1)
    
    # Weight trajectory features
    weight_kg: Optional[float] = Field(None, gt=0)
    delta_kg_next: Optional[float] = Field(None)  # Target variable
    delta_kg: Optional[float] = Field(None)
    
    # Nutrition features
    calories_week: Optional[float] = Field(None, ge=0)
    fiber_g_week: Optional[float] = Field(None, ge=0)
    
    # Food category features
    beans_week: Optional[float] = Field(None, ge=0)
    potato_week: Optional[float] = Field(None, ge=0)
    rice_week: Optional[float] = Field(None, ge=0)
    oats_week: Optional[float] = Field(None, ge=0)
    bread_week: Optional[float] = Field(None, ge=0)
    fruit_week: Optional[float] = Field(None, ge=0)
    vegetables_week: Optional[float] = Field(None, ge=0)
    dairy_week: Optional[float] = Field(None, ge=0)
    meat_week: Optional[float] = Field(None, ge=0)
    egg_week: Optional[float] = Field(None, ge=0)
    nuts_week: Optional[float] = Field(None, ge=0)


def validate_dataframe_schema(df: pd.DataFrame, schema_class: type[BaseModel]) -> list[str]:
    """Validate a pandas DataFrame against a Pydantic schema.
    
    Args:
        df: DataFrame to validate
        schema_class: Pydantic model class to validate against
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if df.empty:
        return ["DataFrame is empty"]
    
    # Check required columns
    schema_fields = schema_class.__fields__
    required_cols = [name for name, field in schema_fields.items() if field.is_required()]
    missing_cols = set(required_cols) - set(df.columns)
    
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return errors
    
    # Validate a sample of rows (first 100 for performance)
    sample_size = min(100, len(df))
    sample_df = df.head(sample_size)
    
    for idx, row in sample_df.iterrows():
        try:
            # Convert row to dict and validate
            row_dict = row.to_dict()
            schema_class(**row_dict)
        except Exception as e:
            errors.append(f"Row {idx}: {str(e)}")
            if len(errors) >= 10:  # Limit error reporting
                errors.append("... (additional validation errors truncated)")
                break
    
    return errors


def validate_polars_schema(df: pl.DataFrame, expected_schema: dict[str, pl.DataType]) -> list[str]:
    """Validate a Polars DataFrame schema.
    
    Args:
        df: Polars DataFrame to validate
        expected_schema: Dictionary mapping column names to expected data types
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if df.is_empty():
        return ["DataFrame is empty"]
    
    # Check for missing columns
    missing_cols = set(expected_schema.keys()) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # Check for unexpected columns
    extra_cols = set(df.columns) - set(expected_schema.keys())
    if extra_cols:
        errors.append(f"Unexpected columns: {extra_cols}")
    
    # Check data types
    for col_name, expected_type in expected_schema.items():
        if col_name in df.columns:
            actual_type = df[col_name].dtype
            if actual_type != expected_type:
                errors.append(f"Column '{col_name}': expected {expected_type}, got {actual_type}")
    
    return errors


# Polars schema definitions
TIDY_DATA_SCHEMA = {
    "subject_id": pl.Int64,
    "date": pl.Date,
    "week": pl.Int64,
    "weight_kg": pl.Float64,
    "energy_1_10": pl.Float64,
    "mood_1_10": pl.Float64
}

NUTRITION_SCHEMA = {
    "subject_id": pl.Int64,
    "date": pl.Date,
    "food_item": pl.Utf8,
    "category": pl.Utf8,
    "calories": pl.Float64,
    "protein_g": pl.Float64,
    "carbs_g": pl.Float64,
    "fat_g": pl.Float64,
    "fiber_g": pl.Float64
}

FIBER_DAILY_SCHEMA = {
    "subject_id": pl.Int64,
    "date": pl.Date,
    "week": pl.Int64,
    "total_calories": pl.Float64,
    "total_fiber_g": pl.Float64
}

WEIGHT_TRAJECTORY_SCHEMA = {
    "subject_id": pl.Int64,
    "date": pl.Date,
    "week": pl.Int64,
    "weight_kg": pl.Float64,
    "energy_1_10": pl.Float64,
    "mood_1_10": pl.Float64,
    "delta_kg": pl.Float64,
    "delta_kg_pct": pl.Float64,
    "delta_kg_next": pl.Float64,
    "baseline_weight_kg": pl.Float64,
    "delta_from_baseline_kg": pl.Float64,
    "delta_from_baseline_pct": pl.Float64,
    "weight_slope_kg_per_week": pl.Float64
}

NUTRITION_WEEKLY_SCHEMA = {
    "subject_id": pl.Int64,
    "week": pl.Int64,
    "calories_mean": pl.Float64,
    "calories_sum": pl.Float64,
    "calories_std": pl.Float64,
    "calories_days": pl.Int64,
    "fiber_g_mean": pl.Float64,
    "fiber_g_sum": pl.Float64,
    "fiber_g_std": pl.Float64,
    "fiber_days": pl.Int64,
    "calories_week": pl.Float64,
    "fiber_g_week": pl.Float64,
    "calories_day_avg": pl.Float64,
    "fiber_g_day_avg": pl.Float64,
    "fiber_per_1000_cal": pl.Float64
}

ANALYSIS_SCHEMA = {
    "subject_id": pl.Int64,
    "week": pl.Int64,
    "weight_kg": pl.Float64,
    "delta_kg_next": pl.Float64,
    "delta_kg": pl.Float64,
    "calories_week": pl.Float64,
    "fiber_g_week": pl.Float64,
    "beans_week": pl.Float64,
    "potato_week": pl.Float64,
    "rice_week": pl.Float64,
    "oats_week": pl.Float64,
    "bread_week": pl.Float64,
    "fruit_week": pl.Float64,
    "vegetables_week": pl.Float64,
    "dairy_week": pl.Float64,
    "meat_week": pl.Float64,
    "egg_week": pl.Float64,
    "nuts_week": pl.Float64
}