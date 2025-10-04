"""Simple data schemas for validation."""

from typing import Optional

from pydantic import BaseModel


class TidyDataRecord(BaseModel):
    """Schema for Potato_tidy.csv records."""
    subject_id: int
    date: str
    week: int
    weight_kg: float | None = None
    energy_1_10: float | None = None
    mood_1_10: float | None = None


class NutritionRecord(BaseModel):
    """Schema for Potato_nutrition_rows.csv records."""
    subject_id: int
    date: str
    food_item: str
    category: str
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    fiber_g: float


class FiberDailyRecord(BaseModel):
    """Schema for Potato_fiber_daily.csv records."""
    subject_id: int
    date: str
    week: int
    total_calories: float | None = None
    total_fiber_g: float | None = None


def validate_dataframe_schema(df, schema_class):
    """Simple validation function."""
    errors = []
    
    if df.empty:
        return ["DataFrame is empty"]
    
    # Check required columns
    schema_fields = schema_class.__annotations__
    required_cols = [name for name, annotation in schema_fields.items() 
                    if not (hasattr(annotation, '__origin__') and annotation.__origin__ is type(Optional[int]).__origin__)]
    missing_cols = set(required_cols) - set(df.columns)
    
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return errors
    
    # Validate a sample of rows
    sample_size = min(10, len(df))
    sample_df = df.head(sample_size)
    
    for idx, row in sample_df.iterrows():
        try:
            schema_class(**row.to_dict())
        except Exception as e:
            errors.append(f"Row {idx}: {str(e)}")
            if len(errors) >= 5:
                errors.append("... (additional validation errors truncated)")
                break
    
    return errors