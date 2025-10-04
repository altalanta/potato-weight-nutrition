"""
Pandera schemas for nutrition data validation.

Optional complement to Great Expectations - provides programmatic 
data validation with type hints and runtime checking.
"""


import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series


class BronzeNutritionSchema(pa.SchemaModel):
    """Schema for raw nutrition data from CSV files."""
    
    id: Series[str] = pa.Field(
        regex=r"^[A-Z0-9_]+$",
        description="Subject identifier (alphanumeric with underscores)"
    )
    
    date: Series[pd.Timestamp] = pa.Field(
        ge=pd.Timestamp("2020-01-01"),
        le=pd.Timestamp("2030-12-31"),
        description="Date of nutrition entry"
    )
    
    food_desc: Series[str] = pa.Field(
        str_length={"min_val": 2, "max_val": 500},
        nullable=False,
        description="Food description text"
    )
    
    calories: Series[float] = pa.Field(
        ge=0.0,
        le=10000.0,
        description="Daily caloric intake (kcal)"
    )
    
    fiber_g: Series[float] = pa.Field(
        ge=0.0,
        le=200.0,
        nullable=True,
        description="Fiber intake (grams)"
    )
    
    protein_g: Series[float] = pa.Field(
        ge=0.0,
        le=500.0,
        nullable=True,
        description="Protein intake (grams)"
    )
    
    fat_g: Series[float] = pa.Field(
        ge=0.0,
        le=300.0,
        nullable=True,
        description="Fat intake (grams)"
    )
    
    carbs_g: Series[float] = pa.Field(
        ge=0.0,
        le=1000.0,
        nullable=True,
        description="Carbohydrate intake (grams)"
    )
    
    class Config:
        name = "bronze_nutrition_data"
        description = "Raw nutrition data from CSV files"
        strict = False  # Allow additional columns
        coerce = True   # Convert types when possible


class SilverFeaturesSchema(pa.SchemaModel):
    """Schema for engineered features ready for modeling."""
    
    subject_id: Series[str] = pa.Field(
        regex=r"^[A-Z0-9_]+$",
        description="Clean subject identifier"
    )
    
    study_week: Series[int] = pa.Field(
        ge=1,
        le=52,
        description="Study week number (1-52)"
    )
    
    delta_kg_next: Series[float] = pa.Field(
        ge=-10.0,
        le=10.0,
        nullable=True,
        description="Weight change to next week (kg)"
    )
    
    fiber_per_1000kcal: Series[float] = pa.Field(
        ge=0.0,
        le=100.0,
        nullable=True,
        description="Fiber density (g per 1000 kcal)"
    )
    
    calories_week: Series[float] = pa.Field(
        ge=5000.0,
        le=30000.0,
        nullable=True,
        description="Weekly caloric intake (kcal)"
    )
    
    potato_week: Series[int] = pa.Field(
        isin=[0, 1],
        description="Binary indicator for potato consumption week"
    )
    
    beans_week: Series[int] = pa.Field(
        isin=[0, 1],
        description="Binary indicator for beans consumption week"
    )
    
    weight_kg_baseline: Series[float] = pa.Field(
        ge=30.0,
        le=300.0,
        description="Baseline weight (kg)"
    )
    
    weight_kg_current: Series[float] = pa.Field(
        ge=30.0,
        le=300.0,
        description="Current week weight (kg)"
    )
    
    @pa.check("delta_kg_next", name="reasonable_weight_change")
    def weight_change_distribution(cls, series: Series[float]) -> bool:
        """Check that weight changes have reasonable statistical properties."""
        valid_changes = series.dropna()
        if len(valid_changes) == 0:
            return True
        
        # Mean weekly change should be small
        mean_change = valid_changes.mean()
        if abs(mean_change) > 2.0:
            return False
            
        # Standard deviation should be reasonable  
        std_change = valid_changes.std()
        if std_change > 5.0:
            return False
            
        return True
    
    @pa.check("fiber_per_1000kcal", name="fiber_density_sanity")
    def fiber_density_reasonable(cls, series: Series[float]) -> bool:
        """Check that fiber density values are nutritionally reasonable."""
        valid_fiber = series.dropna()
        if len(valid_fiber) == 0:
            return True
            
        # Median should be in typical range
        median_fiber = valid_fiber.median()
        return 5.0 <= median_fiber <= 50.0
    
    class Config:
        name = "silver_features"
        description = "Engineered features for statistical modeling"
        strict = True   # Exact column matching for modeling data
        coerce = True   # Convert types when possible


# Type aliases for use in pipeline code
BronzeNutritionDF = DataFrame[BronzeNutritionSchema]
SilverFeaturesDF = DataFrame[SilverFeaturesSchema]


def validate_bronze_nutrition(df: pd.DataFrame) -> BronzeNutritionDF:
    """
    Validate raw nutrition data against bronze schema.
    
    Args:
        df: Raw nutrition DataFrame from CSV
        
    Returns:
        Validated DataFrame with correct types
        
    Raises:
        pandera.errors.SchemaError: If validation fails
    """
    return BronzeNutritionSchema.validate(df, lazy=True)


def validate_silver_features(df: pd.DataFrame) -> SilverFeaturesDF:
    """
    Validate engineered features against silver schema.
    
    Args:
        df: Features DataFrame for modeling
        
    Returns:
        Validated DataFrame with correct types
        
    Raises:
        pandera.errors.SchemaError: If validation fails
    """
    return SilverFeaturesSchema.validate(df, lazy=True)


# Example usage in pipeline:
"""
from data_contracts.schemas.nutrition_schema import validate_bronze_nutrition

# In data loading step
raw_df = pd.read_csv("bronze_nutrition.csv")
validated_df = validate_bronze_nutrition(raw_df)

# In feature engineering step  
features_df = engineer_features(validated_df)
validated_features = validate_silver_features(features_df)
"""