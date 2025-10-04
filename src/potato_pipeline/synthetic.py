"""Synthetic data generation matching README schema for demonstration and testing."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def generate_synthetic_data(
    output_dir: Path,
    n_subjects: int = 50,
    n_weeks: int = 12,
    seed: int = 42
) -> dict[str, Any]:
    """Generate synthetic data matching README schema.
    
    Creates three files:
    - Potato_tidy.csv: Weight/energy/mood measurements
    - Potato_nutrition_rows.csv: Daily food entries with nutrients
    - Potato_fiber_daily.csv: Daily fiber/calorie rollups
    
    Args:
        output_dir: Directory to save synthetic data files
        n_subjects: Number of subjects to generate
        n_weeks: Number of weeks per subject
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with generation statistics
    """
    np.random.seed(seed)
    logger = logging.getLogger("pwn.synthetic")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate Potato_tidy.csv (weight/energy/mood measurements)
    logger.info(f"Generating Potato_tidy.csv with {n_subjects} subjects, {n_weeks} weeks")
    tidy_data = _generate_tidy_data(n_subjects, n_weeks)
    tidy_path = output_dir / "Potato_tidy.csv"
    tidy_data.to_csv(tidy_path, index=False)
    
    # Generate Potato_nutrition_rows.csv (daily food entries)
    logger.info("Generating Potato_nutrition_rows.csv with daily food entries")
    nutrition_data = _generate_nutrition_data(n_subjects, n_weeks)
    nutrition_path = output_dir / "Potato_nutrition_rows.csv"
    nutrition_data.to_csv(nutrition_path, index=False)
    
    # Generate Potato_fiber_daily.csv (daily fiber/calorie rollups)
    logger.info("Generating Potato_fiber_daily.csv with daily rollups")
    fiber_data = _generate_fiber_data(n_subjects, n_weeks)
    fiber_path = output_dir / "Potato_fiber_daily.csv"
    fiber_data.to_csv(fiber_path, index=False)
    
    stats = {
        "n_subjects": n_subjects,
        "n_weeks": n_weeks,
        "files_generated": [
            str(tidy_path.name),
            str(nutrition_path.name), 
            str(fiber_path.name)
        ],
        "tidy_rows": len(tidy_data),
        "nutrition_rows": len(nutrition_data),
        "fiber_rows": len(fiber_data),
        "seed": seed
    }
    
    logger.info(f"Synthetic data generation completed: {stats}")
    return stats


def _generate_tidy_data(n_subjects: int, n_weeks: int) -> pd.DataFrame:
    """Generate weight/energy/mood measurements (Potato_tidy.csv)."""
    
    records = []
    base_date = datetime(2023, 1, 1)
    
    for subject_id in range(1, n_subjects + 1):
        # Subject-specific baseline weight (50-100 kg)
        baseline_weight = np.random.uniform(50, 100)
        
        for week in range(1, n_weeks + 1):
            # Weekly measurements with realistic variations
            date = base_date + timedelta(weeks=week-1, days=np.random.randint(0, 7))
            
            # Weight trajectory with some randomness
            weight_trend = np.random.normal(0, 0.5)  # kg per week
            week_noise = np.random.normal(0, 1.0)   # measurement noise
            weight = baseline_weight + (week * weight_trend) + week_noise
            
            # Energy and mood correlated with weight changes
            energy = np.random.uniform(1, 10)
            mood = np.random.uniform(1, 10)
            
            # Some missing data (10% probability)
            if np.random.random() < 0.1:
                weight = np.nan
            if np.random.random() < 0.05:
                energy = np.nan
            if np.random.random() < 0.05:
                mood = np.nan
            
            records.append({
                "subject_id": subject_id,
                "date": date.strftime("%Y-%m-%d"),
                "week": week,
                "weight_kg": weight,
                "energy_1_10": energy,
                "mood_1_10": mood
            })
    
    return pd.DataFrame(records)


def _generate_nutrition_data(n_subjects: int, n_weeks: int) -> pd.DataFrame:
    """Generate daily food entries (Potato_nutrition_rows.csv)."""
    
    # Food categories from README
    food_categories = {
        "potato": ["baked potato", "mashed potato", "roasted potatoes", "potato salad"],
        "beans": ["black beans", "chickpeas", "lentils", "kidney beans"],
        "rice": ["brown rice", "white rice", "wild rice", "jasmine rice"],
        "oats": ["steel cut oats", "oatmeal", "overnight oats"],
        "bread": ["whole wheat bread", "sourdough", "bagel", "toast"],
        "fruit": ["apple", "banana", "berries", "orange"],
        "vegetables": ["broccoli", "spinach", "carrots", "peppers"],
        "dairy": ["greek yogurt", "cheese", "milk", "cottage cheese"],
        "meat": ["chicken breast", "salmon", "ground turkey", "lean beef"],
        "egg": ["scrambled eggs", "omelet", "hard boiled eggs"],
        "nuts": ["almonds", "walnuts", "cashews", "peanut butter"]
    }
    
    records = []
    base_date = datetime(2023, 1, 1)
    
    for subject_id in range(1, n_subjects + 1):
        for week in range(1, n_weeks + 1):
            # 3-7 food entries per week per subject
            n_entries = np.random.randint(3, 8)
            
            for entry in range(n_entries):
                # Random day within the week
                day_offset = week * 7 + np.random.randint(0, 7)
                date = base_date + timedelta(days=day_offset)
                
                # Random food category and item
                category = np.random.choice(list(food_categories.keys()))
                food_item = np.random.choice(food_categories[category])
                
                # Nutritional values with realistic ranges
                fiber_g = np.random.uniform(0.5, 15.0) if category in ["beans", "oats", "fruit", "vegetables"] else np.random.uniform(0, 5.0)
                protein_g = np.random.uniform(1.0, 30.0) if category in ["meat", "dairy", "egg", "beans"] else np.random.uniform(0.5, 10.0)
                carbs_g = np.random.uniform(5.0, 50.0) if category in ["potato", "rice", "bread", "fruit"] else np.random.uniform(1.0, 20.0)
                fat_g = np.random.uniform(0.1, 25.0) if category in ["nuts", "dairy", "meat"] else np.random.uniform(0.1, 10.0)
                
                # Calculate calories (4 cal/g carbs+protein, 9 cal/g fat)
                calories = (carbs_g + protein_g) * 4 + fat_g * 9
                
                records.append({
                    "subject_id": subject_id,
                    "date": date.strftime("%Y-%m-%d"),
                    "food_item": food_item,
                    "category": category,
                    "calories": round(calories, 1),
                    "protein_g": round(protein_g, 1),
                    "carbs_g": round(carbs_g, 1),
                    "fat_g": round(fat_g, 1),
                    "fiber_g": round(fiber_g, 1)
                })
    
    return pd.DataFrame(records)


def _generate_fiber_data(n_subjects: int, n_weeks: int) -> pd.DataFrame:
    """Generate daily fiber/calorie rollups (Potato_fiber_daily.csv)."""
    
    records = []
    base_date = datetime(2023, 1, 1)
    
    for subject_id in range(1, n_subjects + 1):
        for week in range(1, n_weeks + 1):
            # Generate 5-7 days per week (some missing days)
            n_days = np.random.randint(5, 8)
            days = np.random.choice(7, size=n_days, replace=False)
            
            for day in days:
                date = base_date + timedelta(weeks=week-1, days=int(day))
                
                # Daily totals with realistic ranges
                total_calories = np.random.uniform(1200, 3000)
                total_fiber = np.random.uniform(10, 50)  # grams per day
                
                # Some days have missing data
                if np.random.random() < 0.08:
                    total_calories = np.nan
                if np.random.random() < 0.05:
                    total_fiber = np.nan
                
                records.append({
                    "subject_id": subject_id,
                    "date": date.strftime("%Y-%m-%d"),
                    "week": week,
                    "total_calories": round(total_calories, 1) if not pd.isna(total_calories) else np.nan,
                    "total_fiber_g": round(total_fiber, 1) if not pd.isna(total_fiber) else np.nan
                })
    
    return pd.DataFrame(records)