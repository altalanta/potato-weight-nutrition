"""Data contracts and validation using Great Expectations."""

import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd


def validate_data_contracts(input_dir: Path) -> Dict[str, Any]:
    """Run Great Expectations validation on data files.
    
    Args:
        input_dir: Directory containing data files to validate
        
    Returns:
        Dictionary with validation results
    """
    logger = logging.getLogger("pwn.contracts")
    
    try:
        import great_expectations as gx
        from great_expectations.core import ExpectationSuite
        from great_expectations.dataset import PandasDataset
    except ImportError:
        raise ImportError("Great Expectations not installed. Install with: pip install pwn[contracts]")
    
    results = {
        "validation_timestamp": pd.Timestamp.now().isoformat(),
        "files_validated": [],
        "validation_results": {},
        "overall_success": True
    }
    
    # Expected data files
    expected_files = [
        "Potato_tidy.csv",
        "Potato_nutrition_rows.csv", 
        "Potato_fiber_daily.csv"
    ]
    
    for filename in expected_files:
        file_path = input_dir / filename
        
        if not file_path.exists():
            logger.warning(f"Data file not found: {filename}")
            results["validation_results"][filename] = {
                "success": False,
                "error": "File not found"
            }
            results["overall_success"] = False
            continue
            
        try:
            # Load data
            df = pd.read_csv(file_path)
            results["files_validated"].append(filename)
            
            # Create PandasDataset for validation
            dataset = PandasDataset(df)
            
            # Run file-specific validations
            if filename == "Potato_tidy.csv":
                file_results = _validate_tidy_data(dataset)
            elif filename == "Potato_nutrition_rows.csv":
                file_results = _validate_nutrition_data(dataset)
            elif filename == "Potato_fiber_daily.csv":
                file_results = _validate_fiber_data(dataset)
            else:
                file_results = {"success": True, "expectations_met": 0}
            
            results["validation_results"][filename] = file_results
            
            if not file_results["success"]:
                results["overall_success"] = False
                
        except Exception as e:
            logger.error(f"Validation failed for {filename}: {e}")
            results["validation_results"][filename] = {
                "success": False,
                "error": str(e)
            }
            results["overall_success"] = False
    
    logger.info(f"Data validation completed. Overall success: {results['overall_success']}")
    return results


def _validate_tidy_data(dataset: "PandasDataset") -> Dict[str, Any]:
    """Validate Potato_tidy.csv structure and content."""
    
    expectations_met = 0
    total_expectations = 0
    failures = []
    
    # Expected columns
    expected_columns = ["subject_id", "date", "week", "weight_kg", "energy_1_10", "mood_1_10"]
    
    total_expectations += 1
    if dataset.expect_table_columns_to_match_ordered_list(column_list=expected_columns).success:
        expectations_met += 1
    else:
        failures.append("Column structure mismatch")
    
    # Data type expectations
    total_expectations += 1
    if dataset.expect_column_values_to_be_of_type("subject_id", "int64").success:
        expectations_met += 1
    else:
        failures.append("subject_id should be integer")
    
    total_expectations += 1
    if dataset.expect_column_values_to_not_be_null("subject_id").success:
        expectations_met += 1
    else:
        failures.append("subject_id should not be null")
    
    # Weight should be reasonable
    total_expectations += 1
    if dataset.expect_column_values_to_be_between("weight_kg", min_value=30, max_value=200, mostly=0.95).success:
        expectations_met += 1
    else:
        failures.append("weight_kg values outside reasonable range")
    
    # Energy and mood should be 1-10 scale
    total_expectations += 1
    if dataset.expect_column_values_to_be_between("energy_1_10", min_value=1, max_value=10, mostly=0.90).success:
        expectations_met += 1
    else:
        failures.append("energy_1_10 values outside 1-10 range")
    
    total_expectations += 1
    if dataset.expect_column_values_to_be_between("mood_1_10", min_value=1, max_value=10, mostly=0.90).success:
        expectations_met += 1
    else:
        failures.append("mood_1_10 values outside 1-10 range")
    
    # Week should be sequential and reasonable
    total_expectations += 1
    if dataset.expect_column_values_to_be_between("week", min_value=1, max_value=52).success:
        expectations_met += 1
    else:
        failures.append("week values outside reasonable range")
    
    success = len(failures) == 0
    
    return {
        "success": success,
        "expectations_met": expectations_met,
        "total_expectations": total_expectations,
        "failures": failures
    }


def _validate_nutrition_data(dataset: "PandasDataset") -> Dict[str, Any]:
    """Validate Potato_nutrition_rows.csv structure and content."""
    
    expectations_met = 0
    total_expectations = 0
    failures = []
    
    # Expected columns
    expected_columns = ["subject_id", "date", "food_item", "category", "calories", "protein_g", "carbs_g", "fat_g", "fiber_g"]
    
    total_expectations += 1
    if dataset.expect_table_columns_to_match_ordered_list(column_list=expected_columns).success:
        expectations_met += 1
    else:
        failures.append("Column structure mismatch")
    
    # Required fields should not be null
    total_expectations += 1
    if dataset.expect_column_values_to_not_be_null("subject_id").success:
        expectations_met += 1
    else:
        failures.append("subject_id should not be null")
    
    total_expectations += 1
    if dataset.expect_column_values_to_not_be_null("food_item").success:
        expectations_met += 1
    else:
        failures.append("food_item should not be null")
    
    # Nutritional values should be non-negative
    total_expectations += 1
    if dataset.expect_column_values_to_be_between("calories", min_value=0, max_value=5000).success:
        expectations_met += 1
    else:
        failures.append("calories values outside reasonable range")
    
    total_expectations += 1
    if dataset.expect_column_values_to_be_between("fiber_g", min_value=0, max_value=100).success:
        expectations_met += 1
    else:
        failures.append("fiber_g values outside reasonable range")
    
    total_expectations += 1
    if dataset.expect_column_values_to_be_between("protein_g", min_value=0, max_value=200).success:
        expectations_met += 1
    else:
        failures.append("protein_g values outside reasonable range")
    
    # Food categories should be from expected set
    expected_categories = ["potato", "beans", "rice", "oats", "bread", "fruit", "vegetables", "dairy", "meat", "egg", "nuts"]
    total_expectations += 1
    if dataset.expect_column_values_to_be_in_set("category", value_set=expected_categories).success:
        expectations_met += 1
    else:
        failures.append("Unknown food categories found")
    
    success = len(failures) == 0
    
    return {
        "success": success,
        "expectations_met": expectations_met,
        "total_expectations": total_expectations,
        "failures": failures
    }


def _validate_fiber_data(dataset: "PandasDataset") -> Dict[str, Any]:
    """Validate Potato_fiber_daily.csv structure and content."""
    
    expectations_met = 0
    total_expectations = 0
    failures = []
    
    # Expected columns
    expected_columns = ["subject_id", "date", "week", "total_calories", "total_fiber_g"]
    
    total_expectations += 1
    if dataset.expect_table_columns_to_match_ordered_list(column_list=expected_columns).success:
        expectations_met += 1
    else:
        failures.append("Column structure mismatch")
    
    # Required fields
    total_expectations += 1
    if dataset.expect_column_values_to_not_be_null("subject_id").success:
        expectations_met += 1
    else:
        failures.append("subject_id should not be null")
    
    # Reasonable value ranges
    total_expectations += 1
    if dataset.expect_column_values_to_be_between("total_calories", min_value=500, max_value=8000, mostly=0.95).success:
        expectations_met += 1
    else:
        failures.append("total_calories values outside reasonable range")
    
    total_expectations += 1
    if dataset.expect_column_values_to_be_between("total_fiber_g", min_value=0, max_value=150, mostly=0.95).success:
        expectations_met += 1
    else:
        failures.append("total_fiber_g values outside reasonable range")
    
    total_expectations += 1
    if dataset.expect_column_values_to_be_between("week", min_value=1, max_value=52).success:
        expectations_met += 1
    else:
        failures.append("week values outside reasonable range")
    
    success = len(failures) == 0
    
    return {
        "success": success,
        "expectations_met": expectations_met,
        "total_expectations": total_expectations,
        "failures": failures
    }