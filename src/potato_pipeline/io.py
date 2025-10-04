"""I/O utilities for loading and saving data."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger


def load_tidy(path: Path) -> pd.DataFrame:
    """Load tidy weight/energy/mood data."""
    if not path.exists():
        logger.warning(f"Tidy data file not found: {path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(path, dtype={
            'subject_id': 'str',
            'metric': 'str',
            'phase': 'str',
            'week': 'int',
            'value': 'float',
            'tp_index': 'Int64',  # Nullable integer
            'timepoint': 'str',
            'note': 'str',
            'sheet_name': 'str',
            'source_path': 'str'
        })
        logger.info(f"Loaded tidy data: {len(df)} rows, {df['subject_id'].nunique()} subjects")
        return df
    except Exception as e:
        logger.error(f"Error loading tidy data from {path}: {e}")
        return pd.DataFrame()


def load_nutrition_rows(path: Path) -> pd.DataFrame:
    """Load nutrition rows data with proper date parsing."""
    if not path.exists():
        logger.warning(f"Nutrition rows file not found: {path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(path, dtype={
            'subject_id': 'str',
            'food': 'str',
            'fiber_g': 'float',
            'calories_kcal': 'float',
            'carb_g': 'float',
            'protein_g': 'float',
            'fat_g': 'float',
            'sugar_g': 'float',
            'insoluble_fiber_g': 'float',
            'soluble_fiber_g': 'float',
            'sheet_name': 'str',
            'source_path': 'str'
        })
        
        # Parse date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
        logger.info(f"Loaded nutrition rows: {len(df)} rows, {df['subject_id'].nunique()} subjects")
        return df
    except Exception as e:
        logger.error(f"Error loading nutrition rows from {path}: {e}")
        return pd.DataFrame()


def load_fiber_daily(path: Path) -> pd.DataFrame:
    """Load daily fiber rollup data with proper date parsing."""
    if not path.exists():
        logger.warning(f"Daily fiber file not found: {path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(path, dtype={
            'subject_id': 'str',
            'fiber_g_total': 'float',
            'calories_kcal_total': 'float',
            'carb_g_total': 'float',
            'protein_g_total': 'float',
            'fat_g_total': 'float',
            'n_entries': 'int'
        })
        
        # Parse date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
        logger.info(f"Loaded daily fiber: {len(df)} rows, {df['subject_id'].nunique()} subjects")
        return df
    except Exception as e:
        logger.error(f"Error loading daily fiber from {path}: {e}")
        return pd.DataFrame()


def save_text(path: Path, text: str) -> None:
    """Save text to file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(text)
        logger.info(f"Saved text to {path}")
    except Exception as e:
        logger.error(f"Error saving text to {path}: {e}")


def save_table(df: pd.DataFrame, path_csv: Path, path_parquet: Path | None = None) -> None:
    """Save DataFrame to CSV and optionally Parquet."""
    try:
        # Create parent directories
        path_csv.parent.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        df.to_csv(path_csv, index=False)
        logger.info(f"Saved CSV: {path_csv} ({len(df)} rows)")
        
        # Save Parquet if path provided
        if path_parquet is not None:
            path_parquet.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, path_parquet)
            logger.info(f"Saved Parquet: {path_parquet}")
            
    except Exception as e:
        logger.error(f"Error saving table: {e}")