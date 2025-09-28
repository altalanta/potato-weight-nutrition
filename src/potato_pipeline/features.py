"""Feature engineering functions for weight and nutrition data."""

import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger

# Food keyword patterns for extraction
KEYWORDS = {
    "potato": r"\b(potato|potatoes|russet|yukon|red potato|baked potato|mashed)\b",
    "beans": r"\b(beans?|black bean|pinto|navy|kidney|garbanzo|chickpea|lentils?)\b",
    "rice": r"\b(rice|brown rice|white rice|wild rice|jasmine|basmati)\b",
    "oats": r"\b(oats?|oatmeal|rolled oats|steel cut|porridge)\b",
    "bread": r"\b(bread|toast|bagel|roll|bun|loaf|wheat bread|white bread)\b",
    "fruit": r"\b(fruit|apple|banana|orange|berry|berries|grape|melon|peach|pear)\b",
    "vegetables": r"\b(vegetables?|veggie|carrot|broccoli|spinach|lettuce|tomato|pepper)\b",
    "dairy": r"\b(dairy|milk|cheese|yogurt|butter|cream|cottage cheese)\b",
    "meat": r"\b(meat|beef|chicken|pork|turkey|fish|salmon|tuna|steak|ham)\b",
    "egg": r"\b(eggs?|scrambled|boiled|fried|omelet|egg white)\b",
    "nuts": r"\b(nuts?|almond|walnut|peanut|cashew|pecan|pistachio|seed)\b"
}


def prepare_weight(tidy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process weight data to create trajectories with deltas and slopes.
    Returns DataFrame with weight trajectories per subject.
    """
    if tidy_df.empty:
        logger.warning("Empty tidy data provided")
        return pd.DataFrame()
    
    # Filter for weight data only
    weight_df = tidy_df[tidy_df['metric'] == 'weight'].copy()
    
    if weight_df.empty:
        logger.warning("No weight data found")
        return pd.DataFrame()
    
    logger.info(f"Processing weight data for {weight_df['subject_id'].nunique()} subjects")
    
    # Include baseline (phase=start, week=0) and end measurements
    weight_clean = weight_df[
        ((weight_df['phase'] == 'start') & (weight_df['week'] == 0)) |
        (weight_df['phase'] == 'end')
    ].copy()
    
    # Convert value to numeric
    weight_clean['value'] = pd.to_numeric(weight_clean['value'], errors='coerce')
    weight_clean = weight_clean.dropna(subset=['value'])
    
    results = []
    
    for subject_id, group in weight_clean.groupby('subject_id'):
        try:
            # Sort by week
            group = group.sort_values('week')
            
            # Find baseline weight (week 0)
            baseline_rows = group[group['week'] == 0]
            if baseline_rows.empty:
                logger.warning(f"No baseline weight for subject {subject_id}")
                continue
                
            start_weight = baseline_rows['value'].iloc[0]
            
            # Calculate deltas and percentage changes for each observation
            for _, row in group.iterrows():
                delta_kg = row['value'] - start_weight
                pct_change = 100 * (row['value'] / start_weight - 1)
                
                results.append({
                    'subject_id': subject_id,
                    'week': row['week'],
                    'value': row['value'],
                    'start_weight': start_weight,
                    'delta_kg': delta_kg,
                    'pct_change': pct_change,
                    'phase': row['phase'],
                    'note': row['note']
                })
            
            # Calculate OLS slope for this subject (kg/week)
            if len(group) >= 2:
                weeks = group['week'].values
                weights = group['value'].values
                
                # Fit OLS
                X = sm.add_constant(weeks)
                model = sm.OLS(weights, X).fit()
                slope = model.params[1]  # kg/week
                slope_ci = model.conf_int()[1]  # 95% CI for slope
                
                # Add slope info to all rows for this subject
                for i, result in enumerate(results):
                    if result['subject_id'] == subject_id:
                        results[i]['slope_kg_per_week'] = slope
                        results[i]['slope_ci_lower'] = slope_ci[0]
                        results[i]['slope_ci_upper'] = slope_ci[1]
            
        except Exception as e:
            logger.warning(f"Error processing subject {subject_id}: {e}")
            continue
    
    weight_trajectories = pd.DataFrame(results)
    logger.info(f"Created weight trajectories for {weight_trajectories['subject_id'].nunique()} subjects")
    
    return weight_trajectories


def build_keywords(nutrition_rows_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract food keywords from nutrition rows and create daily boolean flags.
    Returns DataFrame with keyword columns added.
    """
    if nutrition_rows_df.empty:
        logger.warning("Empty nutrition rows provided")
        return pd.DataFrame()
    
    nutrition_rows = nutrition_rows_df.copy()
    
    # Ensure required columns exist
    if 'date' not in nutrition_rows.columns:
        logger.warning("No date column in nutrition rows")
        return nutrition_rows
    
    if 'food' not in nutrition_rows.columns:
        logger.warning("No food column in nutrition rows")
        return nutrition_rows
        
    nutrition_rows['date'] = pd.to_datetime(nutrition_rows['date'], errors='coerce')
    nutrition_rows = nutrition_rows.dropna(subset=['date'])
    
    if nutrition_rows.empty:
        logger.warning("No valid dates in nutrition rows")
        return pd.DataFrame()
    
    # Create daily aggregation with keyword flags
    daily_keywords = []
    
    # Group by subject and date
    for (subject_id, date), group in nutrition_rows.groupby(['subject_id', 'date']):
        record = {
            'subject_id': subject_id,
            'date': date
        }
        
        # Combine all food entries for this day
        day_foods = group['food'].fillna('').astype(str)
        day_text = ' '.join(day_foods).lower()
        
        # Check each keyword pattern
        for keyword, pattern in KEYWORDS.items():
            has_keyword = bool(re.search(pattern, day_text, re.IGNORECASE))
            record[f'kw_{keyword}'] = 1 if has_keyword else 0
        
        daily_keywords.append(record)
    
    daily_df = pd.DataFrame(daily_keywords)
    
    # Log keyword prevalence
    if not daily_df.empty:
        keyword_stats = {}
        for keyword in KEYWORDS.keys():
            col_name = f'kw_{keyword}'
            if col_name in daily_df.columns:
                total_days = len(daily_df)
                positive_days = daily_df[col_name].sum()
                keyword_stats[keyword] = f"{positive_days}/{total_days} days ({100*positive_days/total_days:.1f}%)"
        
        logger.info(f"Keyword prevalence: {keyword_stats}")
    
    return daily_df


def prepare_daily(nutrition_rows_df: pd.DataFrame, fiber_daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge nutrition rows with daily fiber data to create comprehensive daily features.
    """
    if fiber_daily_df.empty:
        logger.warning("Empty daily fiber data provided")
        return pd.DataFrame()
    
    # Get keyword flags from nutrition rows
    daily_keywords = build_keywords(nutrition_rows_df)
    
    # Start with fiber daily data
    daily_df = fiber_daily_df.copy()
    
    # Merge with keyword flags
    if not daily_keywords.empty:
        daily_df = daily_df.merge(
            daily_keywords,
            on=['subject_id', 'date'],
            how='left'
        )
        
        # Fill missing keyword flags with 0
        for keyword in KEYWORDS.keys():
            col_name = f'kw_{keyword}'
            if col_name in daily_df.columns:
                daily_df[col_name] = daily_df[col_name].fillna(0).astype(int)
    
    logger.info(f"Prepared daily data: {len(daily_df)} subject-days")
    return daily_df


def aggregate_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily features to weekly aggregates.
    Returns DataFrame with weekly nutrition features.
    """
    if daily_df.empty:
        logger.warning("Empty daily data provided")
        return pd.DataFrame()
    
    # Create week index for each subject (weeks since first observation)
    daily_df = daily_df.copy()
    daily_df['week_index'] = None
    
    for subject_id in daily_df['subject_id'].unique():
        mask = daily_df['subject_id'] == subject_id
        subj_data = daily_df.loc[mask]
        
        if subj_data.empty:
            continue
            
        # Calculate weeks since first date for this subject
        first_date = subj_data['date'].min()
        daily_df.loc[mask, 'week_index'] = ((subj_data['date'] - first_date).dt.days // 7).astype(int)
    
    # Aggregate by subject and week_index - only for columns that exist
    agg_dict = {}
    
    # Add nutrition columns if they exist
    nutrition_cols = [
        'fiber_g_total', 'calories_kcal_total', 'carb_g_total', 
        'protein_g_total', 'fat_g_total'
    ]
    for col in nutrition_cols:
        if col in daily_df.columns:
            agg_dict[col] = ['mean', 'sum']
    
    if 'n_entries' in daily_df.columns:
        agg_dict['n_entries'] = 'sum'
    
    # Add keyword aggregations
    for keyword in KEYWORDS.keys():
        col_name = f'kw_{keyword}'
        if col_name in daily_df.columns:
            agg_dict[col_name] = ['mean', 'max']  # share_of_days and any_day
    
    weekly_agg = daily_df.groupby(['subject_id', 'week_index']).agg(agg_dict).reset_index()
    
    # Flatten column names
    new_columns = ['subject_id', 'week_index']
    for col in weekly_agg.columns[2:]:
        if isinstance(col, tuple):
            base_name, agg_type = col
            if agg_type == 'mean' and base_name.startswith('kw_'):
                new_name = f"share_of_days_{base_name}"
            elif agg_type == 'max' and base_name.startswith('kw_'):
                new_name = f"any_{base_name}"
            elif agg_type == 'mean':
                new_name = f"{base_name}_week_mean"
            elif agg_type == 'sum':
                new_name = f"{base_name}_week_sum"
            else:
                new_name = f"{base_name}_{agg_type}"
            new_columns.append(new_name)
        else:
            new_columns.append(str(col))
    
    weekly_agg.columns = new_columns
    
    logger.info(f"Created weekly aggregates: {len(weekly_agg)} subject-weeks")
    return weekly_agg


def align_weeks(weight_trajectories: pd.DataFrame, weekly_nutrition: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Align weight trajectories with weekly nutrition features by subject and week.
    Returns tuple of (weight_trajectories, weekly_nutrition, merged_analysis_df).
    """
    if weight_trajectories.empty:
        logger.warning("Empty weight trajectories provided")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    weight_df = weight_trajectories.copy()
    nutrition_df = weekly_nutrition.copy() if not weekly_nutrition.empty else pd.DataFrame()
    
    # Rename week column in nutrition to match weight
    if not nutrition_df.empty:
        nutrition_df = nutrition_df.rename(columns={'week_index': 'week'})
        
        # Merge on subject_id and week
        analysis_df = weight_df.merge(
            nutrition_df,
            on=['subject_id', 'week'],
            how='left'  # Keep all weight observations, even without nutrition
        )
    else:
        analysis_df = weight_df.copy()
        logger.warning("No weekly nutrition data for merging")
    
    # Add lead variables for modeling next-week changes
    analysis_df = analysis_df.sort_values(['subject_id', 'week'])
    analysis_df['delta_kg_next'] = analysis_df.groupby('subject_id')['delta_kg'].shift(-1)
    
    logger.info(f"Merged analysis dataset: {len(analysis_df)} subject-weeks, "
               f"{analysis_df['subject_id'].nunique()} subjects")
    
    return weight_df, nutrition_df, analysis_df