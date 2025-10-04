#!/usr/bin/env python3
"""
Potato Fiber Analysis Script - Statistical Analysis of Weight Trajectories and Nutrition
Processes cleaned datasets from potato fiber study to analyze weight changes and associations
with nutrition patterns, especially fiber intake and food categories.
"""

import argparse
import logging
import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols

# Suppress common warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('potato_analysis.log')
        ]
    )
    return logging.getLogger(__name__)


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


def load_frames(base_path: Path, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three cleaned datasets from the _clean directory.
    Returns tuple of (tidy_df, nutrition_rows_df, fiber_daily_df).
    """
    clean_dir = base_path / "_clean"
    
    if not clean_dir.exists():
        raise FileNotFoundError(f"Clean directory not found: {clean_dir}")
    
    # Load datasets
    try:
        tidy_df = pd.read_csv(clean_dir / "Potato_tidy.csv")
        logger.info(f"Loaded tidy data: {len(tidy_df)} rows, {len(tidy_df.columns)} columns")
        
        nutrition_rows_df = pd.read_csv(clean_dir / "Potato_nutrition_rows.csv")
        logger.info(f"Loaded nutrition rows: {len(nutrition_rows_df)} rows, {len(nutrition_rows_df.columns)} columns")
        
        fiber_daily_df = pd.read_csv(clean_dir / "Potato_fiber_daily.csv")
        logger.info(f"Loaded daily fiber: {len(fiber_daily_df)} rows, {len(fiber_daily_df.columns)} columns")
        
    except FileNotFoundError as e:
        logger.error(f"Required data file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data files: {e}")
        raise
    
    return tidy_df, nutrition_rows_df, fiber_daily_df


def prepare_weight(tidy_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Process weight data to create trajectories with deltas and slopes.
    Returns DataFrame with weight trajectories per subject.
    """
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
            
            # Calculate deltas and percentage changes
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


def prepare_daily_nutrition(nutrition_rows_df: pd.DataFrame, fiber_daily_df: pd.DataFrame, 
                          weight_trajectories: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Process daily nutrition data and align with weight observation periods.
    Returns DataFrame with daily nutrition features.
    """
    # Parse dates
    fiber_daily = fiber_daily_df.copy()
    nutrition_rows = nutrition_rows_df.copy()
    
    # Convert date columns to datetime
    fiber_daily['date'] = pd.to_datetime(fiber_daily['date'], errors='coerce')
    nutrition_rows['date'] = pd.to_datetime(nutrition_rows['date'], errors='coerce')
    
    # Remove rows with invalid dates
    fiber_daily = fiber_daily.dropna(subset=['date'])
    nutrition_rows = nutrition_rows.dropna(subset=['date'])
    
    logger.info(f"Processing daily nutrition for {fiber_daily['subject_id'].nunique()} subjects")
    
    # Build calendar for each subject covering weight observation period
    subject_calendars = []
    
    for subject_id in weight_trajectories['subject_id'].unique():
        # Get date range for this subject from nutrition data
        subj_fiber = fiber_daily[fiber_daily['subject_id'] == subject_id]
        subj_nutrition = nutrition_rows[nutrition_rows['subject_id'] == subject_id]
        
        if subj_fiber.empty and subj_nutrition.empty:
            continue
            
        # Find date range
        all_dates = []
        if not subj_fiber.empty:
            all_dates.extend(subj_fiber['date'].tolist())
        if not subj_nutrition.empty:
            all_dates.extend(subj_nutrition['date'].tolist())
            
        if not all_dates:
            continue
            
        min_date = min(all_dates)
        max_date = max(all_dates)
        
        # Create calendar
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        for date in date_range:
            subject_calendars.append({
                'subject_id': subject_id,
                'date': date
            })
    
    calendar_df = pd.DataFrame(subject_calendars)
    
    # Join daily fiber data onto calendar (left join to preserve gaps)
    daily_nutrition = calendar_df.merge(
        fiber_daily[['subject_id', 'date', 'fiber_g_total', 'calories_kcal_total', 
                    'carb_g_total', 'protein_g_total', 'fat_g_total', 'n_entries']],
        on=['subject_id', 'date'],
        how='left'
    )
    
    logger.info(f"Created daily nutrition calendar with {len(daily_nutrition)} subject-days")
    
    return daily_nutrition


def build_keywords(nutrition_rows_df: pd.DataFrame, daily_nutrition_df: pd.DataFrame, 
                  logger: logging.Logger) -> pd.DataFrame:
    """
    Extract food keywords from nutrition rows and create daily boolean flags.
    Returns daily_nutrition_df with keyword columns added.
    """
    nutrition_rows = nutrition_rows_df.copy()
    nutrition_rows['date'] = pd.to_datetime(nutrition_rows['date'], errors='coerce')
    nutrition_rows = nutrition_rows.dropna(subset=['date'])
    
    # Initialize keyword columns
    daily_with_keywords = daily_nutrition_df.copy()
    
    for keyword in KEYWORDS.keys():
        daily_with_keywords[f'kw_{keyword}'] = 0
    
    logger.info(f"Extracting keywords: {list(KEYWORDS.keys())}")
    
    # Process each subject-date
    for subject_id in daily_with_keywords['subject_id'].unique():
        subj_nutrition = nutrition_rows[nutrition_rows['subject_id'] == subject_id]
        
        if subj_nutrition.empty:
            continue
            
        for date in daily_with_keywords[daily_with_keywords['subject_id'] == subject_id]['date'].unique():
            day_foods = subj_nutrition[subj_nutrition['date'] == date]['food']
            
            if day_foods.empty:
                continue
            
            # Combine all food entries for this day
            day_text = ' '.join(day_foods.fillna('').astype(str).str.lower())
            
            # Check each keyword pattern
            for keyword, pattern in KEYWORDS.items():
                if re.search(pattern, day_text, re.IGNORECASE):
                    mask = (daily_with_keywords['subject_id'] == subject_id) & \
                           (daily_with_keywords['date'] == date)
                    daily_with_keywords.loc[mask, f'kw_{keyword}'] = 1
    
    # Log keyword prevalence
    keyword_stats = {}
    for keyword in KEYWORDS.keys():
        col_name = f'kw_{keyword}'
        if col_name in daily_with_keywords.columns:
            total_days = len(daily_with_keywords)
            positive_days = daily_with_keywords[col_name].sum()
            keyword_stats[keyword] = f"{positive_days}/{total_days} days ({100*positive_days/total_days:.1f}%)"
    
    logger.info(f"Keyword prevalence: {keyword_stats}")
    
    return daily_with_keywords


def aggregate_weekly(daily_with_keywords: pd.DataFrame, weight_trajectories: pd.DataFrame, 
                    logger: logging.Logger) -> pd.DataFrame:
    """
    Convert daily features to weekly aggregates matching weight observation weeks.
    Returns DataFrame with weekly nutrition features.
    """
    daily_df = daily_with_keywords.copy()
    
    # Create baseline date mapping for each subject
    baseline_dates = {}
    
    for subject_id in weight_trajectories['subject_id'].unique():
        subj_daily = daily_df[daily_df['subject_id'] == subject_id]
        if not subj_daily.empty:
            baseline_dates[subject_id] = subj_daily['date'].min()
    
    # Calculate week indices for daily data
    daily_df['week_index'] = None
    
    for subject_id, baseline_date in baseline_dates.items():
        mask = daily_df['subject_id'] == subject_id
        daily_df.loc[mask, 'week_index'] = ((daily_df.loc[mask, 'date'] - baseline_date).dt.days // 7).astype(int)
    
    # Aggregate by subject and week_index
    agg_dict = {
        'fiber_g_total': ['mean', 'sum'],
        'calories_kcal_total': ['mean', 'sum'],
        'carb_g_total': ['mean', 'sum'],
        'protein_g_total': ['mean', 'sum'],
        'fat_g_total': ['mean', 'sum'],
        'n_entries': 'sum'
    }
    
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
    
    logger.info(f"Created weekly aggregates for {weekly_agg['subject_id'].nunique()} subjects, "
               f"{len(weekly_agg)} subject-weeks")
    
    return weekly_agg


def align_weeks(weight_trajectories: pd.DataFrame, weekly_nutrition: pd.DataFrame,
               logger: logging.Logger) -> pd.DataFrame:
    """
    Merge weight trajectories with weekly nutrition features by subject and week.
    Returns merged analysis DataFrame.
    """
    weight_df = weight_trajectories.copy()
    nutrition_df = weekly_nutrition.copy()
    
    # Rename week column in nutrition to match weight
    nutrition_df = nutrition_df.rename(columns={'week_index': 'week'})
    
    # Merge on subject_id and week
    analysis_df = weight_df.merge(
        nutrition_df,
        on=['subject_id', 'week'],
        how='left'  # Keep all weight observations, even without nutrition
    )
    
    # Add lead variables for modeling next-week changes
    analysis_df = analysis_df.sort_values(['subject_id', 'week'])
    analysis_df['delta_kg_next'] = analysis_df.groupby('subject_id')['delta_kg'].shift(-1)
    
    logger.info(f"Merged analysis dataset: {len(analysis_df)} subject-weeks, "
               f"{analysis_df['subject_id'].nunique()} subjects")
    
    return analysis_df


def fit_models(analysis_df: pd.DataFrame, logger: logging.Logger) -> dict[str, str | float]:
    """
    Fit statistical models and compute correlations.
    Returns dictionary with model results.
    """
    results = {}
    
    # Filter to rows with both weight and fiber data for main analysis
    model_df = analysis_df.dropna(subset=['delta_kg_next', 'fiber_g_total_week_mean']).copy()
    
    if model_df.empty:
        logger.warning("No complete data for modeling")
        return results
    
    logger.info(f"Fitting models with {len(model_df)} observations")
    
    # 1. Simple Spearman correlation
    try:
        fiber_vals = model_df['fiber_g_total_week_mean']
        delta_next_vals = model_df['delta_kg_next']
        
        spearman_corr, spearman_p = stats.spearmanr(fiber_vals, delta_next_vals)
        results['spearman_correlation'] = spearman_corr
        results['spearman_p_value'] = spearman_p
        
        logger.info(f"Spearman correlation (fiber, next-week Δkg): {spearman_corr:.3f} (p={spearman_p:.3f})")
        
    except Exception as e:
        logger.warning(f"Error computing correlation: {e}")
    
    # 2. Overall OLS with fixed effects
    try:
        # Prepare formula
        formula_parts = ['delta_kg_next ~ fiber_g_total_week_mean + calories_kcal_total_week_mean']
        
        # Add keyword variables if they exist
        if 'any_kw_beans' in model_df.columns:
            formula_parts.append('any_kw_beans')
        if 'any_kw_potato' in model_df.columns:
            formula_parts.append('any_kw_potato')
        
        # Add subject fixed effects
        formula_parts.append('C(subject_id)')
        
        formula = ' + '.join(formula_parts)
        
        # Fit model
        ols_model = ols(formula, data=model_df).fit(cov_type='cluster', 
                                                   cov_kwds={'groups': model_df['subject_id']})
        
        # Extract fiber coefficient and CI
        if 'fiber_g_total_week_mean' in ols_model.params.index:
            fiber_coef = ols_model.params['fiber_g_total_week_mean']
            fiber_ci = ols_model.conf_int().loc['fiber_g_total_week_mean']
            
            results['ols_fiber_coef'] = fiber_coef
            results['ols_fiber_ci_lower'] = fiber_ci[0]
            results['ols_fiber_ci_upper'] = fiber_ci[1]
            results['ols_fiber_pvalue'] = ols_model.pvalues['fiber_g_total_week_mean']
            
            logger.info(f"OLS fiber coefficient: {fiber_coef:.4f} "
                       f"(95% CI: {fiber_ci[0]:.4f}, {fiber_ci[1]:.4f})")
        
        results['ols_summary'] = str(ols_model.summary())
        
    except Exception as e:
        logger.warning(f"Error fitting OLS model: {e}")
    
    # 3. Within-subject (fixed effects by demeaning)
    try:
        within_subject_coefs = []
        
        for subject_id, group in model_df.groupby('subject_id'):
            if len(group) >= 3:  # Need at least 3 observations for meaningful regression
                # Demean variables
                group_demeaned = group.copy()
                for col in ['delta_kg_next', 'fiber_g_total_week_mean', 'calories_kcal_total_week_mean']:
                    if col in group_demeaned.columns:
                        group_demeaned[col] = group_demeaned[col] - group_demeaned[col].mean()
                
                # Fit within-subject model
                try:
                    within_model = ols('delta_kg_next ~ fiber_g_total_week_mean + calories_kcal_total_week_mean - 1', 
                                     data=group_demeaned).fit()
                    
                    if 'fiber_g_total_week_mean' in within_model.params.index:
                        within_subject_coefs.append(within_model.params['fiber_g_total_week_mean'])
                        
                except:
                    continue
        
        if within_subject_coefs:
            results['within_subject_coefs'] = within_subject_coefs
            results['within_subject_mean'] = np.mean(within_subject_coefs)
            results['within_subject_std'] = np.std(within_subject_coefs)
            
            logger.info(f"Within-subject fiber coefficients: mean={np.mean(within_subject_coefs):.4f}, "
                       f"std={np.std(within_subject_coefs):.4f}, n={len(within_subject_coefs)}")
        
    except Exception as e:
        logger.warning(f"Error in within-subject analysis: {e}")
    
    # 4. Beans vs non-beans comparison (Mann-Whitney test)
    try:
        if 'any_kw_beans' in model_df.columns:
            beans_weeks = model_df[model_df['any_kw_beans'] == 1]['delta_kg_next'].dropna()
            no_beans_weeks = model_df[model_df['any_kw_beans'] == 0]['delta_kg_next'].dropna()
            
            if len(beans_weeks) > 0 and len(no_beans_weeks) > 0:
                mw_stat, mw_p = stats.mannwhitneyu(beans_weeks, no_beans_weeks, alternative='two-sided')
                
                results['beans_median_delta'] = beans_weeks.median()
                results['no_beans_median_delta'] = no_beans_weeks.median()
                results['beans_vs_no_beans_p'] = mw_p
                
                logger.info(f"Beans vs no-beans median Δkg: {beans_weeks.median():.3f} vs "
                           f"{no_beans_weeks.median():.3f} (p={mw_p:.3f})")
        
    except Exception as e:
        logger.warning(f"Error in beans comparison: {e}")
    
    return results


def compute_descriptive_stats(analysis_df: pd.DataFrame, logger: logging.Logger) -> dict[str, pd.DataFrame]:
    """
    Compute descriptive statistics tables as specified in requirements.
    Returns dictionary of summary tables.
    """
    stats_tables = {}
    
    try:
        # 1. Weight change at specific time points
        weight_changes = []
        
        for subject_id in analysis_df['subject_id'].unique():
            subj_data = analysis_df[analysis_df['subject_id'] == subject_id].copy()
            
            record = {'subject_id': subject_id}
            
            # Weight changes at weeks 4, 8, 12, and 60
            for week in [4, 8, 12, 60]:
                week_data = subj_data[subj_data['week'] == week]
                if not week_data.empty:
                    record[f'delta_kg_week_{week}'] = week_data['delta_kg'].iloc[0]
                    record[f'pct_change_week_{week}'] = week_data['pct_change'].iloc[0]
                else:
                    record[f'delta_kg_week_{week}'] = np.nan
                    record[f'pct_change_week_{week}'] = np.nan
            
            weight_changes.append(record)
        
        weight_summary = pd.DataFrame(weight_changes)
        
        # Overall summary statistics
        weight_summary_stats = weight_summary.describe()
        stats_tables['weight_changes_by_subject'] = weight_summary
        stats_tables['weight_changes_summary'] = weight_summary_stats
        
        logger.info("Computed weight change statistics")
        
        # 2. Nutrition patterns: first 4 weeks vs later weeks
        nutrition_periods = []
        
        for subject_id in analysis_df['subject_id'].unique():
            subj_data = analysis_df[analysis_df['subject_id'] == subject_id].copy()
            
            # First 4 weeks (0-3)
            early_weeks = subj_data[subj_data['week'] <= 3]
            # Later weeks (4+)
            later_weeks = subj_data[subj_data['week'] >= 4]
            
            record = {'subject_id': subject_id}
            
            # Early period stats
            if not early_weeks.empty:
                record['fiber_g_early_mean'] = early_weeks['fiber_g_total_week_mean'].mean()
                record['calories_early_mean'] = early_weeks['calories_kcal_total_week_mean'].mean()
                record['n_early_weeks'] = len(early_weeks)
            else:
                record['fiber_g_early_mean'] = np.nan
                record['calories_early_mean'] = np.nan
                record['n_early_weeks'] = 0
            
            # Later period stats
            if not later_weeks.empty:
                record['fiber_g_later_mean'] = later_weeks['fiber_g_total_week_mean'].mean()
                record['calories_later_mean'] = later_weeks['calories_kcal_total_week_mean'].mean()
                record['n_later_weeks'] = len(later_weeks)
            else:
                record['fiber_g_later_mean'] = np.nan
                record['calories_later_mean'] = np.nan
                record['n_later_weeks'] = 0
            
            nutrition_periods.append(record)
        
        nutrition_summary = pd.DataFrame(nutrition_periods)
        stats_tables['nutrition_early_vs_later'] = nutrition_summary
        
        # 3. Cross-tabulation of keyword presence vs weight change
        crosstab_data = []
        
        # Only look at weeks with complete data
        complete_data = analysis_df.dropna(subset=['delta_kg']).copy()
        
        if not complete_data.empty:
            # Median weight change overall
            median_delta = complete_data['delta_kg'].median()
            
            for keyword in ['beans', 'potato']:
                kw_col = f'any_kw_{keyword}'
                if kw_col in complete_data.columns:
                    # Weeks with keyword
                    kw_weeks = complete_data[complete_data[kw_col] == 1]
                    no_kw_weeks = complete_data[complete_data[kw_col] == 0]
                    
                    crosstab_data.append({
                        'keyword': keyword,
                        'n_weeks_with_keyword': len(kw_weeks),
                        'n_weeks_without_keyword': len(no_kw_weeks),
                        'median_delta_with_keyword': kw_weeks['delta_kg'].median() if len(kw_weeks) > 0 else np.nan,
                        'median_delta_without_keyword': no_kw_weeks['delta_kg'].median() if len(no_kw_weeks) > 0 else np.nan,
                        'overall_median_delta': median_delta
                    })
        
        if crosstab_data:
            crosstab_df = pd.DataFrame(crosstab_data)
            stats_tables['keyword_crosstab'] = crosstab_df
            logger.info("Computed keyword cross-tabulations")
        
    except Exception as e:
        logger.warning(f"Error computing descriptive statistics: {e}")
    
    return stats_tables


def make_plots(analysis_df: pd.DataFrame, weight_trajectories: pd.DataFrame, 
              figs_dir: Path, logger: logging.Logger) -> None:
    """
    Create matplotlib plots for the analysis.
    """
    logger.info("Creating plots...")
    
    # 1. Individual subject weight trajectories with OLS lines
    subjects = weight_trajectories['subject_id'].unique()
    
    for subject_id in subjects:
        try:
            subj_data = weight_trajectories[weight_trajectories['subject_id'] == subject_id]
            
            if len(subj_data) < 2:
                continue
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # Plot data points
            ax.scatter(subj_data['week'], subj_data['value'], alpha=0.7, s=50)
            
            # Fit and plot OLS line with CI
            weeks = subj_data['week'].values
            weights = subj_data['value'].values
            
            X = sm.add_constant(weeks)
            model = sm.OLS(weights, X).fit()
            
            # Prediction with CI
            week_range = np.linspace(weeks.min(), weeks.max(), 100)
            X_pred = sm.add_constant(week_range)
            pred = model.get_prediction(X_pred)
            
            # Plot line and CI
            ax.plot(week_range, pred.predicted_mean, 'r-', linewidth=2, label='OLS fit')
            ax.fill_between(week_range, pred.conf_int()[:, 0], pred.conf_int()[:, 1], 
                           alpha=0.3, color='red', label='95% CI')
            
            ax.set_xlabel('Week')
            ax.set_ylabel('Weight (kg)')
            ax.set_title(f'Weight Trajectory - Subject {subject_id}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Save plot
            filename = figs_dir / f"subject_weight_{subject_id}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error plotting subject {subject_id}: {e}")
            continue
    
    # 2. Fiber vs next-week delta scatter
    try:
        plot_df = analysis_df.dropna(subset=['fiber_g_total_week_mean', 'delta_kg_next']).copy()
        
        if not plot_df.empty:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Base scatter plot
            mask_base = True
            if 'any_kw_beans' in plot_df.columns and 'any_kw_potato' in plot_df.columns:
                mask_base = (plot_df['any_kw_beans'] == 0) & (plot_df['any_kw_potato'] == 0)
            
            ax.scatter(plot_df.loc[mask_base, 'fiber_g_total_week_mean'], 
                      plot_df.loc[mask_base, 'delta_kg_next'], 
                      alpha=0.6, s=40, label='Other weeks')
            
            # Beans weeks (triangles)
            if 'any_kw_beans' in plot_df.columns:
                beans_mask = plot_df['any_kw_beans'] == 1
                if beans_mask.sum() > 0:
                    ax.scatter(plot_df.loc[beans_mask, 'fiber_g_total_week_mean'],
                              plot_df.loc[beans_mask, 'delta_kg_next'],
                              marker='^', s=60, alpha=0.8, label='Beans weeks')
            
            # Potato weeks (squares)
            if 'any_kw_potato' in plot_df.columns:
                potato_mask = plot_df['any_kw_potato'] == 1
                if potato_mask.sum() > 0:
                    ax.scatter(plot_df.loc[potato_mask, 'fiber_g_total_week_mean'],
                              plot_df.loc[potato_mask, 'delta_kg_next'],
                              marker='s', s=60, alpha=0.8, label='Potato weeks')
            
            # Add OLS line
            X = sm.add_constant(plot_df['fiber_g_total_week_mean'])
            model = sm.OLS(plot_df['delta_kg_next'], X).fit()
            
            fiber_range = np.linspace(plot_df['fiber_g_total_week_mean'].min(),
                                    plot_df['fiber_g_total_week_mean'].max(), 100)
            X_pred = sm.add_constant(fiber_range)
            y_pred = model.predict(X_pred)
            
            ax.plot(fiber_range, y_pred, 'r-', linewidth=2, alpha=0.8, label='OLS fit')
            
            ax.set_xlabel('Weekly Fiber (g)')
            ax.set_ylabel('Next Week Δkg')
            ax.set_title('Weekly Fiber vs Next-Week Weight Change')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.savefig(figs_dir / 'fiber_vs_next_delta.png', dpi=150, bbox_inches='tight')
            plt.close()
        
    except Exception as e:
        logger.warning(f"Error creating fiber scatter plot: {e}")
    
    # 3. Small multiples: calories vs delta_kg by subject
    try:
        plot_df = analysis_df.dropna(subset=['calories_kcal_total_week_mean', 'delta_kg']).copy()
        subjects = plot_df['subject_id'].unique()[:12]  # Max 12 subjects
        
        if len(subjects) > 0:
            n_cols = min(4, len(subjects))
            n_rows = int(np.ceil(len(subjects) / n_cols))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            for i, subject_id in enumerate(subjects):
                row = i // n_cols
                col = i % n_cols
                ax = axes[row, col]
                
                subj_data = plot_df[plot_df['subject_id'] == subject_id]
                
                if len(subj_data) > 1:
                    ax.scatter(subj_data['calories_kcal_total_week_mean'], subj_data['delta_kg'],
                              alpha=0.7, s=30)
                    
                    # Fit line if enough points
                    if len(subj_data) >= 3:
                        X = sm.add_constant(subj_data['calories_kcal_total_week_mean'])
                        try:
                            model = sm.OLS(subj_data['delta_kg'], X).fit()
                            cal_range = np.linspace(subj_data['calories_kcal_total_week_mean'].min(),
                                                  subj_data['calories_kcal_total_week_mean'].max(), 50)
                            X_pred = sm.add_constant(cal_range)
                            y_pred = model.predict(X_pred)
                            ax.plot(cal_range, y_pred, 'r-', linewidth=1, alpha=0.8)
                        except:
                            pass
                
                ax.set_title(f'{subject_id}', fontsize=10)
                ax.set_xlabel('Weekly Calories', fontsize=8)
                ax.set_ylabel('Δkg', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
            
            # Hide empty subplots
            for i in range(len(subjects), n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(figs_dir / 'calories_vs_delta_multiples.png', dpi=150, bbox_inches='tight')
            plt.close()
        
    except Exception as e:
        logger.warning(f"Error creating calories multiples plot: {e}")
    
    logger.info(f"Plots saved to {figs_dir}")


def save_artifacts(weight_trajectories: pd.DataFrame, weekly_nutrition: pd.DataFrame,
                  analysis_df: pd.DataFrame, model_results: dict, analysis_dir: Path,
                  logger: logging.Logger) -> None:
    """
    Save all analysis outputs to CSV/Parquet files and text summaries.
    """
    logger.info("Saving analysis artifacts...")
    
    # Save main datasets
    datasets = {
        'weight_trajectories': weight_trajectories,
        'nutrition_weekly': weekly_nutrition,
        'analysis_df': analysis_df
    }
    
    for name, df in datasets.items():
        if not df.empty:
            # CSV
            csv_path = analysis_dir / f"{name}.csv"
            df.to_csv(csv_path, index=False)
            
            # Parquet
            parquet_path = analysis_dir / f"{name}.parquet"
            table = pa.Table.from_pandas(df)
            pq.write_table(table, parquet_path)
            
            logger.info(f"Saved {name}: {len(df)} rows to {csv_path.name} and {parquet_path.name}")
    
    # Save model results
    results_path = analysis_dir / "ols_results.txt"
    with open(results_path, 'w') as f:
        f.write("POTATO FIBER ANALYSIS - MODEL RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        if 'spearman_correlation' in model_results:
            f.write("Spearman Correlation (Fiber vs Next-Week Δkg):\n")
            f.write(f"  r = {model_results['spearman_correlation']:.4f}\n")
            f.write(f"  p = {model_results['spearman_p_value']:.4f}\n\n")
        
        if 'ols_fiber_coef' in model_results:
            f.write("Overall OLS Model - Fiber Coefficient:\n")
            f.write(f"  β = {model_results['ols_fiber_coef']:.4f}\n")
            f.write(f"  95% CI: ({model_results['ols_fiber_ci_lower']:.4f}, {model_results['ols_fiber_ci_upper']:.4f})\n")
            f.write(f"  p = {model_results['ols_fiber_pvalue']:.4f}\n\n")
        
        if 'within_subject_mean' in model_results:
            f.write("Within-Subject Analysis:\n")
            f.write(f"  Mean coefficient = {model_results['within_subject_mean']:.4f}\n")
            f.write(f"  Std deviation = {model_results['within_subject_std']:.4f}\n")
            f.write(f"  N subjects = {len(model_results['within_subject_coefs'])}\n\n")
        
        if 'beans_vs_no_beans_p' in model_results:
            f.write("Beans vs No-Beans Comparison (Mann-Whitney):\n")
            f.write(f"  Beans weeks median Δkg = {model_results['beans_median_delta']:.4f}\n")
            f.write(f"  No-beans weeks median Δkg = {model_results['no_beans_median_delta']:.4f}\n")
            f.write(f"  p-value = {model_results['beans_vs_no_beans_p']:.4f}\n\n")
        
        if 'ols_summary' in model_results:
            f.write("FULL OLS MODEL SUMMARY:\n")
            f.write("-" * 30 + "\n")
            f.write(model_results['ols_summary'])
    
    logger.info(f"Model results saved to {results_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze potato fiber study data')
    parser.add_argument('--base', default='~/Downloads/Potato Raw Dato',
                       help='Base directory (default: ~/Downloads/Potato Raw Dato)')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    base_path = Path(args.base).expanduser()
    
    if not base_path.exists():
        logger.error(f"Base path does not exist: {base_path}")
        sys.exit(1)
    
    logger.info(f"Starting analysis of {base_path}")
    
    # Create output directories
    analysis_dir = base_path / "_clean" / "analysis"
    figs_dir = base_path / "_clean" / "figs"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        logger.info("Loading datasets...")
        tidy_df, nutrition_rows_df, fiber_daily_df = load_frames(base_path, logger)
        
        # Process weight trajectories
        logger.info("Processing weight trajectories...")
        weight_trajectories = prepare_weight(tidy_df, logger)
        
        if weight_trajectories.empty:
            logger.error("No weight data processed successfully")
            return
        
        # Process daily nutrition
        logger.info("Processing daily nutrition...")
        daily_nutrition = prepare_daily_nutrition(nutrition_rows_df, fiber_daily_df, 
                                                 weight_trajectories, logger)
        
        # Extract keywords
        logger.info("Extracting food keywords...")
        daily_with_keywords = build_keywords(nutrition_rows_df, daily_nutrition, logger)
        
        # Weekly aggregation
        logger.info("Aggregating weekly nutrition features...")
        weekly_nutrition = aggregate_weekly(daily_with_keywords, weight_trajectories, logger)
        
        # Align and merge
        logger.info("Merging weight and nutrition data...")
        analysis_df = align_weeks(weight_trajectories, weekly_nutrition, logger)
        
        # Descriptive statistics
        logger.info("Computing descriptive statistics...")
        descriptive_stats = compute_descriptive_stats(analysis_df, logger)
        
        # Statistical analysis
        logger.info("Fitting statistical models...")
        model_results = fit_models(analysis_df, logger)
        
        # Create plots
        logger.info("Creating plots...")
        make_plots(analysis_df, weight_trajectories, figs_dir, logger)
        
        # Save outputs
        logger.info("Saving analysis artifacts...")
        save_artifacts(weight_trajectories, weekly_nutrition, analysis_df, 
                      model_results, analysis_dir, logger)
        
        # Save descriptive statistics
        for name, df in descriptive_stats.items():
            if not df.empty:
                csv_path = analysis_dir / f"descriptive_{name}.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved descriptive stats: {name} to {csv_path.name}")
        
        # Print summary results
        print("\n" + "="*60)
        print("POTATO FIBER ANALYSIS - SUMMARY RESULTS")
        print("="*60)
        
        n_subjects = analysis_df['subject_id'].nunique()
        n_subject_weeks = len(analysis_df)
        
        print(f"Subjects: {n_subjects}")
        print(f"Subject-weeks: {n_subject_weeks}")
        
        if 'spearman_correlation' in model_results:
            print(f"Correlation (fiber, next-week Δkg): {model_results['spearman_correlation']:.3f}")
        
        if 'ols_fiber_coef' in model_results:
            ci_lower = model_results['ols_fiber_ci_lower']
            ci_upper = model_results['ols_fiber_ci_upper']
            print(f"OLS fiber coefficient: {model_results['ols_fiber_coef']:.4f} "
                  f"(95% CI: {ci_lower:.4f}, {ci_upper:.4f})")
        
        if 'beans_vs_no_beans_p' in model_results:
            beans_median = model_results['beans_median_delta']
            no_beans_median = model_results['no_beans_median_delta']
            p_val = model_results['beans_vs_no_beans_p']
            print(f"Beans vs no-beans median Δkg: {beans_median:.3f} vs {no_beans_median:.3f} "
                  f"(Mann-Whitney p={p_val:.3f})")
        
        print(f"\nResults saved to: {analysis_dir}")
        print(f"Figures saved to: {figs_dir}")
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()