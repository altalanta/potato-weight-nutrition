#!/usr/bin/env python3
"""
Create a real data teaser plot for the potato-weight-nutrition pipeline.
This script looks for any available data and creates a meaningful visualization.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


def find_and_load_data():
    """Try to find any available data sources."""
    
    # Check for the cleaned CSVs first
    clean_dir = Path("~/Downloads/Potato Raw Dato/_clean").expanduser()
    
    if clean_dir.exists():
        tidy_path = clean_dir / "Potato_tidy.csv"
        nutrition_path = clean_dir / "Potato_nutrition_rows.csv" 
        fiber_path = clean_dir / "Potato_fiber_daily.csv"
        
        data_found = {}
        
        if tidy_path.exists():
            try:
                data_found['tidy'] = pd.read_csv(tidy_path)
                print(f"Found tidy data: {len(data_found['tidy'])} rows")
            except:
                pass
                
        if nutrition_path.exists():
            try:
                data_found['nutrition'] = pd.read_csv(nutrition_path)
                data_found['nutrition']['date'] = pd.to_datetime(data_found['nutrition']['date'], errors='coerce')
                print(f"Found nutrition data: {len(data_found['nutrition'])} rows")
            except:
                pass
                
        if fiber_path.exists():
            try:
                data_found['fiber'] = pd.read_csv(fiber_path)
                data_found['fiber']['date'] = pd.to_datetime(data_found['fiber']['date'], errors='coerce')
                print(f"Found fiber data: {len(data_found['fiber'])} rows")
            except:
                pass
        
        return data_found
    
    return {}

def create_weight_trajectory_plot(tidy_data, output_path):
    """Create a weight trajectory plot from tidy data."""
    
    # Filter for weight data
    weight_data = tidy_data[tidy_data['metric'] == 'weight'].copy()
    weight_data['value'] = pd.to_numeric(weight_data['value'], errors='coerce')
    weight_data = weight_data.dropna(subset=['value'])
    
    if weight_data.empty:
        return False
    
    print(f"Creating weight trajectory plot with {len(weight_data)} observations")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot trajectories for each subject
    subjects = weight_data['subject_id'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(subjects), 10)))
    
    for i, subject_id in enumerate(subjects[:10]):  # Limit to 10 subjects for clarity
        subj_data = weight_data[weight_data['subject_id'] == subject_id].sort_values('week')
        
        if len(subj_data) >= 2:
            color = colors[i % len(colors)]
            
            # Plot the trajectory
            ax.plot(subj_data['week'], subj_data['value'], 'o-', 
                   color=color, alpha=0.7, linewidth=2, markersize=6,
                   label=f'Subject {subject_id}' if i < 5 else "")
            
            # Fit and plot trend line
            try:
                weeks = subj_data['week'].values
                weights = subj_data['value'].values
                
                X = sm.add_constant(weeks)
                model = sm.OLS(weights, X).fit()
                
                week_range = np.linspace(weeks.min(), weeks.max(), 50)
                X_pred = sm.add_constant(week_range)
                y_pred = model.predict(X_pred)
                
                ax.plot(week_range, y_pred, '--', color=color, alpha=0.5, linewidth=1)
                
            except:
                pass
    
    # Calculate overall statistics
    total_subjects = len(subjects)
    total_observations = len(weight_data)
    weight_range = (weight_data['value'].min(), weight_data['value'].max())
    week_range = (weight_data['week'].min(), weight_data['week'].max())
    
    ax.set_xlabel('Study Week', fontsize=12)
    ax.set_ylabel('Weight (kg)', fontsize=12)
    ax.set_title(f'Weight Trajectories - Potato Fiber Study\n'
                f'{total_subjects} subjects, {total_observations} observations, '
                f'weeks {week_range[0]}-{week_range[1]}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Legend for first few subjects only
    if len(subjects) <= 5:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.text(0.02, 0.98, f'Showing {min(len(subjects), 10)} of {len(subjects)} subjects', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Created weight trajectory plot: {output_path}")
    return True

def create_nutrition_overview_plot(nutrition_data, fiber_data, output_path):
    """Create a nutrition overview plot."""
    
    if nutrition_data is not None and not nutrition_data.empty:
        data_to_use = nutrition_data
        title_suffix = "Nutrition Data"
    elif fiber_data is not None and not fiber_data.empty:
        data_to_use = fiber_data
        title_suffix = "Daily Fiber Data"
    else:
        return False
    
    print(f"Creating nutrition plot with {len(data_to_use)} observations")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Fiber over time
    if 'fiber_g' in data_to_use.columns:
        fiber_col = 'fiber_g'
    elif 'fiber_g_total' in data_to_use.columns:
        fiber_col = 'fiber_g_total'
    else:
        fiber_col = None
    
    if fiber_col and 'date' in data_to_use.columns:
        fiber_data_clean = data_to_use.dropna(subset=[fiber_col, 'date'])
        if not fiber_data_clean.empty:
            ax1.scatter(fiber_data_clean['date'], fiber_data_clean[fiber_col], 
                       alpha=0.6, s=30)
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Fiber (g)')
            ax1.set_title('Fiber Intake Over Time')
            ax1.grid(True, alpha=0.3)
            
            # Add trend line
            try:
                days_numeric = (fiber_data_clean['date'] - fiber_data_clean['date'].min()).dt.days
                X = sm.add_constant(days_numeric)
                model = sm.OLS(fiber_data_clean[fiber_col], X).fit()
                
                day_range = np.linspace(days_numeric.min(), days_numeric.max(), 100)
                X_pred = sm.add_constant(day_range)
                y_pred = model.predict(X_pred)
                date_range = fiber_data_clean['date'].min() + pd.to_timedelta(day_range, unit='D')
                
                ax1.plot(date_range, y_pred, 'r-', linewidth=2, alpha=0.8)
            except:
                pass
    
    # Plot 2: Fiber distribution
    if fiber_col:
        fiber_values = data_to_use[fiber_col].dropna()
        if not fiber_values.empty:
            ax2.hist(fiber_values, bins=20, alpha=0.7, edgecolor='black')
            ax2.axvline(fiber_values.median(), color='red', linestyle='--', linewidth=2, 
                       label=f'Median: {fiber_values.median():.1f}g')
            ax2.set_xlabel('Fiber (g)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Fiber Intake Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    # Plot 3: Calories over time (if available)
    cal_cols = ['calories_kcal', 'calories_kcal_total']
    cal_col = None
    for col in cal_cols:
        if col in data_to_use.columns:
            cal_col = col
            break
    
    if cal_col and 'date' in data_to_use.columns:
        cal_data_clean = data_to_use.dropna(subset=[cal_col, 'date'])
        if not cal_data_clean.empty:
            ax3.scatter(cal_data_clean['date'], cal_data_clean[cal_col], 
                       alpha=0.6, s=30, color='orange')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Calories (kcal)')
            ax3.set_title('Calorie Intake Over Time')
            ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax4.axis('off')
    
    # Calculate summary stats
    n_subjects = data_to_use['subject_id'].nunique() if 'subject_id' in data_to_use.columns else 'N/A'
    n_observations = len(data_to_use)
    date_range_str = 'N/A'
    
    if 'date' in data_to_use.columns:
        dates = data_to_use['date'].dropna()
        if not dates.empty:
            date_range_str = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
    
    summary_text = f"""
Dataset Summary:
• Subjects: {n_subjects}
• Observations: {n_observations:,}
• Date range: {date_range_str}
"""
    
    if fiber_col:
        fiber_stats = data_to_use[fiber_col].describe()
        summary_text += f"""
Fiber Intake (g):
• Mean: {fiber_stats['mean']:.1f}
• Median: {fiber_stats['50%']:.1f}
• Range: {fiber_stats['min']:.1f} - {fiber_stats['max']:.1f}
"""
    
    if cal_col:
        cal_stats = data_to_use[cal_col].describe()
        summary_text += f"""
Calories (kcal):
• Mean: {cal_stats['mean']:.0f}
• Median: {cal_stats['50%']:.0f}
• Range: {cal_stats['min']:.0f} - {cal_stats['max']:.0f}
"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(f'Potato Fiber Study - {title_suffix} Overview', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Created nutrition overview plot: {output_path}")
    return True

def main():
    """Main function to create real data teaser plot."""
    
    # Find available data
    data = find_and_load_data()
    
    if not data:
        print("No real data found - keeping synthetic plot")
        return False
    
    output_path = Path("potato-weight-nutrition/reports/figs/teaser.png")
    
    # Try weight trajectory first (most interesting)
    if 'tidy' in data:
        if create_weight_trajectory_plot(data['tidy'], output_path):
            return True
    
    # Fall back to nutrition overview
    nutrition_data = data.get('nutrition')
    fiber_data = data.get('fiber')
    
    if create_nutrition_overview_plot(nutrition_data, fiber_data, output_path):
        return True
    
    print("Could not create real data plot")
    return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Real data teaser plot created successfully!")
    else:
        print("❌ Could not create real data plot")
        sys.exit(1)