"""Plotting functions using matplotlib for weight-nutrition analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger


def create_teaser_plot(analysis_df: pd.DataFrame, output_path: Path, 
                      title: str = "Weekly Fiber vs Next-Week Weight Change") -> bool:
    """
    Create teaser plot showing fiber vs next-week weight change.
    If real data is present, use it; otherwise create synthetic data.
    Returns True if real data was used, False if synthetic.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if we have real data
    required_cols = ['fiber_g_total_week_mean', 'delta_kg_next']
    has_real_data = (
        not analysis_df.empty and 
        all(col in analysis_df.columns for col in required_cols) and
        analysis_df[required_cols].dropna().shape[0] > 5
    )
    
    if has_real_data:
        return _create_real_teaser_plot(analysis_df, output_path, title)
    else:
        return _create_synthetic_teaser_plot(output_path, title)


def _create_real_teaser_plot(analysis_df: pd.DataFrame, output_path: Path, title: str) -> bool:
    """Create teaser plot using real data."""
    try:
        # Filter to complete data
        plot_df = analysis_df.dropna(subset=['fiber_g_total_week_mean', 'delta_kg_next']).copy()
        
        if plot_df.empty:
            logger.warning("No complete data for real teaser plot")
            return _create_synthetic_teaser_plot(output_path, title)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Base scatter plot
        mask_base = pd.Series([True] * len(plot_df), index=plot_df.index)
        
        # Check for keyword columns and create masks
        if 'any_kw_beans' in plot_df.columns and 'any_kw_potato' in plot_df.columns:
            mask_base = (plot_df['any_kw_beans'] == 0) & (plot_df['any_kw_potato'] == 0)
        
        # Plot base points
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
        
        # Add correlation info to title
        from scipy import stats
        corr, p_val = stats.spearmanr(plot_df['fiber_g_total_week_mean'], plot_df['delta_kg_next'])
        title_with_stats = f"{title}\n(r = {corr:.3f}, p = {p_val:.3f}, n = {len(plot_df)})"
        
        ax.set_xlabel('Weekly Fiber (g)')
        ax.set_ylabel('Next Week Δkg')
        ax.set_title(title_with_stats)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created real data teaser plot: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating real teaser plot: {e}")
        return _create_synthetic_teaser_plot(output_path, title)


def _create_synthetic_teaser_plot(output_path: Path, title: str) -> bool:
    """Create synthetic teaser plot as fallback."""
    try:
        # Generate synthetic data
        np.random.seed(42)  # Reproducible
        n_points = 80
        
        # Synthetic fiber intake (realistic range)
        fiber_intake = np.random.gamma(2, 8) + np.random.normal(0, 2, n_points)
        fiber_intake = np.clip(fiber_intake, 5, 50)  # Reasonable range
        
        # Synthetic weight change with weak fiber relationship + noise
        weight_change = -0.02 * fiber_intake + np.random.normal(0, 0.3, n_points)
        
        # Add some extreme values for realism
        weight_change[::20] += np.random.normal(0, 0.5, len(weight_change[::20]))
        
        # Create categorical variables
        beans_weeks = np.random.binomial(1, 0.15, n_points)  # 15% beans weeks
        potato_weeks = np.random.binomial(1, 0.25, n_points)  # 25% potato weeks
        
        # Slight effect for beans/potato weeks
        weight_change[beans_weeks == 1] -= 0.1  # Slight benefit for beans
        weight_change[potato_weeks == 1] += 0.05  # Slight negative for potatoes
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Base points
        other_mask = (beans_weeks == 0) & (potato_weeks == 0)
        ax.scatter(fiber_intake[other_mask], weight_change[other_mask], 
                  alpha=0.6, s=40, label='Other weeks')
        
        # Beans weeks (triangles)
        if beans_weeks.sum() > 0:
            ax.scatter(fiber_intake[beans_weeks == 1], weight_change[beans_weeks == 1],
                      marker='^', s=60, alpha=0.8, label='Beans weeks')
        
        # Potato weeks (squares)
        if potato_weeks.sum() > 0:
            ax.scatter(fiber_intake[potato_weeks == 1], weight_change[potato_weeks == 1],
                      marker='s', s=60, alpha=0.8, label='Potato weeks')
        
        # Add OLS line
        X = sm.add_constant(fiber_intake)
        model = sm.OLS(weight_change, X).fit()
        
        fiber_range = np.linspace(fiber_intake.min(), fiber_intake.max(), 100)
        X_pred = sm.add_constant(fiber_range)
        y_pred = model.predict(X_pred)
        
        ax.plot(fiber_range, y_pred, 'r-', linewidth=2, alpha=0.8, label='OLS fit')
        
        # Add synthetic correlation info
        from scipy import stats
        corr, p_val = stats.spearmanr(fiber_intake, weight_change)
        title_with_stats = f"{title} - Synthetic Preview\n(r = {corr:.3f}, p = {p_val:.3f}, n = {n_points})"
        
        ax.set_xlabel('Weekly Fiber (g)')
        ax.set_ylabel('Next Week Δkg')
        ax.set_title(title_with_stats)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add watermark
        ax.text(0.02, 0.98, 'SYNTHETIC DATA', transform=ax.transAxes, 
                fontsize=12, alpha=0.7, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created synthetic teaser plot: {output_path}")
        return False
        
    except Exception as e:
        logger.error(f"Error creating synthetic teaser plot: {e}")
        return False


def create_subject_weight_plots(weight_trajectories: pd.DataFrame, output_dir: Path) -> int:
    """
    Create individual weight trajectory plots for each subject.
    Returns number of plots created.
    """
    if weight_trajectories.empty:
        logger.warning("No weight trajectories to plot")
        return 0
    
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_created = 0
    
    subjects = weight_trajectories['subject_id'].unique()
    logger.info(f"Creating weight plots for {len(subjects)} subjects")
    
    for subject_id in subjects:
        try:
            subj_data = weight_trajectories[weight_trajectories['subject_id'] == subject_id].copy()
            
            if len(subj_data) < 2:
                logger.debug(f"Insufficient data for subject {subject_id}")
                continue
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            
            # Plot data points
            ax.scatter(subj_data['week'], subj_data['value'], alpha=0.7, s=50, zorder=3)
            
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
            ax.plot(week_range, pred.predicted_mean, 'r-', linewidth=2, 
                   label='OLS fit', zorder=2)
            ax.fill_between(week_range, pred.conf_int()[:, 0], pred.conf_int()[:, 1], 
                           alpha=0.3, color='red', label='95% CI', zorder=1)
            
            # Add slope information
            slope = model.params[1]
            slope_ci = model.conf_int()[1]
            slope_text = f"Slope: {slope:.3f} kg/week\n95% CI: [{slope_ci[0]:.3f}, {slope_ci[1]:.3f}]"
            
            ax.text(0.02, 0.98, slope_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Week')
            ax.set_ylabel('Weight (kg)')
            ax.set_title(f'Weight Trajectory - Subject {subject_id}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Clean subject_id for filename
            safe_subject_id = str(subject_id).replace('/', '_').replace('\\', '_')
            filename = output_dir / f"subject_weight_{safe_subject_id}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            plots_created += 1
            
        except Exception as e:
            logger.warning(f"Error plotting subject {subject_id}: {e}")
            continue
    
    logger.info(f"Created {plots_created} subject weight plots in {output_dir}")
    return plots_created


def create_calories_multiples_plot(analysis_df: pd.DataFrame, output_path: Path, 
                                 max_subjects: int = 12) -> bool:
    """
    Create small multiples plot of calories vs weight change by subject.
    Returns True if plot was created successfully.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check for required columns
        required_cols = ['calories_kcal_total_week_mean', 'delta_kg', 'subject_id']
        if not all(col in analysis_df.columns for col in required_cols):
            logger.warning("Missing required columns for calories multiples plot")
            return False
        
        plot_df = analysis_df.dropna(subset=required_cols).copy()
        subjects = plot_df['subject_id'].unique()[:max_subjects]
        
        if len(subjects) == 0:
            logger.warning("No subjects with complete calories/delta data")
            return False
        
        # Setup subplot grid
        n_cols = min(4, len(subjects))
        n_rows = int(np.ceil(len(subjects) / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        
        # Handle single subplot case
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
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
                        pass  # Skip line if regression fails
            
            # Clean subject_id for display
            display_id = str(subject_id)[:10]  # Truncate if too long
            ax.set_title(f'{display_id}', fontsize=10)
            ax.set_xlabel('Weekly Calories', fontsize=8)
            ax.set_ylabel('Δkg', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
        
        # Hide empty subplots
        for i in range(len(subjects), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle('Weekly Calories vs Weight Change by Subject', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created calories multiples plot: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating calories multiples plot: {e}")
        return False


def create_all_plots(weight_trajectories: pd.DataFrame, analysis_df: pd.DataFrame, 
                    output_dir: Path) -> tuple[bool, int]:
    """
    Create all plots for the analysis.
    Returns tuple of (teaser_used_real_data, num_subject_plots_created).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create teaser plot
    teaser_path = output_dir / "teaser.png"
    teaser_real = create_teaser_plot(analysis_df, teaser_path)
    
    # Create subject weight plots
    num_subject_plots = create_subject_weight_plots(weight_trajectories, output_dir)
    
    # Create calories multiples plot
    calories_path = output_dir / "calories_vs_delta_multiples.png"
    create_calories_multiples_plot(analysis_df, calories_path)
    
    logger.info(f"Created all plots in {output_dir}")
    return teaser_real, num_subject_plots