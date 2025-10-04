"""Statistical modeling functions for weight-nutrition associations."""


import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger
from scipy import stats
from statsmodels.formula.api import ols

try:
    from statsmodels.regression.mixed_linear_model import MixedLM
    MIXED_EFFECTS_AVAILABLE = True
except ImportError:
    MIXED_EFFECTS_AVAILABLE = False
    logger.warning("Mixed effects models not available")


def fit_ols_clustered(analysis_df: pd.DataFrame) -> dict[str, str | float]:
    """
    Fit OLS model with cluster-robust standard errors by subject.
    Model: delta_kg_next ~ fiber_g_week_mean + calories_kcal_week_mean + kw_beans + kw_potato + C(subject_id)
    """
    results = {}
    
    # Filter to rows with complete data
    required_cols = ['delta_kg_next', 'subject_id']
    
    # Check which nutrition variables are available
    nutrition_vars = []
    if 'fiber_g_total_week_mean' in analysis_df.columns:
        nutrition_vars.append('fiber_g_total_week_mean')
        required_cols.append('fiber_g_total_week_mean')
    
    if 'calories_kcal_total_week_mean' in analysis_df.columns:
        nutrition_vars.append('calories_kcal_total_week_mean')
        required_cols.append('calories_kcal_total_week_mean')
    
    keyword_vars = []
    for kw in ['beans', 'potato']:
        col_name = f'any_kw_{kw}'
        if col_name in analysis_df.columns:
            keyword_vars.append(col_name)
    
    model_df = analysis_df.dropna(subset=required_cols).copy()
    
    if model_df.empty:
        logger.warning("No complete data for OLS modeling")
        return results
    
    logger.info(f"Fitting OLS with {len(model_df)} observations")
    
    try:
        # Build formula
        formula_parts = ['delta_kg_next ~'] + nutrition_vars + keyword_vars + ['C(subject_id)']
        formula = ' + '.join(formula_parts)
        
        logger.info(f"OLS formula: {formula}")
        
        # Fit model with cluster-robust standard errors
        ols_model = ols(formula, data=model_df).fit(
            cov_type='cluster', 
            cov_kwds={'groups': model_df['subject_id']}
        )
        
        # Extract fiber coefficient if available
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
        results['ols_rsquared'] = ols_model.rsquared
        results['ols_n_obs'] = int(ols_model.nobs)
        
    except Exception as e:
        logger.error(f"Error fitting OLS model: {e}")
    
    return results


def fit_within_subject_effects(analysis_df: pd.DataFrame) -> dict[str, float | list[float]]:
    """
    Fit within-subject fixed effects by demeaning variables.
    Returns distribution of within-subject coefficients.
    """
    results = {}
    
    required_cols = ['delta_kg_next', 'fiber_g_total_week_mean', 'calories_kcal_total_week_mean']
    model_df = analysis_df.dropna(subset=required_cols).copy()
    
    if model_df.empty:
        logger.warning("No complete data for within-subject analysis")
        return results
    
    within_subject_coefs = []
    
    for subject_id, group in model_df.groupby('subject_id'):
        if len(group) >= 3:  # Need at least 3 observations
            try:
                # Demean variables within subject
                group_demeaned = group.copy()
                for col in required_cols:
                    if col in group_demeaned.columns:
                        group_demeaned[col] = group_demeaned[col] - group_demeaned[col].mean()
                
                # Fit within-subject model (no intercept since demeaned)
                within_model = ols(
                    'delta_kg_next ~ fiber_g_total_week_mean + calories_kcal_total_week_mean - 1', 
                    data=group_demeaned
                ).fit()
                
                if 'fiber_g_total_week_mean' in within_model.params.index:
                    within_subject_coefs.append(within_model.params['fiber_g_total_week_mean'])
                    
            except Exception as e:
                logger.debug(f"Error in within-subject model for {subject_id}: {e}")
                continue
    
    if within_subject_coefs:
        results['within_subject_coefs'] = within_subject_coefs
        results['within_subject_mean'] = np.mean(within_subject_coefs)
        results['within_subject_std'] = np.std(within_subject_coefs)
        results['within_subject_n'] = len(within_subject_coefs)
        
        logger.info(f"Within-subject analysis: mean={np.mean(within_subject_coefs):.4f}, "
                   f"std={np.std(within_subject_coefs):.4f}, n={len(within_subject_coefs)}")
    
    return results


def fit_mixed_effects(analysis_df: pd.DataFrame) -> dict[str, str | float]:
    """
    Fit mixed effects model with random intercepts by subject.
    Handles convergence gracefully.
    """
    results = {}
    
    if not MIXED_EFFECTS_AVAILABLE:
        logger.warning("Mixed effects models not available")
        return results
    
    required_cols = ['delta_kg_next', 'fiber_g_total_week_mean', 'calories_kcal_total_week_mean', 'subject_id']
    model_df = analysis_df.dropna(subset=required_cols).copy()
    
    if model_df.empty or len(model_df['subject_id'].unique()) < 3:
        logger.warning("Insufficient data for mixed effects modeling")
        return results
    
    try:
        # Prepare data
        endog = model_df['delta_kg_next']
        exog = model_df[['fiber_g_total_week_mean', 'calories_kcal_total_week_mean']]
        exog = sm.add_constant(exog)
        groups = model_df['subject_id']
        
        # Fit mixed effects model
        mixed_model = MixedLM(endog, exog, groups=groups).fit(reml=False)
        
        # Extract results
        if 'fiber_g_total_week_mean' in mixed_model.params.index:
            fiber_coef = mixed_model.params['fiber_g_total_week_mean']
            fiber_se = mixed_model.bse['fiber_g_total_week_mean']
            fiber_pvalue = mixed_model.pvalues['fiber_g_total_week_mean']
            
            # Calculate 95% CI
            fiber_ci_lower = fiber_coef - 1.96 * fiber_se
            fiber_ci_upper = fiber_coef + 1.96 * fiber_se
            
            results['mixed_fiber_coef'] = fiber_coef
            results['mixed_fiber_ci_lower'] = fiber_ci_lower
            results['mixed_fiber_ci_upper'] = fiber_ci_upper
            results['mixed_fiber_pvalue'] = fiber_pvalue
            
            logger.info(f"Mixed effects fiber coefficient: {fiber_coef:.4f} "
                       f"(95% CI: {fiber_ci_lower:.4f}, {fiber_ci_upper:.4f})")
        
        results['mixed_summary'] = str(mixed_model.summary())
        results['mixed_converged'] = mixed_model.converged
        
    except Exception as e:
        logger.warning(f"Mixed effects model failed: {e}")
        results['mixed_error'] = str(e)
    
    return results


def compute_correlations(analysis_df: pd.DataFrame) -> dict[str, float]:
    """Compute Spearman correlations between fiber and weight changes."""
    results = {}
    
    required_cols = ['fiber_g_total_week_mean', 'delta_kg_next']
    corr_df = analysis_df.dropna(subset=required_cols)
    
    if len(corr_df) < 10:
        logger.warning("Insufficient data for correlation analysis")
        return results
    
    try:
        # Spearman correlation
        fiber_vals = corr_df['fiber_g_total_week_mean']
        delta_vals = corr_df['delta_kg_next']
        
        spearman_corr, spearman_p = stats.spearmanr(fiber_vals, delta_vals)
        
        results['spearman_correlation'] = spearman_corr
        results['spearman_p_value'] = spearman_p
        results['spearman_n'] = len(corr_df)
        
        logger.info(f"Spearman correlation (fiber, next-week Δkg): {spearman_corr:.3f} (p={spearman_p:.3f})")
        
        # Pearson correlation for comparison
        pearson_corr, pearson_p = stats.pearsonr(fiber_vals, delta_vals)
        results['pearson_correlation'] = pearson_corr
        results['pearson_p_value'] = pearson_p
        
    except Exception as e:
        logger.error(f"Error computing correlations: {e}")
    
    return results


def test_group_differences(analysis_df: pd.DataFrame) -> dict[str, float]:
    """
    Test differences between groups using Mann-Whitney U tests.
    Compare beans vs non-beans weeks, etc.
    """
    results = {}
    
    # Beans vs non-beans comparison
    if 'any_kw_beans' in analysis_df.columns and 'delta_kg_next' in analysis_df.columns:
        try:
            test_df = analysis_df.dropna(subset=['any_kw_beans', 'delta_kg_next'])
            
            beans_weeks = test_df[test_df['any_kw_beans'] == 1]['delta_kg_next']
            no_beans_weeks = test_df[test_df['any_kw_beans'] == 0]['delta_kg_next']
            
            if len(beans_weeks) > 0 and len(no_beans_weeks) > 0:
                mw_stat, mw_p = stats.mannwhitneyu(
                    beans_weeks, no_beans_weeks, alternative='two-sided'
                )
                
                results['beans_median_delta'] = beans_weeks.median()
                results['no_beans_median_delta'] = no_beans_weeks.median()
                results['beans_vs_no_beans_p'] = mw_p
                results['beans_n'] = len(beans_weeks)
                results['no_beans_n'] = len(no_beans_weeks)
                
                logger.info(f"Beans vs no-beans median Δkg: {beans_weeks.median():.3f} vs "
                           f"{no_beans_weeks.median():.3f} (p={mw_p:.3f})")
        except Exception as e:
            logger.error(f"Error in beans comparison: {e}")
    
    # Potato vs non-potato comparison
    if 'any_kw_potato' in analysis_df.columns and 'delta_kg_next' in analysis_df.columns:
        try:
            test_df = analysis_df.dropna(subset=['any_kw_potato', 'delta_kg_next'])
            
            potato_weeks = test_df[test_df['any_kw_potato'] == 1]['delta_kg_next']
            no_potato_weeks = test_df[test_df['any_kw_potato'] == 0]['delta_kg_next']
            
            if len(potato_weeks) > 0 and len(no_potato_weeks) > 0:
                mw_stat, mw_p = stats.mannwhitneyu(
                    potato_weeks, no_potato_weeks, alternative='two-sided'
                )
                
                results['potato_median_delta'] = potato_weeks.median()
                results['no_potato_median_delta'] = no_potato_weeks.median()
                results['potato_vs_no_potato_p'] = mw_p
                
                logger.info(f"Potato vs no-potato median Δkg: {potato_weeks.median():.3f} vs "
                           f"{no_potato_weeks.median():.3f} (p={mw_p:.3f})")
        except Exception as e:
            logger.error(f"Error in potato comparison: {e}")
    
    return results


def run_full_analysis(analysis_df: pd.DataFrame) -> dict[str, str | float | list[float]]:
    """
    Run complete statistical analysis pipeline.
    Returns combined results dictionary.
    """
    logger.info("Running full statistical analysis")
    
    all_results = {}
    
    # OLS with cluster-robust standard errors
    ols_results = fit_ols_clustered(analysis_df)
    all_results.update(ols_results)
    
    # Within-subject fixed effects
    within_results = fit_within_subject_effects(analysis_df)
    all_results.update(within_results)
    
    # Mixed effects (if available)
    mixed_results = fit_mixed_effects(analysis_df)
    all_results.update(mixed_results)
    
    # Correlations
    corr_results = compute_correlations(analysis_df)
    all_results.update(corr_results)
    
    # Group comparisons
    group_results = test_group_differences(analysis_df)
    all_results.update(group_results)
    
    # Summary metrics
    all_results['total_observations'] = len(analysis_df)
    all_results['total_subjects'] = analysis_df['subject_id'].nunique()
    
    logger.info("Statistical analysis completed")
    
    return all_results


def format_results_summary(results: dict[str, str | float | list[float]]) -> str:
    """Format statistical results into human-readable summary."""
    
    lines = [
        "POTATO WEIGHT-NUTRITION ANALYSIS - STATISTICAL RESULTS",
        "=" * 60,
        "",
        f"Dataset: {results.get('total_observations', 'N/A')} observations, "
        f"{results.get('total_subjects', 'N/A')} subjects",
        ""
    ]
    
    # Correlations
    if 'spearman_correlation' in results:
        lines.extend([
            "CORRELATIONS",
            "-" * 20,
            f"Spearman correlation (fiber, next-week Δkg): {results['spearman_correlation']:.4f}",
            f"  p-value: {results['spearman_p_value']:.4f}",
            f"  n: {results['spearman_n']}",
            ""
        ])
    
    # OLS Results
    if 'ols_fiber_coef' in results:
        lines.extend([
            "OLS MODEL (Cluster-Robust SEs)",
            "-" * 30,
            f"Fiber coefficient: {results['ols_fiber_coef']:.4f}",
            f"  95% CI: ({results['ols_fiber_ci_lower']:.4f}, {results['ols_fiber_ci_upper']:.4f})",
            f"  p-value: {results['ols_fiber_pvalue']:.4f}",
            f"  R-squared: {results.get('ols_rsquared', 'N/A'):.4f}",
            ""
        ])
    
    # Within-subject results
    if 'within_subject_mean' in results:
        lines.extend([
            "WITHIN-SUBJECT FIXED EFFECTS",
            "-" * 30,
            f"Mean fiber coefficient: {results['within_subject_mean']:.4f}",
            f"  Standard deviation: {results['within_subject_std']:.4f}",
            f"  Number of subjects: {results['within_subject_n']}",
            ""
        ])
    
    # Mixed effects results
    if 'mixed_fiber_coef' in results:
        lines.extend([
            "MIXED EFFECTS MODEL",
            "-" * 20,
            f"Fiber coefficient: {results['mixed_fiber_coef']:.4f}",
            f"  95% CI: ({results['mixed_fiber_ci_lower']:.4f}, {results['mixed_fiber_ci_upper']:.4f})",
            f"  p-value: {results['mixed_fiber_pvalue']:.4f}",
            f"  Converged: {results.get('mixed_converged', 'N/A')}",
            ""
        ])
    
    # Group comparisons
    if 'beans_vs_no_beans_p' in results:
        lines.extend([
            "GROUP COMPARISONS (Mann-Whitney U)",
            "-" * 35,
            "Beans vs no-beans weeks:",
            f"  Beans median Δkg: {results['beans_median_delta']:.4f} (n={results.get('beans_n', 'N/A')})",
            f"  No-beans median Δkg: {results['no_beans_median_delta']:.4f} (n={results.get('no_beans_n', 'N/A')})",
            f"  p-value: {results['beans_vs_no_beans_p']:.4f}",
            ""
        ])
    
    if 'potato_vs_no_potato_p' in results:
        lines.extend([
            "Potato vs no-potato weeks:",
            f"  Potato median Δkg: {results['potato_median_delta']:.4f}",
            f"  No-potato median Δkg: {results['no_potato_median_delta']:.4f}",
            f"  p-value: {results['potato_vs_no_potato_p']:.4f}",
            ""
        ])
    
    # Full model summaries
    if 'ols_summary' in results:
        lines.extend([
            "DETAILED OLS MODEL SUMMARY",
            "-" * 30,
            results['ols_summary'],
            ""
        ])
    
    if 'mixed_summary' in results:
        lines.extend([
            "DETAILED MIXED EFFECTS SUMMARY",
            "-" * 35,
            results['mixed_summary'],
            ""
        ])
    
    return "\n".join(lines)