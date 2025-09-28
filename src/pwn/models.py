"""Statistical models: OLS, Fixed Effects, and Mixed Effects models."""

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


def run_statistical_model(
    features_dir: Path,
    output_dir: Path,
    model_type: str = "ols",
    obs_enabled: bool = False
) -> Dict[str, Any]:
    """Run statistical models on processed features.
    
    Args:
        features_dir: Directory containing processed features
        output_dir: Directory to save model results
        model_type: Type of model to run ("ols", "fe", "mixed")
        obs_enabled: Enable observability and metrics collection
        
    Returns:
        Dictionary with model results and metrics
    """
    logger = logging.getLogger("pwn.models")
    logger.info(f"Running {model_type} statistical model")
    
    # Load analysis dataset
    analysis_path = features_dir / "analysis_df.csv"
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis dataset not found: {analysis_path}")
    
    df = pd.read_csv(analysis_path)
    logger.info(f"Loaded analysis dataset with {len(df)} rows")
    
    # Prepare output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if model_type == "ols":
        results = _run_ols_model(df, output_dir, obs_enabled)
    elif model_type == "fe":
        results = _run_fixed_effects_model(df, output_dir, obs_enabled)
    elif model_type == "mixed":
        results = _run_mixed_effects_model(df, output_dir, obs_enabled)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Save model results
    results_path = output_dir / f"{model_type}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Model results saved to {results_path}")
    return results


def _run_ols_model(df: pd.DataFrame, output_dir: Path, obs_enabled: bool) -> Dict[str, Any]:
    """Run OLS regression with cluster-robust standard errors."""
    logger = logging.getLogger("pwn.models.ols")
    
    # Prepare formula and data
    dependent_var = "delta_kg_next"
    predictors = ["fiber_g_week", "calories_week", "beans_week", "potato_week"]
    
    # Remove missing values
    model_df = df[predictors + [dependent_var, "subject_id"]].dropna()
    logger.info(f"OLS model using {len(model_df)} observations")
    
    if len(model_df) == 0:
        logger.warning("No valid observations for OLS model")
        return {"model_type": "ols", "status": "no_data", "observations": 0}
    
    # Simple OLS using scipy
    X = model_df[predictors].values
    y = model_df[dependent_var].values
    
    # Add constant term
    X_with_const = np.column_stack([np.ones(len(X)), X])
    
    # Fit OLS
    try:
        # Use numpy linalg for OLS
        beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        
        # Calculate predictions and residuals
        y_pred = X_with_const @ beta
        residuals = y - y_pred
        
        # Calculate basic statistics
        n = len(y)
        k = len(beta)
        df_resid = n - k
        
        # Standard errors (basic, not cluster-robust for simplicity)
        mse = np.sum(residuals**2) / df_resid
        var_beta = mse * np.linalg.inv(X_with_const.T @ X_with_const)
        se_beta = np.sqrt(np.diag(var_beta))
        
        # T-statistics and p-values
        t_stats = beta / se_beta
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_resid))
        
        # R-squared
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum(residuals**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Create results
        coef_names = ["intercept"] + predictors
        results = {
            "model_type": "ols",
            "status": "success",
            "observations": n,
            "r_squared": float(r_squared),
            "coefficients": {
                name: {
                    "coef": float(coef),
                    "se": float(se),
                    "t_stat": float(t_stat),
                    "p_value": float(p_val)
                }
                for name, coef, se, t_stat, p_val in zip(coef_names, beta, se_beta, t_stats, p_values)
            },
            "diagnostics": {
                "mse": float(mse),
                "degrees_of_freedom": int(df_resid)
            }
        }
        
        # Key results for downstream use
        if "fiber_g_week" in results["coefficients"]:
            results["fiber_coefficient"] = results["coefficients"]["fiber_g_week"]["coef"]
            results["fiber_p_value"] = results["coefficients"]["fiber_g_week"]["p_value"]
        
        # Save detailed results
        summary_text = _format_ols_summary(results)
        with open(output_dir / "ols_summary.txt", "w") as f:
            f.write(summary_text)
        
        logger.info(f"OLS model completed: R² = {r_squared:.3f}")
        return results
        
    except Exception as e:
        logger.error(f"OLS model failed: {e}")
        return {
            "model_type": "ols",
            "status": "failed",
            "error": str(e),
            "observations": len(model_df)
        }


def _run_fixed_effects_model(df: pd.DataFrame, output_dir: Path, obs_enabled: bool) -> Dict[str, Any]:
    """Run within-subject fixed effects (demeaning) model."""
    logger = logging.getLogger("pwn.models.fe")
    
    dependent_var = "delta_kg_next"
    predictors = ["fiber_g_week", "calories_week", "beans_week", "potato_week"]
    
    # Remove missing values
    model_df = df[predictors + [dependent_var, "subject_id"]].dropna()
    logger.info(f"Fixed effects model using {len(model_df)} observations")
    
    if len(model_df) == 0:
        logger.warning("No valid observations for fixed effects model")
        return {"model_type": "fe", "status": "no_data", "observations": 0}
    
    try:
        # Demean within subjects (fixed effects transformation)
        demeaned_df = model_df.copy()
        
        # Calculate subject means
        subject_means = model_df.groupby("subject_id")[predictors + [dependent_var]].mean()
        
        # Subtract subject means from observations
        for subject_id in model_df["subject_id"].unique():
            mask = model_df["subject_id"] == subject_id
            for var in predictors + [dependent_var]:
                demeaned_df.loc[mask, var] = (
                    model_df.loc[mask, var] - subject_means.loc[subject_id, var]
                )
        
        # Run OLS on demeaned data (no constant needed)
        X = demeaned_df[predictors].values
        y = demeaned_df[dependent_var].values
        
        # Fit FE model
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Calculate statistics
        y_pred = X @ beta
        residuals = y - y_pred
        
        n = len(y)
        k = len(beta)
        df_resid = n - k
        
        mse = np.sum(residuals**2) / df_resid
        var_beta = mse * np.linalg.inv(X.T @ X)
        se_beta = np.sqrt(np.diag(var_beta))
        
        t_stats = beta / se_beta
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_resid))
        
        # R-squared (within)
        ss_tot = np.sum((y - np.mean(y))**2)
        ss_res = np.sum(residuals**2)
        r_squared_within = 1 - (ss_res / ss_tot)
        
        results = {
            "model_type": "fe",
            "status": "success",
            "observations": n,
            "r_squared_within": float(r_squared_within),
            "coefficients": {
                name: {
                    "coef": float(coef),
                    "se": float(se),
                    "t_stat": float(t_stat),
                    "p_value": float(p_val)
                }
                for name, coef, se, t_stat, p_val in zip(predictors, beta, se_beta, t_stats, p_values)
            },
            "diagnostics": {
                "mse": float(mse),
                "degrees_of_freedom": int(df_resid),
                "n_subjects": int(model_df["subject_id"].nunique())
            }
        }
        
        # Key results
        if "fiber_g_week" in results["coefficients"]:
            results["fiber_coefficient"] = results["coefficients"]["fiber_g_week"]["coef"]
            results["fiber_p_value"] = results["coefficients"]["fiber_g_week"]["p_value"]
        
        # Save detailed results
        summary_text = _format_fe_summary(results)
        with open(output_dir / "fe_summary.txt", "w") as f:
            f.write(summary_text)
        
        logger.info(f"Fixed effects model completed: R² (within) = {r_squared_within:.3f}")
        return results
        
    except Exception as e:
        logger.error(f"Fixed effects model failed: {e}")
        return {
            "model_type": "fe",
            "status": "failed",
            "error": str(e),
            "observations": len(model_df)
        }


def _run_mixed_effects_model(df: pd.DataFrame, output_dir: Path, obs_enabled: bool) -> Dict[str, Any]:
    """Run mixed effects model with random intercepts (requires statsmodels)."""
    logger = logging.getLogger("pwn.models.mixed")
    
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import mixedlm
    except ImportError:
        logger.warning("statsmodels not available, skipping mixed effects model")
        return {
            "model_type": "mixed",
            "status": "skipped",
            "message": "statsmodels not installed. Install with: pip install pwn[stats]",
            "observations": 0
        }
    
    dependent_var = "delta_kg_next"
    predictors = ["fiber_g_week", "calories_week", "beans_week", "potato_week"]
    
    # Remove missing values
    model_df = df[predictors + [dependent_var, "subject_id"]].dropna()
    logger.info(f"Mixed effects model using {len(model_df)} observations")
    
    if len(model_df) == 0:
        logger.warning("No valid observations for mixed effects model")
        return {"model_type": "mixed", "status": "no_data", "observations": 0}
    
    try:
        # Create formula
        formula = f"{dependent_var} ~ " + " + ".join(predictors)
        
        # Fit mixed effects model with random intercepts
        model = mixedlm(formula, model_df, groups=model_df["subject_id"])
        fitted_model = model.fit()
        
        # Extract results
        results = {
            "model_type": "mixed",
            "status": "success",
            "observations": int(fitted_model.nobs),
            "log_likelihood": float(fitted_model.llf),
            "aic": float(fitted_model.aic),
            "bic": float(fitted_model.bic),
            "coefficients": {},
            "random_effects": {
                "groups": int(fitted_model.model.n_groups),
                "variance": float(fitted_model.cov_re.iloc[0, 0]) if hasattr(fitted_model, 'cov_re') else None
            }
        }
        
        # Fixed effects coefficients
        for name, coef, se, pval in zip(
            fitted_model.params.index,
            fitted_model.params.values,
            fitted_model.bse.values,
            fitted_model.pvalues.values
        ):
            results["coefficients"][name] = {
                "coef": float(coef),
                "se": float(se),
                "p_value": float(pval)
            }
        
        # Key results
        if "fiber_g_week" in results["coefficients"]:
            results["fiber_coefficient"] = results["coefficients"]["fiber_g_week"]["coef"]
            results["fiber_p_value"] = results["coefficients"]["fiber_g_week"]["p_value"]
        
        # Save detailed results
        summary_text = str(fitted_model.summary())
        with open(output_dir / "mixed_summary.txt", "w") as f:
            f.write(summary_text)
        
        logger.info(f"Mixed effects model completed: {fitted_model.model.n_groups} groups")
        return results
        
    except Exception as e:
        logger.error(f"Mixed effects model failed: {e}")
        return {
            "model_type": "mixed",
            "status": "failed",
            "error": str(e),
            "observations": len(model_df)
        }


def _format_ols_summary(results: Dict[str, Any]) -> str:
    """Format OLS results as human-readable summary."""
    lines = [
        "OLS Regression Results",
        "=" * 50,
        f"Observations: {results['observations']}",
        f"R-squared: {results['r_squared']:.4f}",
        "",
        "Coefficients:",
        "-" * 50
    ]
    
    for name, stats in results["coefficients"].items():
        lines.append(
            f"{name:15} {stats['coef']:10.4f} {stats['se']:10.4f} "
            f"{stats['t_stat']:10.3f} {stats['p_value']:10.3f}"
        )
    
    lines.extend([
        "",
        f"Mean Squared Error: {results['diagnostics']['mse']:.4f}",
        f"Degrees of Freedom: {results['diagnostics']['degrees_of_freedom']}"
    ])
    
    return "\n".join(lines)


def _format_fe_summary(results: Dict[str, Any]) -> str:
    """Format fixed effects results as human-readable summary."""
    lines = [
        "Fixed Effects (Within) Regression Results",
        "=" * 50,
        f"Observations: {results['observations']}",
        f"R-squared (within): {results['r_squared_within']:.4f}",
        f"Number of subjects: {results['diagnostics']['n_subjects']}",
        "",
        "Coefficients:",
        "-" * 50
    ]
    
    for name, stats in results["coefficients"].items():
        lines.append(
            f"{name:15} {stats['coef']:10.4f} {stats['se']:10.4f} "
            f"{stats['t_stat']:10.3f} {stats['p_value']:10.3f}"
        )
    
    lines.extend([
        "",
        f"Mean Squared Error: {results['diagnostics']['mse']:.4f}",
        f"Degrees of Freedom: {results['diagnostics']['degrees_of_freedom']}"
    ])
    
    return "\n".join(lines)