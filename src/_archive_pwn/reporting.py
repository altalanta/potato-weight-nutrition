"""Report generation: figures, summary markdown, and metrics."""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_reports(
    results_dir: Path,
    output_dir: Path,
    teaser_only: bool = False
) -> dict[str, Any]:
    """Generate figures and summary reports from analysis results.
    
    Args:
        results_dir: Directory containing analysis results
        output_dir: Directory to save reports and figures
        teaser_only: Generate only the teaser plot
        
    Returns:
        Dictionary with report generation statistics
    """
    logger = logging.getLogger("pwn.reporting")
    logger.info("Starting report generation")
    
    # Create output directories
    figs_dir = output_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {"figures_created": [], "reports_created": []}
    
    # Load analysis data if available
    analysis_path = results_dir / "analysis_df.csv"
    weight_path = results_dir / "weight_trajectories.csv"
    
    if analysis_path.exists():
        analysis_df = pd.read_csv(analysis_path)
        logger.info(f"Loaded analysis data: {len(analysis_df)} rows")
    else:
        analysis_df = pd.DataFrame()
        logger.warning("No analysis data found, will generate synthetic plots")
    
    if weight_path.exists():
        weight_df = pd.read_csv(weight_path)
        logger.info(f"Loaded weight trajectories: {len(weight_df)} rows")
    else:
        weight_df = pd.DataFrame()
    
    # Generate teaser plot
    teaser_path = figs_dir / "teaser.png"
    _create_teaser_plot(analysis_df, teaser_path)
    stats["figures_created"].append("teaser.png")
    
    if not teaser_only:
        # Generate subject trajectory plots
        if not weight_df.empty:
            subject_plots = _create_subject_plots(weight_df, figs_dir)
            stats["figures_created"].extend(subject_plots)
        
        # Generate analysis plots
        if not analysis_df.empty:
            analysis_plots = _create_analysis_plots(analysis_df, figs_dir)
            stats["figures_created"].extend(analysis_plots)
        
        # Generate summary report
        report_path = output_dir / "index.md"
        _create_summary_report(results_dir, report_path)
        stats["reports_created"].append("index.md")
        
        # Create metrics summary
        metrics_path = output_dir / "metrics.json"
        _create_metrics_summary(results_dir, metrics_path)
        stats["reports_created"].append("metrics.json")
    
    logger.info(f"Report generation completed: {len(stats['figures_created'])} figures, {len(stats['reports_created'])} reports")
    return stats


def _create_teaser_plot(analysis_df: pd.DataFrame, output_path: Path) -> None:
    """Create the main teaser plot showing fiber vs weight change."""
    logger = logging.getLogger("pwn.reporting.teaser")
    
    plt.figure(figsize=(10, 6))
    
    if not analysis_df.empty and "fiber_g_week" in analysis_df.columns and "delta_kg_next" in analysis_df.columns:
        # Real data plot
        logger.info("Creating teaser plot with real data")
        
        # Remove missing values
        plot_df = analysis_df[["fiber_g_week", "delta_kg_next"]].dropna()
        
        if not plot_df.empty:
            x = plot_df["fiber_g_week"]
            y = plot_df["delta_kg_next"]
            
            # Scatter plot
            plt.scatter(x, y, alpha=0.6, s=30, color='steelblue')
            
            # Add trend line if enough data
            if len(plot_df) > 5:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
            
            plt.xlabel("Weekly Fiber Intake (g)")
            plt.ylabel("Next Week Weight Change (kg)")
            plt.title("Weekly Fiber vs Next-Week Weight Change")
            
            # Add correlation info
            corr = x.corr(y)
            plt.text(0.05, 0.95, f"Correlation: {corr:.3f}", 
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            _create_synthetic_teaser_plot(output_path, "Weekly Fiber vs Next-Week Weight Change (No Data)")
            return
    else:
        # Synthetic data plot
        logger.info("Creating synthetic teaser plot")
        _create_synthetic_teaser_plot(output_path, "Weekly Fiber vs Next-Week Weight Change (Synthetic)")
        return
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Teaser plot saved: {output_path}")


def _create_synthetic_teaser_plot(output_path: Path, title: str) -> None:
    """Create a synthetic teaser plot for demonstration."""
    np.random.seed(42)
    
    # Generate synthetic data with realistic correlation
    n_points = 200
    fiber_intake = np.random.uniform(10, 50, n_points)
    
    # Negative correlation: more fiber -> less weight gain (or more loss)
    weight_change = -0.02 * fiber_intake + np.random.normal(0, 0.5, n_points)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(fiber_intake, weight_change, alpha=0.6, s=30, color='steelblue')
    
    # Add trend line
    z = np.polyfit(fiber_intake, weight_change, 1)
    p = np.poly1d(z)
    plt.plot(fiber_intake, p(fiber_intake), "r--", alpha=0.8, linewidth=2)
    
    plt.xlabel("Weekly Fiber Intake (g)")
    plt.ylabel("Next Week Weight Change (kg)")
    plt.title(title)
    
    # Add correlation info
    corr = np.corrcoef(fiber_intake, weight_change)[0, 1]
    plt.text(0.05, 0.95, f"Correlation: {corr:.3f}", 
            transform=plt.gca().transAxes, fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _create_subject_plots(weight_df: pd.DataFrame, figs_dir: Path) -> list[str]:
    """Create individual subject weight trajectory plots."""
    logger = logging.getLogger("pwn.reporting.subjects")
    
    plots_created = []
    
    if "subject_id" not in weight_df.columns:
        logger.warning("No subject_id column found in weight data")
        return plots_created
    
    subjects = weight_df["subject_id"].unique()
    n_subjects = min(len(subjects), 6)  # Limit to 6 plots
    
    logger.info(f"Creating trajectory plots for {n_subjects} subjects")
    
    for i, subject_id in enumerate(subjects[:n_subjects]):
        subject_data = weight_df[weight_df["subject_id"] == subject_id].copy()
        
        if "week" in subject_data.columns and "weight_kg" in subject_data.columns:
            subject_data = subject_data.sort_values("week")
            
            plt.figure(figsize=(8, 5))
            plt.plot(subject_data["week"], subject_data["weight_kg"], 
                    marker="o", linewidth=2, markersize=6)
            plt.xlabel("Study Week")
            plt.ylabel("Weight (kg)")
            plt.title(f"Subject {subject_id} - Weight Trajectory")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_name = f"subject_weight_{subject_id}.png"
            plot_path = figs_dir / plot_name
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots_created.append(plot_name)
    
    logger.info(f"Created {len(plots_created)} subject trajectory plots")
    return plots_created


def _create_analysis_plots(analysis_df: pd.DataFrame, figs_dir: Path) -> list[str]:
    """Create additional analysis plots."""
    logger = logging.getLogger("pwn.reporting.analysis")
    
    plots_created = []
    
    # Calories vs weight change plot
    if "calories_week" in analysis_df.columns and "delta_kg_next" in analysis_df.columns:
        plot_df = analysis_df[["calories_week", "delta_kg_next"]].dropna()
        
        if not plot_df.empty:
            plt.figure(figsize=(8, 6))
            plt.scatter(plot_df["calories_week"], plot_df["delta_kg_next"], 
                       alpha=0.6, s=30, color='orange')
            plt.xlabel("Weekly Calories")
            plt.ylabel("Next Week Weight Change (kg)")
            plt.title("Weekly Calories vs Weight Change")
            
            # Add trend line
            if len(plot_df) > 5:
                z = np.polyfit(plot_df["calories_week"], plot_df["delta_kg_next"], 1)
                p = np.poly1d(z)
                plt.plot(plot_df["calories_week"], p(plot_df["calories_week"]), 
                        "r--", alpha=0.8, linewidth=2)
            
            plt.tight_layout()
            plot_path = figs_dir / "calories_vs_delta_kg.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots_created.append("calories_vs_delta_kg.png")
    
    logger.info(f"Created {len(plots_created)} analysis plots")
    return plots_created


def _create_summary_report(results_dir: Path, output_path: Path) -> None:
    """Create markdown summary report."""
    logger = logging.getLogger("pwn.reporting.summary")
    
    # Load model results if available
    model_results = {}
    for model_type in ["ols", "fe", "mixed"]:
        results_file = results_dir / f"{model_type}_results.json"
        if results_file.exists():
            with open(results_file) as f:
                model_results[model_type] = json.load(f)
    
    # Load feature stats if available
    lineage_file = results_dir / "lineage.json"
    feature_stats = {}
    if lineage_file.exists():
        with open(lineage_file) as f:
            lineage = json.load(f)
            feature_stats = lineage.get("config", {}).get("feature_stats", {})
    
    # Create markdown content
    lines = [
        "# Potato Weight-Nutrition Analysis Report",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "This report presents the results of analyzing the relationship between dietary fiber intake and weight change patterns.",
        ""
    ]
    
    # Add figures section
    lines.extend([
        "## Figures",
        "",
        "### Main Results",
        "",
        "![Teaser Plot](figs/teaser.png)",
        "",
        "*Weekly fiber intake versus next-week weight change*",
        ""
    ])
    
    # Add model results
    if model_results:
        lines.extend([
            "## Statistical Models",
            ""
        ])
        
        for model_type, results in model_results.items():
            if results.get("status") == "success":
                lines.extend([
                    f"### {model_type.upper()} Model",
                    "",
                    f"- Observations: {results.get('observations', 'N/A')}",
                    f"- R-squared: {results.get('r_squared', results.get('r_squared_within', 'N/A')):.4f}" if isinstance(results.get('r_squared', results.get('r_squared_within')), (int, float)) else "- R-squared: N/A",
                    ""
                ])
                
                if "fiber_coefficient" in results:
                    lines.extend([
                        f"**Fiber Coefficient**: {results['fiber_coefficient']:.4f}",
                        f"**P-value**: {results.get('fiber_p_value', 'N/A'):.4f}" if isinstance(results.get('fiber_p_value'), (int, float)) else "**P-value**: N/A",
                        ""
                    ])
    
    # Add data summary
    if feature_stats:
        lines.extend([
            "## Data Summary",
            "",
            f"- Subjects: {feature_stats.get('n_subjects', 'N/A')}",
            f"- Weeks: {feature_stats.get('n_weeks', 'N/A')}",
            f"- Total observations: {feature_stats.get('total_observations', 'N/A')}",
            ""
        ])
    
    lines.extend([
        "## Methods",
        "",
        "- **Data Processing**: Weekly aggregation with subject baselines",
        "- **Feature Engineering**: Food keyword extraction, weight trajectories",
        "- **Statistical Models**: OLS with cluster-robust SE, fixed effects, mixed effects",
        "- **Visualization**: Publication-ready matplotlib figures",
        ""
    ])
    
    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Summary report created: {output_path}")


def _create_metrics_summary(results_dir: Path, output_path: Path) -> None:
    """Create JSON metrics summary for monitoring."""
    logger = logging.getLogger("pwn.reporting.metrics")
    
    metrics = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "pipeline_status": "success",
        "models": {}
    }
    
    # Collect model metrics
    for model_type in ["ols", "fe", "mixed"]:
        results_file = results_dir / f"{model_type}_results.json"
        if results_file.exists():
            with open(results_file) as f:
                model_results = json.load(f)
                
            metrics["models"][model_type] = {
                "status": model_results.get("status", "unknown"),
                "observations": model_results.get("observations", 0),
                "r_squared": model_results.get("r_squared", model_results.get("r_squared_within")),
                "fiber_coefficient": model_results.get("fiber_coefficient"),
                "fiber_p_value": model_results.get("fiber_p_value")
            }
    
    # Pipeline health metrics
    metrics["health"] = {
        "models_run": len(metrics["models"]),
        "successful_models": sum(1 for m in metrics["models"].values() if m["status"] == "success"),
        "total_observations": sum(m.get("observations", 0) for m in metrics["models"].values())
    }
    
    # Write metrics
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    
    logger.info(f"Metrics summary created: {output_path}")


# Configure matplotlib for publication-quality plots
plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})