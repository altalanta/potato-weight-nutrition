"""Command-line interface for potato-weight-nutrition package."""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import click
import numpy as np
import pandas as pd

from . import __version__


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up structured logging with run IDs."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create custom formatter for JSON logging
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_obj = {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "module": record.module,
                "message": record.getMessage(),
                "run_id": getattr(record, 'run_id', None)
            }
            return json.dumps(log_obj)
    
    logger = logging.getLogger("pwn")
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    
    return logger


def generate_run_id() -> str:
    """Generate unique run ID."""
    return f"run_{int(time.time())}_{np.random.randint(1000, 9999)}"


def save_lineage(config: dict[str, Any], run_id: str, out_dir: Path) -> None:
    """Save run lineage metadata."""
    import subprocess
    import os
    
    lineage = {
        "run_id": run_id,
        "timestamp": time.time(),
        "config": config,
        "environment": {
            "python_version": sys.version,
            "platform": sys.platform,
            "cwd": str(Path.cwd()),
        },
        "git_info": {}
    }
    
    # Add git information if available
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.DEVNULL, 
            text=True
        ).strip()
        lineage["git_info"]["sha"] = git_sha
        
        git_dirty = subprocess.call(
            ["git", "diff-index", "--quiet", "HEAD", "--"],
            stderr=subprocess.DEVNULL
        ) != 0
        lineage["git_info"]["dirty"] = git_dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    lineage_path = out_dir / "lineage.json"
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lineage_path, "w") as f:
        json.dump(lineage, f, indent=2)


@click.group()
@click.version_option(version=__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--config", type=click.Path(exists=True, path_type=Path), 
              help="Configuration file path")
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility")
@click.option("--obs", is_flag=True, help="Enable observability and telemetry")
@click.option("--lineage", is_flag=True, help="Enable lineage tracking")
@click.pass_context
def main(ctx: click.Context, verbose: bool, config: Optional[Path], seed: int, 
         obs: bool, lineage: bool) -> None:
    """Potato Weight Nutrition: Reproducible pipeline for weight change vs diet analysis."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["config"] = config
    ctx.obj["seed"] = seed
    ctx.obj["obs"] = obs
    ctx.obj["lineage"] = lineage
    ctx.obj["logger"] = setup_logging(verbose)
    
    # Set global random seed
    np.random.seed(seed)


@main.group()
@click.pass_context
def data(ctx: click.Context) -> None:
    """Data operations: synthesis, validation, and quality checks."""
    pass


@data.command("synth")
@click.option("--output-dir", type=click.Path(path_type=Path), default="data/",
              help="Output directory for synthetic data")
@click.option("--n-subjects", type=int, default=50, 
              help="Number of subjects to generate")
@click.option("--n-weeks", type=int, default=12, 
              help="Number of weeks per subject")
@click.pass_context
def data_synth(ctx: click.Context, output_dir: Path, n_subjects: int, n_weeks: int) -> None:
    """Generate synthetic data matching README schema (Potato_tidy.csv etc.)."""
    logger = ctx.obj["logger"]
    run_id = generate_run_id()
    
    logger.info("Starting synthetic data generation", extra={"run_id": run_id})
    
    from .synthetic import generate_synthetic_data
    
    output_dir.mkdir(exist_ok=True)
    
    stats = generate_synthetic_data(
        output_dir=output_dir,
        n_subjects=n_subjects,
        n_weeks=n_weeks,
        seed=ctx.obj["seed"]
    )
    
    # Save lineage if requested
    if ctx.obj["lineage"]:
        save_lineage({
            "n_subjects": n_subjects,
            "n_weeks": n_weeks,
            "seed": ctx.obj["seed"]
        }, run_id, output_dir)
    
    results = {
        "run_id": run_id,
        "status": "success",
        "stats": stats,
        "output_dir": str(output_dir)
    }
    
    click.echo(json.dumps(results, indent=2))
    logger.info("Synthetic data generation completed", extra={"run_id": run_id})


@data.command("validate")
@click.option("--input-dir", type=click.Path(exists=True, path_type=Path), 
              default="data/", help="Input directory containing data files")
@click.pass_context
def data_validate(ctx: click.Context, input_dir: Path) -> None:
    """Run Great Expectations contracts (if installed)."""
    logger = ctx.obj["logger"]
    run_id = generate_run_id()
    
    logger.info("Starting data validation", extra={"run_id": run_id})
    
    try:
        import great_expectations as gx
        from .contracts import validate_data_contracts
        
        validation_results = validate_data_contracts(input_dir)
        
        results = {
            "run_id": run_id,
            "status": "success",
            "validation_results": validation_results
        }
        
    except ImportError:
        results = {
            "run_id": run_id,
            "status": "skipped",
            "message": "Great Expectations not installed. Install with: pip install pwn[contracts]"
        }
    
    click.echo(json.dumps(results, indent=2))
    logger.info("Data validation completed", extra={"run_id": run_id})


@main.command("features")
@click.option("--input-dir", type=click.Path(exists=True, path_type=Path), 
              default="data/", help="Input directory containing data files")
@click.option("--output-dir", type=click.Path(path_type=Path), default="reports/analysis/",
              help="Output directory for processed features")
@click.pass_context
def features(ctx: click.Context, input_dir: Path, output_dir: Path) -> None:
    """Build features: weekly aggregates, keyword categories, baselines."""
    logger = ctx.obj["logger"]
    run_id = generate_run_id()
    
    logger.info("Starting feature engineering", extra={"run_id": run_id})
    
    from .features import build_features
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    feature_stats = build_features(
        input_dir=input_dir,
        output_dir=output_dir,
        config=ctx.obj.get("config")
    )
    
    # Save lineage if requested
    if ctx.obj["lineage"]:
        save_lineage({
            "input_dir": str(input_dir),
            "output_dir": str(output_dir)
        }, run_id, output_dir)
    
    results = {
        "run_id": run_id,
        "status": "success",
        "feature_stats": feature_stats,
        "output_dir": str(output_dir)
    }
    
    click.echo(json.dumps(results, indent=2))
    logger.info("Feature engineering completed", extra={"run_id": run_id})


@main.command("model")
@click.option("--features-dir", type=click.Path(exists=True, path_type=Path),
              default="reports/analysis/", help="Directory containing processed features")
@click.option("--output-dir", type=click.Path(path_type=Path), default="reports/analysis/",
              help="Output directory for model results")
@click.option("--model", type=click.Choice(["ols", "fe", "mixed"]), default="ols",
              help="Statistical model to run")
@click.pass_context
def model(ctx: click.Context, features_dir: Path, output_dir: Path, model: str) -> None:
    """Run statistical models: OLS, FE (fixed effects), Mixed (random effects)."""
    logger = ctx.obj["logger"]
    run_id = generate_run_id()
    
    logger.info(f"Starting {model} model", extra={"run_id": run_id})
    
    from .models import run_statistical_model
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_results = run_statistical_model(
        features_dir=features_dir,
        output_dir=output_dir,
        model_type=model,
        obs_enabled=ctx.obj["obs"]
    )
    
    # Save lineage if requested
    if ctx.obj["lineage"]:
        save_lineage({
            "features_dir": str(features_dir),
            "model_type": model,
            "obs_enabled": ctx.obj["obs"]
        }, run_id, output_dir)
    
    results = {
        "run_id": run_id,
        "status": "success",
        "model_type": model,
        "model_results": model_results,
        "output_dir": str(output_dir)
    }
    
    click.echo(json.dumps(results, indent=2))
    logger.info(f"{model} model completed", extra={"run_id": run_id})


@main.command("report")
@click.option("--results-dir", type=click.Path(exists=True, path_type=Path),
              default="reports/analysis/", help="Directory containing analysis results")
@click.option("--output-dir", type=click.Path(path_type=Path), default="reports/",
              help="Output directory for reports and figures")
@click.option("--teaser", is_flag=True, help="Generate teaser plot only")
@click.pass_context
def report(ctx: click.Context, results_dir: Path, output_dir: Path, teaser: bool) -> None:
    """Generate figures and summary markdown reports."""
    logger = ctx.obj["logger"]
    run_id = generate_run_id()
    
    logger.info("Starting report generation", extra={"run_id": run_id})
    
    from .reporting import generate_reports
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_stats = generate_reports(
        results_dir=results_dir,
        output_dir=output_dir,
        teaser_only=teaser
    )
    
    # Save lineage if requested
    if ctx.obj["lineage"]:
        save_lineage({
            "results_dir": str(results_dir),
            "teaser_only": teaser
        }, run_id, output_dir)
    
    results = {
        "run_id": run_id,
        "status": "success",
        "report_stats": report_stats,
        "output_dir": str(output_dir)
    }
    
    click.echo(json.dumps(results, indent=2))
    logger.info("Report generation completed", extra={"run_id": run_id})


@main.command("repro")
@click.option("--output-dir", type=click.Path(path_type=Path), default="reports/",
              help="Output directory for all pipeline artifacts")
@click.pass_context
def repro(ctx: click.Context, output_dir: Path) -> None:
    """Run end-to-end reproducible pipeline: synth→features→model→report."""
    logger = ctx.obj["logger"]
    run_id = generate_run_id()
    
    logger.info("Starting end-to-end reproducible pipeline", extra={"run_id": run_id})
    start_time = time.time()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Generate synthetic data
        logger.info("Step 1: Generating synthetic data", extra={"run_id": run_id})
        data_dir = output_dir / "data"
        from .synthetic import generate_synthetic_data
        data_stats = generate_synthetic_data(data_dir, seed=ctx.obj["seed"])
        
        # Step 2: Build features
        logger.info("Step 2: Building features", extra={"run_id": run_id})
        analysis_dir = output_dir / "analysis"
        from .features import build_features
        feature_stats = build_features(data_dir, analysis_dir)
        
        # Step 3: Run models
        logger.info("Step 3: Running statistical models", extra={"run_id": run_id})
        from .models import run_statistical_model
        
        model_results = {}
        for model_type in ["ols", "fe"]:  # Skip mixed if stats not available
            try:
                model_results[model_type] = run_statistical_model(
                    analysis_dir, analysis_dir, model_type, ctx.obj["obs"]
                )
            except ImportError as e:
                if "statsmodels" in str(e) and model_type in ["fe", "mixed"]:
                    logger.warning(f"Skipping {model_type} model: statsmodels not installed")
                    continue
                raise
        
        # Step 4: Generate reports
        logger.info("Step 4: Generating reports", extra={"run_id": run_id})
        from .reporting import generate_reports
        report_stats = generate_reports(analysis_dir, output_dir)
        
        elapsed_time = time.time() - start_time
        
        # Save comprehensive lineage
        if ctx.obj["lineage"]:
            save_lineage({
                "pipeline_type": "repro",
                "seed": ctx.obj["seed"],
                "obs_enabled": ctx.obj["obs"],
                "data_stats": data_stats,
                "feature_stats": feature_stats,
                "model_results": {k: v for k, v in model_results.items() if isinstance(v, dict)},
                "report_stats": report_stats
            }, run_id, output_dir)
        
        results = {
            "run_id": run_id,
            "status": "success",
            "elapsed_time_seconds": round(elapsed_time, 2),
            "steps_completed": 4,
            "data_stats": data_stats,
            "feature_stats": feature_stats,
            "model_results": model_results,
            "report_stats": report_stats,
            "output_dir": str(output_dir)
        }
        
        click.echo(json.dumps(results, indent=2))
        logger.info(f"Reproducible pipeline completed in {elapsed_time:.1f}s", 
                   extra={"run_id": run_id})
        
        if elapsed_time > 90:  # 90 seconds target
            logger.warning(f"Pipeline took {elapsed_time:.1f}s (>90s target)", 
                          extra={"run_id": run_id})
    
    except Exception as e:
        logger.error(f"Reproducible pipeline failed: {e}", extra={"run_id": run_id})
        results = {
            "run_id": run_id,
            "status": "failed",
            "error": str(e),
            "elapsed_time_seconds": round(time.time() - start_time, 2)
        }
        click.echo(json.dumps(results, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()