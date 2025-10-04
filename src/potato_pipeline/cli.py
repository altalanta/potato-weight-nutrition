"""Command line interface using Typer."""

import sys

import typer
from loguru import logger

from .config import get_config
from .features import aggregate_weekly, align_weeks, prepare_daily, prepare_weight
from .io import load_fiber_daily, load_nutrition_rows, load_tidy, save_table, save_text
from .modeling import format_results_summary, run_full_analysis
from .plotting import create_all_plots

app = typer.Typer(help="Potato Weight-Nutrition Analysis Pipeline")


@app.command()
def pipeline() -> None:
    """Run the complete analysis pipeline."""
    logger.info("Starting complete analysis pipeline")
    
    # Get configuration
    config = get_config()
    
    try:
        # Load data
        logger.info("Loading data...")
        tidy_df = load_tidy(config.tidy_csv)
        nutrition_rows_df = load_nutrition_rows(config.nutrition_rows_csv)
        fiber_daily_df = load_fiber_daily(config.fiber_daily_csv)
        
        # Check if we have any data
        has_data = not (tidy_df.empty and nutrition_rows_df.empty and fiber_daily_df.empty)
        
        if not has_data:
            logger.warning("No data files found - will generate synthetic plots only")
        
        # Feature engineering
        logger.info("Processing features...")
        weight_trajectories = prepare_weight(tidy_df)
        daily_nutrition = prepare_daily(nutrition_rows_df, fiber_daily_df)
        weekly_nutrition = aggregate_weekly(daily_nutrition)
        
        # Align data
        logger.info("Aligning weight and nutrition data...")
        weight_trajectories, weekly_nutrition, analysis_df = align_weeks(
            weight_trajectories, weekly_nutrition
        )
        
        # Statistical modeling
        logger.info("Running statistical analysis...")
        if not analysis_df.empty:
            model_results = run_full_analysis(analysis_df)
            results_summary = format_results_summary(model_results)
        else:
            model_results = {}
            results_summary = "No data available for statistical analysis."
        
        # Create plots
        logger.info("Creating plots...")
        teaser_real, num_subject_plots = create_all_plots(
            weight_trajectories, analysis_df, config.out_fig_dir
        )
        
        # Save outputs
        logger.info("Saving analysis outputs...")
        
        # Save tables
        if not weight_trajectories.empty:
            save_table(
                weight_trajectories,
                config.out_analysis_dir / "weight_trajectories.csv",
                config.out_analysis_dir / "weight_trajectories.parquet"
            )
        
        if not weekly_nutrition.empty:
            save_table(
                weekly_nutrition,
                config.out_analysis_dir / "nutrition_weekly.csv",
                config.out_analysis_dir / "nutrition_weekly.parquet"
            )
        
        if not analysis_df.empty:
            save_table(
                analysis_df,
                config.out_analysis_dir / "analysis_df.csv",
                config.out_analysis_dir / "analysis_df.parquet"
            )
        
        # Save results summary
        save_text(config.out_analysis_dir / "ols_results.txt", results_summary)
        
        # Print summary
        logger.info("Pipeline completed successfully!")
        
        data_type = "real" if has_data else "synthetic"
        teaser_type = "real" if teaser_real else "synthetic"
        
        print(f"\n{'='*60}")
        print("PIPELINE SUMMARY")
        print(f"{'='*60}")
        print(f"Data processed: {data_type}")
        print(f"Teaser plot: {teaser_type}")
        print(f"Subject plots: {num_subject_plots}")
        print(f"Results saved to: {config.out_analysis_dir}")
        print(f"Figures saved to: {config.out_fig_dir}")
        
        if model_results:
            print("\nKey Results:")
            if 'spearman_correlation' in model_results:
                print(f"  Fiber-weight correlation: {model_results['spearman_correlation']:.3f}")
            if 'ols_fiber_coef' in model_results:
                print(f"  OLS fiber coefficient: {model_results['ols_fiber_coef']:.4f}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


@app.command()
def teaser() -> None:
    """Generate or refresh the teaser plot only."""
    logger.info("Creating teaser plot")
    
    config = get_config()
    
    try:
        # Load minimal data needed for teaser
        tidy_df = load_tidy(config.tidy_csv)
        nutrition_rows_df = load_nutrition_rows(config.nutrition_rows_csv)
        fiber_daily_df = load_fiber_daily(config.fiber_daily_csv)
        
        # Quick feature processing for teaser
        weight_trajectories = prepare_weight(tidy_df)
        daily_nutrition = prepare_daily(nutrition_rows_df, fiber_daily_df)
        weekly_nutrition = aggregate_weekly(daily_nutrition)
        _, _, analysis_df = align_weeks(weight_trajectories, weekly_nutrition)
        
        # Create just the teaser plot
        from .plotting import create_teaser_plot
        teaser_path = config.out_fig_dir / "teaser.png"
        teaser_real = create_teaser_plot(analysis_df, teaser_path)
        
        data_type = "real" if teaser_real else "synthetic"
        logger.info(f"Teaser plot created: {teaser_path} ({data_type} data)")
        
        print(f"Teaser plot created: {teaser_path}")
        print(f"Data type: {data_type}")
        
    except Exception as e:
        logger.error(f"Teaser creation failed: {e}")
        # Don't exit with error code - we want this to always succeed
        from .plotting import _create_synthetic_teaser_plot
        teaser_path = config.out_fig_dir / "teaser.png"
        _create_synthetic_teaser_plot(teaser_path, "Weekly Fiber vs Next-Week Weight Change")
        logger.info(f"Fallback synthetic teaser created: {teaser_path}")


@app.command()
def print_config() -> None:
    """Show resolved configuration paths."""
    config = get_config()
    
    print("POTATO PIPELINE CONFIGURATION")
    print("=" * 40)
    print(f"Base clean directory: {config.base_clean}")
    print(f"  Tidy CSV: {config.tidy_csv}")
    print(f"  Nutrition rows CSV: {config.nutrition_rows_csv}")
    print(f"  Fiber daily CSV: {config.fiber_daily_csv}")
    print("Output directories:")
    print(f"  Figures: {config.out_fig_dir}")
    print(f"  Analysis: {config.out_analysis_dir}")
    print()
    
    # Check file existence
    print("File Status:")
    for name, path in [
        ("Tidy CSV", config.tidy_csv),
        ("Nutrition rows CSV", config.nutrition_rows_csv),
        ("Fiber daily CSV", config.fiber_daily_csv)
    ]:
        status = "EXISTS" if path.exists() else "MISSING"
        print(f"  {name}: {status}")


def main() -> None:
    """Main entry point for CLI."""
    # Configure loguru
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO"
    )
    
    app()


if __name__ == "__main__":
    main()