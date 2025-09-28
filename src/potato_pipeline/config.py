"""Configuration management using Pydantic."""

from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    from pydantic import BaseSettings, Field
from loguru import logger


class Config(BaseSettings):
    """Pipeline configuration with automatic path resolution."""
    
    # Input data paths
    base_clean: Path = Field(
        default=Path("~/Downloads/Potato Raw Dato/_clean"),
        description="Base directory containing cleaned data files"
    )
    
    # Output directories
    out_fig_dir: Path = Field(
        default=Path("reports/figs"),
        description="Output directory for figures"
    )
    out_analysis_dir: Path = Field(
        default=Path("reports/analysis"),
        description="Output directory for analysis results"
    )
    
    # Derived input file paths
    @property
    def tidy_csv(self) -> Path:
        """Path to tidy weight/energy/mood data."""
        return self.base_clean / "Potato_tidy.csv"
    
    @property
    def nutrition_rows_csv(self) -> Path:
        """Path to nutrition rows data."""
        return self.base_clean / "Potato_nutrition_rows.csv"
    
    @property
    def fiber_daily_csv(self) -> Path:
        """Path to daily fiber rollup data."""
        return self.base_clean / "Potato_fiber_daily.csv"
    
    def resolve(self) -> "Config":
        """Expand user paths and create output directories."""
        # Expand user paths
        self.base_clean = self.base_clean.expanduser().resolve()
        self.out_fig_dir = self.out_fig_dir.expanduser().resolve()
        self.out_analysis_dir = self.out_analysis_dir.expanduser().resolve()
        
        # Create output directories
        self.out_fig_dir.mkdir(parents=True, exist_ok=True)
        self.out_analysis_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Config resolved: base_clean={self.base_clean}")
        logger.info(f"Output dirs: figs={self.out_fig_dir}, analysis={self.out_analysis_dir}")
        
        return self
    
    class Config:
        """Pydantic config."""
        env_prefix = "POTATO_"


def get_config() -> Config:
    """Get and resolve configuration."""
    return Config().resolve()