# Potato Weight-Nutrition Analysis Pipeline

<!-- Badges -->
[![CI](https://github.com/example/potato-weight-nutrition/workflows/CI/badge.svg)](https://github.com/example/potato-weight-nutrition/actions/workflows/ci.yml)
[![Security](https://github.com/example/potato-weight-nutrition/workflows/Security/badge.svg)](https://github.com/example/potato-weight-nutrition/actions/workflows/security.yml)
[![Reproducible Run](https://github.com/example/potato-weight-nutrition/workflows/Reproducible%20Run/badge.svg)](https://github.com/example/potato-weight-nutrition/actions/workflows/repro.yml)
[![codecov](https://codecov.io/gh/example/potato-weight-nutrition/branch/main/graph/badge.svg)](https://codecov.io/gh/example/potato-weight-nutrition)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready, reproducible data science pipeline for analyzing weight change patterns in relation to dietary fiber intake and specific food categories, with full observability, lineage tracking, and automated deployment to AWS.
<img width="4771" height="3543" alt="potato_diet_teaser" src="https://github.com/user-attachments/assets/1589cef2-caa2-4a25-98bf-d9e98abbd9ea" />



## Quick Start

### Quickstart (Local)
```bash
# 1. Setup environment
make setup

# 2. Generate teaser plot  
make teaser

# 3. Start observability stack
make obs.up

# 4. Run full pipeline with lineage tracking
make repro
```

### Quickstart (AWS Minimal)
```bash
# 1. Deploy infrastructure
make infra.apply

# 2. Configure GitHub secrets with Terraform outputs
make infra.output

# 3. Push to trigger deployment
git push origin main
```

## Pipeline Overview

This pipeline transforms messy nutritional data into insights about weight change patterns using robust statistical methods.

```
Raw Data (Excel) → Cleaned CSVs → Features → Statistical Models → Insights
     ↓                ↓             ↓            ↓               ↓
  Messy nutritional   Tidy tables   Weekly      OLS, Mixed     Publication-ready
  & weight tracking   with proper   aggregates  Effects,       figures & tables
                      data types    & keywords  Correlations
```

### Pipeline Stages

1. **Data Loading**: Robust CSV import with type checking and date parsing
2. **Feature Engineering**: 
   - Weight trajectory calculation (deltas, percent changes, OLS slopes)
   - Food keyword extraction (11 categories with regex matching)
   - Daily-to-weekly aggregation with calendar alignment
3. **Statistical Modeling**:
   - OLS with cluster-robust standard errors
   - Within-subject fixed effects (demeaning)
   - Mixed effects models (random intercepts)
   - Nonparametric comparisons (Mann-Whitney U)
4. **Visualization**: Publication-ready matplotlib figures with proper statistical annotation

## Real-Data Hygiene

### Messy → Tidy → Features

- **Missing data policy**: Preserve gaps in nutrition data (no forward-filling); treat zeros as missing for end-phase weight measurements only
- **Date alignment**: Calendar weeks mapped to study weeks using subject-specific baselines
- **Outlier handling**: Robust regression methods with cluster-adjusted standard errors
- **Zero-as-missing rules**: Applied selectively based on measurement context (weight end-phases vs. baseline)

### Data Quality Assurance

- Type-safe data loading with pandas dtypes
- Graceful handling of missing files (empty DataFrames + warnings)
- Comprehensive logging throughout pipeline
- Robust error handling (individual failures don't crash pipeline)

## Statistical Choices

### Primary Models

1. **OLS with Cluster-Robust Standard Errors**
   ```
   Δkg_next ~ fiber_g_week + calories_week + beans_week + potato_week + C(subject_id)
   ```
   - Accounts for within-subject correlation
   - Fixed effects for subject-level confounders
   - [statsmodels documentation](https://www.statsmodels.org/stable/regression.html)

2. **Within-Subject Fixed Effects** 
   - Variables demeaned within subject to remove time-invariant confounders
   - Identifies effects from within-person variation only
   - Distribution of subject-specific coefficients reported

3. **Mixed Effects Models** (when available)
   - Random intercepts by subject
   - Handles unbalanced panels gracefully
   - [statsmodels MixedLM](https://www.statsmodels.org/stable/mixed_linear.html)

### Nonparametric Methods

- **Spearman correlations**: Robust to non-linear relationships
- **Mann-Whitney U tests**: Compare food category weeks without distributional assumptions
- Used to validate parametric model findings

## Quick Start

```bash
# Setup
make setup

# Generate teaser plot (works with or without real data)
make teaser

# Run full pipeline (if cleaned CSVs exist)
make run
```

### Expected Input Files

The pipeline expects cleaned data files at:
```
~/Downloads/Potato Raw Dato/_clean/
├── Potato_tidy.csv              # Weight/energy/mood measurements
├── Potato_nutrition_rows.csv    # Daily food entries with nutrients
└── Potato_fiber_daily.csv       # Daily fiber/calorie rollups
```

### Output Structure

```
reports/
├── figs/
│   ├── teaser.png                    # Main result visualization
│   ├── subject_weight_*.png          # Individual trajectories
│   └── calories_vs_delta_multiples.png
└── analysis/
    ├── weight_trajectories.csv       # Per-subject weekly weights
    ├── nutrition_weekly.csv          # Weekly nutrition aggregates
    ├── analysis_df.csv               # Merged modeling dataset
    └── ols_results.txt               # Statistical model summaries
```

## Development

### Testing
```bash
make test     # Run test suite
make lint     # Code quality checks
```

### Key Tests
- **Keyword extraction**: Case-insensitive regex matching for 11 food categories
- **Week alignment**: Calendar date → study week mapping with multiple subjects
- **Statistical robustness**: Missing data handling, edge cases

### Code Quality
- **Type hints**: Full typing with pydantic configuration
- **Modular design**: Separate modules for I/O, features, modeling, plotting
- **Logging**: Comprehensive loguru-based logging
- **Pre-commit hooks**: Automated formatting and linting

## Food Categories

The pipeline extracts these food categories using case-insensitive regex with word boundaries:

| Category | Examples |
|----------|----------|
| **Potato** | potato, potatoes, baked potato, mashed |
| **Beans** | beans, black bean, chickpea, lentils |
| **Rice** | rice, brown rice, wild rice, jasmine |
| **Oats** | oats, oatmeal, steel cut |
| **Bread** | bread, toast, bagel, whole wheat |
| **Fruit** | fruit, apple, banana, berries |
| **Vegetables** | vegetables, carrot, broccoli, spinach |
| **Dairy** | dairy, milk, cheese, yogurt |
| **Meat** | meat, chicken, fish, salmon |
| **Egg** | eggs, scrambled, omelet |
| **Nuts** | nuts, almond, walnut, cashew |

## Technology Stack

- **Python 3.11**: Type hints, modern syntax
- **Data**: pandas, numpy, pyarrow (Parquet)
- **Statistics**: scipy, statsmodels (OLS, mixed effects)
- **Visualization**: matplotlib (publication-ready, no seaborn)
- **CLI**: typer for command interface
- **Config**: pydantic for type-safe settings
- **Logging**: loguru for structured logging
- **Testing**: pytest with comprehensive coverage
- **CI/CD**: GitHub Actions, pre-commit hooks

## 🏗️ Infrastructure & Deployment

### AWS Infrastructure
- **S3 Data Lake**: Bronze/silver/gold layers with lifecycle policies
- **MLflow Backend**: DynamoDB for metadata, S3 for artifacts
- **IAM Security**: GitHub OIDC, least-privilege roles
- **Monitoring**: CloudWatch integration

```bash
# Deploy infrastructure
make infra.plan
make infra.apply

# Environment variables generated automatically
make infra.env
```

### GitHub Actions Workflows
- **CI**: Linting, testing, security scanning, coverage
- **Security**: Bandit, pip-audit, secret scanning
- **Reproducible Run**: Nightly pipeline validation
- **Infrastructure**: Terraform plan/apply with OIDC

## Observability & Lineage

### Real-time Monitoring
- **Grafana Dashboards**: Pipeline metrics, data quality, SLO tracking
- **Structured Logging**: JSON logs with correlation IDs
- **Distributed Tracing**: End-to-end request tracking
- **Alerting**: PagerDuty integration for critical issues

### Data Lineage
- **OpenLineage**: Automatic lineage collection
- **Marquez**: Visual data flow graphs
- **Schema Evolution**: Track data contract changes
- **Impact Analysis**: Downstream dependency mapping

```bash
# Start observability stack
make obs.up     # Grafana: http://localhost:3000

# Start lineage tracking  
make lineage.up # Marquez: http://localhost:3000
```

### Data Contracts & Quality
- **Great Expectations**: Automated data validation
- **Schema Contracts**: Bronze and silver layer validation
- **Quality Gates**: CI/CD pipeline integration
- **Anomaly Detection**: Statistical drift monitoring

```bash
# Validate data contracts
make contracts.validate

# Available expectations:
# - data_contracts/expectations/bronze_nutrition_data.yml
# - data_contracts/expectations/silver_features.yml
```

## Security & PHI

### Data Classification
- **Current Status**: No PHI/PII - synthetic data only
- **Security Posture**: Production-ready controls
- **Access Control**: Role-based permissions (Reader/Writer/Admin)
- **Encryption**: At rest (S3, DynamoDB) and in transit (TLS 1.2+)

### Authentication & Authorization
- **GitHub OIDC**: No long-lived AWS keys
- **Least Privilege**: Granular IAM policies
- **Multi-Environment**: Dev/staging/prod isolation
- **Audit Logging**: CloudTrail + application logs

See [Security Documentation](ops/security-notes.md) for detailed policies.

## Operations

### Reproducible Run
```bash
# Complete end-to-end validation
make repro

# Generates:
# - reports/repro_run/index.md
# - reports/repro_run/metrics.json
# - Observability dashboards
# - Lineage graphs
```

### System Health
- **SLO**: Silver data freshness < 2 hours
- **Monitoring**: Pipeline success rate > 95%
- **Performance**: p95 task latency < 5 minutes
- **Capacity**: Auto-scaling based on data volume

### Runbook & Troubleshooting
- [Operational Runbook](ops/runbook.md) - Incident response procedures
- [System Architecture](ops/system-diagram.md) - Component diagrams
- [Security Notes](ops/security-notes.md) - Security policies

## Available Commands

### Development
```bash
make setup          # Install dependencies
make test           # Run test suite  
make lint           # Code quality checks
make teaser         # Generate demo plot
make run            # Run pipeline
```

### Infrastructure
```bash
make infra.init     # Initialize Terraform
make infra.plan     # Plan infrastructure changes
make infra.apply    # Deploy to AWS
make infra.destroy  # Cleanup resources
make infra.output   # Show deployed resources
```

### Observability
```bash
make obs.up         # Start Grafana, Prometheus, Loki
make obs.down       # Stop observability stack
make lineage.up     # Start Marquez lineage tracking
make contracts.validate  # Run data quality checks
```

### Reproducibility
```bash
make repro          # Full reproducible run
make check          # Lint + test
make all            # Setup + teaser + run + test
```

## System Architecture

```
GitHub Actions → AWS (OIDC) → S3 Data Lake + MLflow + DynamoDB
       ↓                           ↓
   CI/CD Pipeline              Pipeline Processing
       ↓                           ↓  
 Security Scanning          OpenTelemetry + Lineage
       ↓                           ↓
   Infrastructure            Grafana + Marquez Dashboards
```

![System Diagram](ops/system-diagram.png)

## Links & Resources

### Documentation
- **[Reproducible Run](reports/repro_run/index.md)** - Latest pipeline execution
- **[Infrastructure Guide](infra/terraform/README.md)** - AWS deployment
- **[Lineage Demo](openlineage/lineage-demo.md)** - Data flow tracking
- **[Security Policies](ops/security-notes.md)** - Data protection

### Dashboards (Local Development)
- **[Grafana](http://localhost:3000)** - Pipeline observability
- **[Marquez](http://localhost:3000)** - Data lineage (different stack)
- **[Prometheus](http://localhost:9090)** - Metrics storage
- **[Jaeger](http://localhost:16686)** - Distributed tracing

### External
- **[Codecov Dashboard](https://codecov.io/gh/example/potato-weight-nutrition)** - Test coverage
- **[GitHub Actions](https://github.com/example/potato-weight-nutrition/actions)** - CI/CD status

## Contributing

1. Fork and clone the repository
2. Run `make setup` to install dependencies
3. Create feature branch
4. Add tests for new functionality
5. Ensure `make test` and `make lint` pass
6. Submit pull request

## License

MIT License - see [LICENSE](LICENSE) for details.
