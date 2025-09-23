# OpenLineage + Marquez Lineage Demo

This directory contains the configuration and documentation for data lineage tracking using OpenLineage and Marquez.

## Overview

- **OpenLineage**: Open standard for data lineage collection
- **Marquez**: Metadata service that implements the OpenLineage API
- **Integration**: Python pipeline automatically emits lineage events

## Quick Start

### 1. Start Marquez Services

```bash
# Start Marquez + PostgreSQL
docker compose -f openlineage/marquez-docker-compose.yml up -d

# Wait for services to be healthy (30-60 seconds)
docker compose -f openlineage/marquez-docker-compose.yml ps

# Check Marquez API is responding
curl http://localhost:5000/api/v1/namespaces
```

### 2. Access Marquez Web UI

Open http://localhost:3000 in your browser to see the lineage graph.

### 3. Run Pipeline with Lineage Tracking

```bash
# Set OpenLineage configuration
export OPENLINEAGE_URL=http://localhost:5000
export OPENLINEAGE_NAMESPACE=potato-pipeline

# Run the pipeline (lineage events will be sent automatically)
PYTHONPATH=src python -m potato_pipeline.cli pipeline
```

### 4. View Lineage in Marquez

1. Go to http://localhost:3000
2. Select the `potato-pipeline` namespace
3. View the job dependency graph
4. Click on jobs to see run details and dataset lineage

## Architecture

```
Pipeline Jobs → OpenLineage Events → Marquez API → PostgreSQL
                                        ↓
                                  Marquez Web UI
```

### Pipeline Integration Points

The pipeline automatically emits OpenLineage events at these stages:

1. **Data Loading**: Input datasets from CSV files
2. **Feature Engineering**: Transformation from bronze → silver
3. **Model Training**: Training job with input features and model outputs
4. **Report Generation**: Analysis outputs and visualizations

### Event Types

- **START**: Job execution begins
- **RUNNING**: Job is actively processing
- **COMPLETE**: Job finished successfully
- **FAIL**: Job encountered an error

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENLINEAGE_URL` | Marquez API endpoint | `http://localhost:5000` |
| `OPENLINEAGE_NAMESPACE` | Namespace for lineage events | `potato-pipeline` |
| `OPENLINEAGE_PRODUCER` | Producer identifier | `potato-weight-nutrition/v1.0` |

### Pipeline Code Integration

The OpenLineage client is integrated into the pipeline at key execution points:

```python
from openlineage.client import OpenLineageClient

# Initialize client
client = OpenLineageClient(url=os.getenv('OPENLINEAGE_URL'))

# Emit job start event
client.emit(
    RunEvent(
        eventType=RunState.START,
        eventTime=datetime.now(),
        run=Run(runId=str(uuid.uuid4())),
        job=Job(
            namespace="potato-pipeline",
            name="feature-engineering"
        ),
        inputs=[
            Dataset(
                namespace="potato-pipeline",
                name="bronze_nutrition_data",
                facets={
                    "schema": SchemaDatasetFacet(
                        fields=[
                            SchemaField(name="date", type="DATE"),
                            SchemaField(name="calories", type="FLOAT"),
                            # ... other fields
                        ]
                    )
                }
            )
        ],
        outputs=[
            Dataset(
                namespace="potato-pipeline", 
                name="silver_features",
                facets={
                    "schema": SchemaDatasetFacet(
                        fields=[
                            SchemaField(name="subject_id", type="STRING"),
                            SchemaField(name="fiber_per_1000kcal", type="FLOAT"),
                            # ... other fields
                        ]
                    )
                }
            )
        ]
    )
)
```

## Sample Lineage Graph

After running the pipeline, you'll see a lineage graph like this:

```
CSV Files → Data Loading → Feature Engineering → Model Training → Report Generation
    ↓           ↓               ↓                    ↓               ↓
bronze_data → silver_data → training_features → trained_model → analysis_reports
```

## Troubleshooting

### Services Won't Start

```bash
# Check Docker logs
docker compose -f openlineage/marquez-docker-compose.yml logs

# Restart services
docker compose -f openlineage/marquez-docker-compose.yml down
docker compose -f openlineage/marquez-docker-compose.yml up -d
```

### No Lineage Events Appearing

1. Check that `OPENLINEAGE_URL` is set correctly
2. Verify Marquez API is responding: `curl http://localhost:5000/api/v1/namespaces`
3. Check pipeline logs for OpenLineage client errors
4. Ensure the pipeline is running with lineage integration enabled

### Database Connection Issues

```bash
# Check PostgreSQL is healthy
docker exec marquez-postgres pg_isready -U marquez

# Reset database if needed
docker compose -f openlineage/marquez-docker-compose.yml down -v
docker compose -f openlineage/marquez-docker-compose.yml up -d
```

## Cleanup

```bash
# Stop all services
docker compose -f openlineage/marquez-docker-compose.yml down

# Remove all data (destructive)
docker compose -f openlineage/marquez-docker-compose.yml down -v
```

## Screenshots

After running the demo, you should see:

1. **Namespace View**: List of pipelines and jobs
2. **Job Graph**: Visual representation of data flow
3. **Dataset Details**: Schema and lineage for each dataset
4. **Run History**: Timeline of job executions

![Marquez Lineage Graph](../docs/images/marquez-lineage-example.png)

## Next Steps

1. **Advanced Schema Tracking**: Add column-level lineage
2. **Data Quality Integration**: Link Great Expectations results
3. **Custom Facets**: Add business metadata to datasets
4. **Alerting**: Set up notifications for lineage breaks
5. **Production Deployment**: Deploy Marquez to AWS/Kubernetes

## References

- [OpenLineage Specification](https://openlineage.io/spec)
- [Marquez Documentation](https://marquezproject.github.io/marquez/)
- [Python Client Guide](https://openlineage.io/docs/client/python)