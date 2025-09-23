# Operational Runbook

## 🚀 Quick Start

### Emergency Contacts
- **On-Call Engineer**: [oncall@example.com](mailto:oncall@example.com)
- **System Owner**: [team@example.com](mailto:team@example.com)
- **Security**: [security@example.com](mailto:security@example.com)

### Service URLs
- **Grafana Dashboard**: http://localhost:3000 (local) / https://grafana.example.com (prod)
- **Marquez Lineage**: http://localhost:3000 (local) / https://lineage.example.com (prod)
- **MLflow UI**: http://localhost:5000 (local) / https://mlflow.example.com (prod)
- **Prometheus**: http://localhost:9090 (local)

## 📊 Monitoring & Alerting

### Key Metrics (SLIs)

#### Silver Data Freshness (Primary SLO)
- **Target**: Silver layer updated within 2 hours of new bronze data
- **Critical**: > 4 hours old (triggers incident)
- **Query**: `(time() - potato_pipeline_last_successful_run_timestamp) / 3600`

#### Pipeline Success Rate
- **Target**: > 95% successful runs over 24h
- **Warning**: < 98% success rate
- **Query**: `rate(potato_pipeline_tasks_total{status="success"}[24h]) / rate(potato_pipeline_tasks_total[24h])`

#### Task Latency
- **Target**: p95 < 300 seconds for any task
- **Warning**: p95 > 600 seconds
- **Query**: `histogram_quantile(0.95, rate(potato_pipeline_task_duration_seconds_bucket[5m]))`

### Alert Definitions

```yaml
groups:
  - name: potato-pipeline
    rules:
      - alert: PipelineDataStale
        expr: (time() - potato_pipeline_last_successful_run_timestamp) / 3600 > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Silver data is stale (> 2 hours old)"
          runbook: "https://github.com/example/potato-weight-nutrition/blob/main/ops/runbook.md#data-freshness-issues"

      - alert: PipelineDataCritical  
        expr: (time() - potato_pipeline_last_successful_run_timestamp) / 3600 > 4
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "Silver data is critically stale (> 4 hours old)"
          runbook: "https://github.com/example/potato-weight-nutrition/blob/main/ops/runbook.md#data-freshness-issues"

      - alert: PipelineFailureRate
        expr: rate(potato_pipeline_tasks_total{status="failure"}[1h]) / rate(potato_pipeline_tasks_total[1h]) > 0.05
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Pipeline failure rate > 5% over 1 hour"
          runbook: "https://github.com/example/potato-weight-nutrition/blob/main/ops/runbook.md#pipeline-failures"

      - alert: PipelineTaskLatency
        expr: histogram_quantile(0.95, rate(potato_pipeline_task_duration_seconds_bucket[5m])) > 600
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Pipeline task p95 latency > 10 minutes"
          runbook: "https://github.com/example/potato-weight-nutrition/blob/main/ops/runbook.md#performance-issues"
```

## 🔥 Incident Response

### Incident Severity Levels

#### P0: Critical (< 15 min response)
- Pipeline completely down > 4 hours
- Data corruption detected
- Security breach

#### P1: High (< 1 hour response)  
- Silver data > 2 hours stale
- Pipeline failure rate > 20%
- Authentication issues

#### P2: Medium (< 4 hours response)
- Performance degradation
- Non-critical errors in logs
- Monitoring gaps

#### P3: Low (< 24 hours response)
- Documentation updates needed
- Enhancement requests
- Non-urgent maintenance

### Escalation Path

1. **L1: On-call Engineer** (0-30 min)
2. **L2: Senior Engineer** (30-60 min)  
3. **L3: Engineering Manager** (1-2 hours)
4. **L4: Director/VP** (2+ hours, P0 only)

## 🛠️ Common Issues & Resolutions

### Data Freshness Issues

#### Symptoms
- Alert: `PipelineDataStale` or `PipelineDataCritical`
- Grafana: "Silver Data Freshness" panel shows > 2 hours
- Stakeholders reporting stale analysis results

#### Investigation Steps

1. **Check pipeline status**:
   ```bash
   # Local development
   make obs.status
   docker compose -f docker/docker-compose.yml logs potato-pipeline
   
   # Production
   kubectl get pods -n potato-pipeline
   kubectl logs -f deployment/potato-pipeline -n potato-pipeline
   ```

2. **Check recent runs in Grafana**:
   - Look at "Task Success/Failure Rates" panel
   - Check "Pipeline Error Logs" for specific errors
   - Review "Rows Processed by Task" for data volume changes

3. **Check data sources**:
   ```bash
   # Verify input data availability
   aws s3 ls s3://cdp-prod-potato-data-lake-xyz/bronze/ --recursive
   
   # Check for file permission issues  
   aws s3api head-object --bucket cdp-prod-potato-data-lake-xyz --key bronze/latest/nutrition.csv
   ```

4. **Check infrastructure**:
   ```bash
   # Check AWS resources
   terraform output -state=infra/terraform/terraform.tfstate
   
   # Verify IAM permissions
   aws sts get-caller-identity
   aws iam simulate-principal-policy --policy-source-arn $(aws sts get-caller-identity --query Arn --output text) --action-names s3:GetObject --resource-arns arn:aws:s3:::cdp-prod-potato-data-lake-xyz/*
   ```

#### Resolution Steps

1. **Manual pipeline trigger** (quick fix):
   ```bash
   # Local
   PYTHONPATH=src python -m potato_pipeline.cli pipeline
   
   # Production
   kubectl create job --from=cronjob/potato-pipeline potato-pipeline-manual-$(date +%s)
   ```

2. **Fix root cause**:
   - **Data source issues**: Contact data provider, check S3 permissions
   - **Code issues**: Deploy hotfix, revert problematic changes
   - **Infrastructure issues**: Scale resources, fix network connectivity
   - **Dependency issues**: Update versions, restart services

3. **Verify resolution**:
   ```bash
   # Check freshness metric returns to normal
   curl -s http://localhost:9090/api/v1/query?query='(time()%20-%20potato_pipeline_last_successful_run_timestamp)%20/%203600'
   
   # Verify new data in silver layer
   aws s3 ls s3://cdp-prod-potato-data-lake-xyz/silver/ --recursive --human-readable
   ```

### Pipeline Failures

#### Symptoms
- Alert: `PipelineFailureRate`
- Multiple tasks showing "FAILED" status in logs
- Grafana showing error spikes

#### Common Failure Modes

1. **Data Quality Issues**:
   ```bash
   # Check Great Expectations validation
   make contracts.validate
   
   # Review data contracts violations in logs
   grep "expectation.*failed" /var/log/potato-pipeline.log
   ```

2. **Resource Constraints**:
   ```bash
   # Check memory/CPU usage
   docker stats
   kubectl top pods -n potato-pipeline
   
   # Check disk space
   df -h
   aws s3api head-object --bucket cdp-prod-potato-data-lake-xyz --key bronze/large-file.csv
   ```

3. **Dependency Failures**:
   ```bash
   # Check external service health
   curl -f http://localhost:5000/api/v1/namespaces  # Marquez
   curl -f http://localhost:4317/v1/traces  # OTEL Collector
   
   # Check database connectivity
   psql $DATABASE_URL -c "SELECT 1"
   ```

#### Resolution Strategies

1. **Immediate**: Skip failed tasks, process remaining data
2. **Short-term**: Fix data quality issues, restart failed tasks
3. **Long-term**: Improve error handling, add retries, enhance monitoring

### Performance Issues

#### Symptoms
- Alert: `PipelineTaskLatency`
- Users reporting slow dashboard loading
- High resource utilization

#### Investigation

1. **Identify bottlenecks**:
   ```bash
   # Check slowest tasks
   grep "duration_seconds" /var/log/potato-pipeline.log | sort -k3 -nr | head -10
   
   # Profile memory usage
   python -m memory_profiler src/potato_pipeline/cli.py pipeline
   ```

2. **Check data volume trends**:
   ```bash
   # Compare current vs historical data sizes
   aws s3api list-objects-v2 --bucket cdp-prod-potato-data-lake-xyz --prefix bronze/ --query 'sum(Contents[].Size)'
   ```

3. **Review query performance**:
   ```sql
   -- Check slow queries if using database
   SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;
   ```

#### Optimization

1. **Code optimizations**:
   - Add data sampling for development
   - Optimize pandas operations (vectorization)
   - Use chunked processing for large datasets

2. **Infrastructure scaling**:
   ```bash
   # Scale compute resources
   kubectl scale deployment potato-pipeline --replicas=3
   
   # Use larger instance types
   terraform apply -var="instance_type=m5.xlarge"
   ```

### Authentication/Authorization Issues

#### Symptoms
- 403 Forbidden errors in logs
- AWS CLI commands failing with AccessDenied
- GitHub Actions failing with OIDC errors

#### Investigation

1. **Check IAM roles and policies**:
   ```bash
   # Verify current identity
   aws sts get-caller-identity
   
   # Check effective permissions
   aws iam simulate-principal-policy --policy-source-arn $(aws sts get-caller-identity --query Arn --output text) --action-names s3:GetObject,s3:PutObject,dynamodb:GetItem
   ```

2. **Check OIDC configuration**:
   ```bash
   # Verify GitHub OIDC provider
   aws iam list-open-id-connect-providers
   
   # Check trust policy
   aws iam get-role --role-name cdp-dev-github-oidc-role --query 'Role.AssumeRolePolicyDocument'
   ```

#### Resolution

1. **Update IAM policies**:
   ```bash
   # Apply latest Terraform changes
   cd infra/terraform
   terraform plan
   terraform apply
   ```

2. **Rotate credentials** (if using access keys):
   ```bash
   aws iam create-access-key --user-name potato-pipeline-user
   # Update secrets in GitHub/environment
   aws iam delete-access-key --access-key-id AKIA... --user-name potato-pipeline-user
   ```

## 📋 Maintenance Procedures

### Regular Maintenance (Weekly)

1. **Review metrics and trends**:
   - Check SLO compliance over past week
   - Review error logs for patterns
   - Monitor resource utilization trends

2. **Update dependencies**:
   ```bash
   # Update Python packages
   pip-audit
   pip install --upgrade -r requirements.txt
   
   # Update Docker images
   docker compose -f docker/docker-compose.yml pull
   ```

3. **Clean up old data**:
   ```bash
   # Remove old pipeline artifacts
   find reports/ -name "*.csv" -mtime +30 -delete
   
   # Clean up old container images
   docker system prune -f
   ```

### Monthly Maintenance

1. **Security patching**:
   ```bash
   # Update base images
   docker pull python:3.11-slim
   
   # Rebuild with latest patches
   docker build --no-cache -t potato-pipeline:latest .
   ```

2. **Backup validation**:
   ```bash
   # Test restore from backup
   aws s3 sync s3://cdp-prod-potato-data-lake-xyz/silver/ ./backup-test/
   
   # Verify data integrity
   python scripts/validate_backup.py ./backup-test/
   ```

3. **Disaster recovery testing**:
   - Deploy to alternate region
   - Test failover procedures
   - Validate recovery time objectives

### Emergency Procedures

#### Complete System Outage

1. **Assess scope**:
   - Check if issue is regional (AWS) or global
   - Verify if other systems are affected
   - Estimate user impact

2. **Communicate**:
   ```bash
   # Post status update
   curl -X POST "https://api.statuspage.io/v1/pages/PAGE_ID/incidents" \
     -H "Authorization: OAuth $STATUSPAGE_TOKEN" \
     -d '{"incident":{"name":"Pipeline Outage","status":"investigating"}}'
   ```

3. **Implement workaround**:
   - Switch to backup region
   - Use cached/historical data
   - Manual processing if needed

4. **Restore service**:
   - Fix root cause
   - Verify functionality
   - Update stakeholders

#### Data Corruption

1. **Isolate affected data**:
   ```bash
   # Move corrupted files to quarantine
   aws s3 mv s3://cdp-prod-potato-data-lake-xyz/silver/corrupted/ s3://cdp-prod-potato-quarantine-xyz/
   ```

2. **Restore from backup**:
   ```bash
   # Restore last known good state
   aws s3 sync s3://cdp-prod-potato-backup-xyz/silver/2024-09-22/ s3://cdp-prod-potato-data-lake-xyz/silver/
   ```

3. **Investigate root cause**:
   - Review code changes
   - Check data lineage in Marquez
   - Analyze processing logs

## 📈 Capacity Planning

### Growth Projections

| Metric | Current | 6 months | 1 year |
|--------|---------|----------|--------|
| Data Volume | 100 MB/day | 500 MB/day | 1 GB/day |
| Processing Time | 5 min | 15 min | 30 min |
| Storage | 10 GB | 100 GB | 500 GB |
| Users | 5 | 20 | 50 |

### Scaling Thresholds

- **CPU**: > 70% sustained → add compute resources
- **Memory**: > 80% usage → increase memory limits
- **Storage**: > 75% full → add storage, implement lifecycle policies
- **Network**: > 1 Gbps → upgrade bandwidth

### Cost Optimization

1. **S3 Lifecycle Policies**:
   ```json
   {
     "Rules": [{
       "Id": "potato-pipeline-lifecycle",
       "Filter": {"Prefix": "bronze/"},
       "Status": "Enabled",
       "Transitions": [
         {"Days": 30, "StorageClass": "STANDARD_IA"},
         {"Days": 90, "StorageClass": "GLACIER"}
       ]
     }]
   }
   ```

2. **Right-sizing compute**:
   - Monitor actual resource usage
   - Use spot instances for batch processing
   - Schedule scaling based on usage patterns

## 🔗 References

### Internal Documentation
- [System Architecture](system-diagram.png)
- [Security Notes](security-notes.md)  
- [API Documentation](../src/potato_pipeline/README.md)

### External Resources
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [Grafana Alerting](https://grafana.com/docs/grafana/latest/alerting/)
- [OpenLineage Spec](https://openlineage.io/spec)

### Emergency Contacts
- **AWS Support**: Case via AWS Console
- **GitHub Support**: [support@github.com](mailto:support@github.com)
- **On-Call Rotation**: [PagerDuty](https://example.pagerduty.com)

---

📅 **Last Updated**: 2024-09-23  
👥 **Maintained By**: Data Engineering Team  
📧 **Questions**: [team@example.com](mailto:team@example.com)