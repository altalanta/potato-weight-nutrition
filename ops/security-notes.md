# Security Documentation

## Overview

This document outlines the security posture, data handling practices, and compliance considerations for the potato-weight-nutrition pipeline.

## 🚨 PHI/PII Policy

### Current Status: NO PHI/PII

**This pipeline currently processes NO protected health information (PHI) or personally identifiable information (PII).**

- All data in this repository is **synthetic** or **anonymized**
- No real patient data, medical records, or personal identifiers
- For demonstration and development purposes only

### If Using Real Data (Future Considerations)

If this pipeline were to process real health data:

1. **HIPAA Compliance**: Would require BAA agreements, encryption, audit logs
2. **Data Minimization**: Only collect necessary data points
3. **Access Controls**: Role-based permissions, least privilege
4. **Retention Policies**: Automated data deletion/archival
5. **Breach Response**: Incident response procedures

## 🔐 Security Architecture

### Authentication & Authorization

#### GitHub OIDC (Recommended)
- **No long-lived AWS keys** stored in GitHub
- Short-lived tokens via OpenID Connect
- Repository-specific trust policies
- Environment-based access controls

```yaml
# GitHub Environments configured:
dev:
  - AWS_ROLE_ARN: arn:aws:iam::ACCOUNT:role/cdp-dev-github-oidc-role
  - Restricted to: main branch, specific workflows
  
prod:
  - AWS_ROLE_ARN: arn:aws:iam::ACCOUNT:role/cdp-prod-github-oidc-role  
  - Restricted to: manual approval, tagged releases
```

#### AWS IAM Policies (Least Privilege)
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::cdp-*-potato-data-lake-*",
        "arn:aws:s3:::cdp-*-potato-data-lake-*/*"
      ]
    },
    {
      "Effect": "Allow", 
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:Query"
      ],
      "Resource": "arn:aws:s3:::cdp-*-mlflow-backend"
    }
  ]
}
```

### Data Encryption

#### At Rest
- **S3 Buckets**: AES-256 server-side encryption (SSE-S3)
- **DynamoDB**: Encryption at rest enabled by default
- **EBS Volumes**: If using EC2, enable EBS encryption
- **RDS**: If using managed database, enable TDE

#### In Transit
- **HTTPS**: All API communications use TLS 1.2+
- **VPC Endpoints**: For private AWS service communication
- **Docker Networks**: Encrypted overlay networks for container communication

#### Application Level
- **Environment Variables**: No secrets in `.env` files committed to git
- **Secrets Management**: Use AWS Secrets Manager or GitHub Secrets
- **Connection Strings**: Parameterized with environment-specific values

### Network Security

#### AWS VPC Configuration (Production)
```yaml
VPC:
  CIDR: 10.0.0.0/16
  Subnets:
    Private: 10.0.1.0/24, 10.0.2.0/24  # Application tier
    Public: 10.0.100.0/24, 10.0.101.0/24  # Load balancer tier
  
Security Groups:
  Application:
    Inbound: Port 443 from ALB only
    Outbound: HTTPS to AWS services
  Database:
    Inbound: Port 5432 from application tier only
```

#### Local Development
- **Docker Networks**: Isolated networks for each service stack
- **Port Binding**: Services only exposed on localhost
- **Firewall**: Host firewall rules for additional protection

## 🛡️ Data Classification & Handling

### Data Categories

| Category | Description | Examples | Security Level |
|----------|-------------|----------|----------------|
| **Public** | Non-sensitive analysis results | Aggregate statistics, plots | Standard |
| **Internal** | Development/test data | Synthetic datasets, configs | Enhanced |
| **Confidential** | Real study data (if used) | Raw participant data | Restricted |
| **Restricted** | Identifiable information | PII, PHI (not in scope) | Maximum |

### Data Redaction & Masking

#### Text Data Processing
```python
# Example redaction patterns for food descriptions
REDACTION_PATTERNS = {
    'phone_numbers': r'\b\d{3}-\d{3}-\d{4}\b',
    'email_addresses': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'social_security': r'\b\d{3}-\d{2}-\d{4}\b',
    'credit_cards': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
}

def redact_sensitive_text(text: str) -> str:
    """Remove potential PII from free-text fields."""
    for pattern_name, pattern in REDACTION_PATTERNS.items():
        text = re.sub(pattern, f'[REDACTED_{pattern_name.upper()}]', text)
    return text
```

#### Identifier Hashing
```python
# Example ID anonymization (for real data scenarios)
import hashlib
import hmac

def anonymize_id(participant_id: str, salt: str) -> str:
    """Create consistent anonymous ID from participant ID."""
    return hmac.new(
        salt.encode(), 
        participant_id.encode(), 
        hashlib.sha256
    ).hexdigest()[:16]
```

## 🔒 Access Control Framework

### Role-Based Access Control (RBAC)

#### Reader Role
- **Purpose**: View analysis results, dashboards
- **Permissions**: 
  - S3: `s3:GetObject` on `silver/` and `gold/` prefixes
  - Grafana: Viewer permissions
  - Marquez: Read-only lineage access

#### Writer Role  
- **Purpose**: Run pipeline, update silver layer
- **Permissions**:
  - S3: `s3:GetObject`, `s3:PutObject` on `bronze/` and `silver/`
  - DynamoDB: Read/write MLflow experiments
  - CloudWatch: Write logs

#### Admin Role
- **Purpose**: Manage infrastructure, deploy changes
- **Permissions**:
  - Full access to pipeline resources
  - Terraform state management
  - User/role management

### Multi-Environment Access

```yaml
Development:
  - Relaxed controls for experimentation
  - Synthetic data only
  - Individual developer access

Staging: 
  - Production-like security controls
  - Anonymized data subsets
  - Team lead approval required

Production:
  - Maximum security controls
  - Real data (if applicable)
  - Change management process
```

## 🚨 Security Monitoring & Alerting

### AWS CloudTrail
- **All API calls** logged and monitored
- **Data access patterns** tracked for anomalies
- **Cross-account access** detected and alerted

### Application Security Monitoring
```python
# Example security event logging
import structlog

security_logger = structlog.get_logger("security")

def log_data_access(user_id: str, dataset: str, action: str):
    """Log all data access for security audit."""
    security_logger.info(
        "data_access",
        user_id=user_id,
        dataset=dataset, 
        action=action,
        timestamp=datetime.utcnow().isoformat(),
        source_ip=get_client_ip()
    )
```

### Alerting Rules
- **Failed authentication attempts** > 5 in 5 minutes
- **Unusual data access patterns** (e.g., bulk downloads)
- **Infrastructure changes** outside maintenance windows
- **Security group modifications**

## 📋 Compliance & Audit

### Security Scanning (Automated)

#### Code Security
- **Bandit**: Static analysis for Python security issues
- **pip-audit**: Dependency vulnerability scanning  
- **Trufflehog**: Secret detection in code/history
- **Safety**: Known security vulnerabilities

#### Infrastructure Security
- **Checkov**: Terraform security best practices
- **Scout Suite**: AWS configuration assessment
- **AWS Config**: Compliance monitoring

### Audit Documentation

#### Required Records
1. **Access Logs**: Who accessed what data when
2. **Change Logs**: All pipeline/infrastructure modifications
3. **Security Events**: Authentication failures, policy violations
4. **Data Processing**: ETL runs, transformations, exports

#### Retention Policies
- **Security logs**: 2 years minimum
- **Access logs**: 1 year minimum  
- **Audit reports**: 7 years (for compliance)
- **Incident reports**: Permanent retention

## 🔧 Security Tools & Integrations

### GitHub Security Features
- **Dependabot**: Automated dependency updates
- **Code Scanning**: CodeQL analysis for vulnerabilities
- **Secret Scanning**: Detect accidentally committed secrets
- **Security Advisories**: Vulnerability disclosure process

### AWS Security Services
- **GuardDuty**: Threat detection and monitoring
- **Security Hub**: Centralized security findings
- **Inspector**: Application vulnerability assessment
- **Macie**: Data classification and protection

### Third-Party Integrations
- **Snyk**: Dependency and container scanning
- **Vault**: Secrets management (if using HashiCorp stack)
- **SIEM**: Security information and event management

## 🆘 Incident Response

### Security Incident Categories

#### Category 1: Data Breach
- **Response Time**: < 1 hour
- **Actions**: Isolate systems, assess scope, notify stakeholders
- **Communication**: Legal, compliance, affected parties

#### Category 2: Unauthorized Access
- **Response Time**: < 4 hours  
- **Actions**: Revoke access, audit trails, strengthen controls
- **Communication**: Security team, system owners

#### Category 3: Infrastructure Compromise
- **Response Time**: < 2 hours
- **Actions**: Isolate affected resources, forensic analysis
- **Communication**: DevOps, security, management

### Response Playbooks

#### Data Access Anomaly
```bash
# 1. Identify scope
aws cloudtrail lookup-events --lookup-attributes AttributeKey=EventName,AttributeValue=GetObject

# 2. Check S3 access logs
aws s3api get-bucket-logging --bucket cdp-prod-potato-data-lake-xyz

# 3. Review IAM activity
aws iam get-credential-report

# 4. Revoke compromised credentials
aws iam update-access-key --access-key-id AKIA... --status Inactive
```

#### Secret Exposure
```bash
# 1. Rotate immediately
aws iam create-access-key --user-name pipeline-user

# 2. Update all dependent systems
# 3. Review access logs for misuse
# 4. Strengthen secret management
```

## 📚 Security Training & Awareness

### Developer Security Guidelines

#### Secure Coding Practices
- **Input validation**: Sanitize all external inputs
- **SQL injection prevention**: Use parameterized queries
- **XSS prevention**: Escape output, validate inputs
- **Authentication**: Never store credentials in code

#### Secret Management
```python
# ❌ Bad: Hardcoded secrets
DATABASE_URL = "postgresql://user:password@host:5432/db"

# ✅ Good: Environment variables
DATABASE_URL = os.environ["DATABASE_URL"]

# ✅ Better: Secrets manager
import boto3
secrets_client = boto3.client('secretsmanager')
secret = secrets_client.get_secret_value(SecretId='db-credentials')
DATABASE_URL = json.loads(secret['SecretString'])['url']
```

### Security Checklists

#### Pre-Deployment Security Review
- [ ] No secrets in code or configuration files
- [ ] All data access properly authenticated/authorized  
- [ ] Encryption enabled for data at rest and in transit
- [ ] Network security groups configured (least privilege)
- [ ] Logging and monitoring enabled
- [ ] Incident response procedures documented
- [ ] Security scanning results reviewed and addressed

#### Production Deployment
- [ ] Infrastructure as Code (Terraform) used
- [ ] Change management process followed
- [ ] Security controls tested in staging
- [ ] Rollback plan prepared
- [ ] Monitoring/alerting verified
- [ ] Documentation updated

## 🔗 References & Resources

### Regulatory Frameworks
- **HIPAA**: Health Insurance Portability and Accountability Act
- **GDPR**: General Data Protection Regulation
- **CCPA**: California Consumer Privacy Act
- **SOX**: Sarbanes-Oxley Act

### Security Standards
- **NIST Cybersecurity Framework**: Risk management approach
- **ISO 27001**: Information security management systems
- **OWASP Top 10**: Web application security risks
- **CIS Controls**: Critical security controls

### AWS Security Best Practices
- [AWS Well-Architected Security Pillar](https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/)
- [AWS Security Best Practices](https://aws.amazon.com/architecture/security-identity-compliance/)
- [AWS Shared Responsibility Model](https://aws.amazon.com/compliance/shared-responsibility-model/)

---

📧 **Security Contact**: [security@example.com](mailto:security@example.com)  
🚨 **Security Incidents**: [incidents@example.com](mailto:incidents@example.com)  
📋 **Last Updated**: 2024-09-23