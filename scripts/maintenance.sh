#!/bin/bash
# Regular maintenance tasks

echo "Running maintenance tasks..."

# Update ML models
python scripts/retrain_model.py

# Clean up old logs
find logs/ -name "*.log" -mtime +30 -delete

# Optimize database
python scripts/optimize_database.py

# Check for security updates
python scripts/security_check.py

# Generate health report
python scripts/generate_health_report.py

echo "Maintenance completed!"