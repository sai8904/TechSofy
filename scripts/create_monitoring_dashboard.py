#!/usr/bin/env python3
import json
import requests
from datetime import datetime

def create_grafana_dashboard():
    """Create comprehensive Grafana dashboard"""
    
    dashboard = {
        "dashboard": {
            "id": None,
            "title": "Health Orchestrator - Production Monitoring",
            "tags": ["health-orchestrator", "production", "monitoring"],
            "timezone": "browser",
            "refresh": "30s",
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "panels": [
                {
                    "id": 1,
                    "title": "System Overview",
                    "type": "row",
                    "gridPos": {"h": 1, "w": 24, "x": 0, "y": 0}
                },
                {
                    "id": 2,
                    "title": "Services Status",
                    "type": "stat",
                    "gridPos": {"h": 8, "w": 6, "x": 0, "y": 1},
                    "targets": [
                        {
                            "expr": "health_orchestrator_services_healthy",
                            "legendFormat": "Healthy"
                        },
                        {
                            "expr": "health_orchestrator_services_unhealthy",
                            "legendFormat": "Unhealthy"
                        }
                    ]
                },
                {
                    "id": 3,
                    "title": "ML Model Performance",
                    "type": "gauge",
                    "gridPos": {"h": 8, "w": 6, "x": 6, "y": 1},
                    "targets": [
                        {
                            "expr": "health_orchestrator_ml_accuracy",
                            "legendFormat": "Accuracy"
                        }
                    ]
                },
                {
                    "id": 4,
                    "title": "Healing Actions",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 1},
                    "targets": [
                        {
                            "expr": "rate(health_orchestrator_healing_actions_total[5m])",
                            "legendFormat": "Actions per second"
                        }
                    ]
                },
                {
                    "id": 5,
                    "title": "System Resources",
                    "type": "row",
                    "gridPos": {"h": 1, "w": 24, "x": 0, "y": 10}
                },
                {
                    "id": 6,
                    "title": "CPU Usage",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 8, "x": 0, "y": 11},
                    "targets": [
                        {
                            "expr": "rate(process_cpu_seconds_total[5m]) * 100",
                            "legendFormat": "CPU %"
                        }
                    ]
                },
                {
                    "id": 7,
                    "title": "Memory Usage",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 8, "x": 8, "y": 11},
                    "targets": [
                        {
                            "expr": "process_resident_memory_bytes / 1024 / 1024",
                            "legendFormat": "Memory MB"
                        }
                    ]
                },
                {
                    "id": 8,
                    "title": "Network I/O",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 8, "x": 16, "y": 11},
                    "targets": [
                        {
                            "expr": "rate(network_receive_bytes_total[5m])",
                            "legendFormat": "Receive"
                        },
                        {
                            "expr": "rate(network_transmit_bytes_total[5m])",
                            "legendFormat": "Transmit"
                        }
                    ]
                }
            ]
        }
    }
    
    # Save dashboard configuration
    with open('config/grafana_dashboard.json', 'w') as f:
        json.dump(dashboard, f, indent=2)
    
    print("Grafana dashboard configuration created")
    return dashboard

if __name__ == "__main__":
    create_grafana_dashboard()