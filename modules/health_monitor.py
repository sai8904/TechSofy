"""
Health Monitoring Module
Collects metrics from Prometheus and analyzes service health
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from prometheus_api_client import PrometheusConnect
import numpy as np
import pandas as pd


@dataclass
class ServiceMetrics:
    """Data class to hold service metrics"""
    service_name: str
    namespace: str
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    error_rate: float
    response_time: float
    request_rate: float
    network_io: float
    availability: float


@dataclass
class HealthStatus:
    """Data class to hold health assessment results"""
    service_name: str
    namespace: str
    timestamp: datetime
    health_score: float  # 0-100 scale
    status: str  # HEALTHY, DEGRADED, UNHEALTHY, CRITICAL
    issues: List[str]
    metrics: ServiceMetrics


class HealthMonitor:
    """
    Health Monitoring Module
    Continuously monitors microservices health using Prometheus metrics
    """
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus = PrometheusConnect(url=prometheus_url, disable_ssl=True)
        self.logger = logging.getLogger(__name__)
        self.health_history: Dict[str, List[HealthStatus]] = {}
        
        # Health thresholds
        self.thresholds = {
            'cpu_critical': 85.0,
            'cpu_warning': 70.0,
            'memory_critical': 85.0,
            'memory_warning': 70.0,
            'error_rate_critical': 10.0,
            'error_rate_warning': 5.0,
            'response_time_critical': 2000.0,  # ms
            'response_time_warning': 1000.0,
            'availability_critical': 95.0,
            'availability_warning': 98.0
        }
    
    async def collect_service_metrics(self, service_name: str, namespace: str) -> Optional[ServiceMetrics]:
        """
        Collect metrics for a specific service from Prometheus
        """
        try:
            current_time = datetime.now()
            
            # Define Prometheus queries for common metrics
            queries = {
                'cpu_usage': f'rate(container_cpu_usage_seconds_total{{namespace="{namespace}",pod=~"{service_name}.*"}}[5m]) * 100',
                'memory_usage': f'(container_memory_usage_bytes{{namespace="{namespace}",pod=~"{service_name}.*"}} / container_spec_memory_limit_bytes{{namespace="{namespace}",pod=~"{service_name}.*"}}) * 100',
                'error_rate': f'rate(http_requests_total{{namespace="{namespace}",service="{service_name}",status=~"5.."}}[5m]) / rate(http_requests_total{{namespace="{namespace}",service="{service_name}"}}[5m]) * 100',
                'response_time': f'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{namespace="{namespace}",service="{service_name}"}}[5m])) * 1000',
                'request_rate': f'rate(http_requests_total{{namespace="{namespace}",service="{service_name}"}}[5m])',
                'network_io': f'rate(container_network_receive_bytes_total{{namespace="{namespace}",pod=~"{service_name}.*"}}[5m]) + rate(container_network_transmit_bytes_total{{namespace="{namespace}",pod=~"{service_name}.*"}}[5m])',
                'availability': f'up{{namespace="{namespace}",service="{service_name}"}} * 100'
            }
            
            metrics_data = {}
            
            # Execute queries and collect results
            for metric_name, query in queries.items():
                try:
                    result = self.prometheus.custom_query(query)
                    if result:
                        # Get the most recent value
                        value = float(result[0]['value'][1])
                        metrics_data[metric_name] = value
                    else:
                        # Set default values if query returns no data
                        metrics_data[metric_name] = self._get_default_value(metric_name)
                except Exception as e:
                    self.logger.warning(f"Failed to collect {metric_name} for {service_name}: {e}")
                    metrics_data[metric_name] = self._get_default_value(metric_name)
            
            return ServiceMetrics(
                service_name=service_name,
                namespace=namespace,
                timestamp=current_time,
                cpu_usage=metrics_data.get('cpu_usage', 0.0),
                memory_usage=metrics_data.get('memory_usage', 0.0),
                error_rate=metrics_data.get('error_rate', 0.0),
                response_time=metrics_data.get('response_time', 0.0),
                request_rate=metrics_data.get('request_rate', 0.0),
                network_io=metrics_data.get('network_io', 0.0),
                availability=metrics_data.get('availability', 100.0)
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics for {service_name}: {e}")
            return None
    
    def _get_default_value(self, metric_name: str) -> float:
        """Get default value for metrics when data is unavailable"""
        defaults = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'error_rate': 0.0,
            'response_time': 100.0,
            'request_rate': 0.0,
            'network_io': 0.0,
            'availability': 100.0
        }
        return defaults.get(metric_name, 0.0)
    
    def assess_health(self, metrics: ServiceMetrics) -> HealthStatus:
        """
        Assess service health based on collected metrics
        """
        issues = []
        scores = []
        
        # CPU Usage Assessment
        if metrics.cpu_usage >= self.thresholds['cpu_critical']:
            issues.append(f"Critical CPU usage: {metrics.cpu_usage:.1f}%")
            scores.append(20)
        elif metrics.cpu_usage >= self.thresholds['cpu_warning']:
            issues.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
            scores.append(60)
        else:
            scores.append(100)
        
        # Memory Usage Assessment
        if metrics.memory_usage >= self.thresholds['memory_critical']:
            issues.append(f"Critical memory usage: {metrics.memory_usage:.1f}%")
            scores.append(20)
        elif metrics.memory_usage >= self.thresholds['memory_warning']:
            issues.append(f"High memory usage: {metrics.memory_usage:.1f}%")
            scores.append(60)
        else:
            scores.append(100)
        
        # Error Rate Assessment
        if metrics.error_rate >= self.thresholds['error_rate_critical']:
            issues.append(f"Critical error rate: {metrics.error_rate:.1f}%")
            scores.append(10)
        elif metrics.error_rate >= self.thresholds['error_rate_warning']:
            issues.append(f"High error rate: {metrics.error_rate:.1f}%")
            scores.append(50)
        else:
            scores.append(100)
        
        # Response Time Assessment
        if metrics.response_time >= self.thresholds['response_time_critical']:
            issues.append(f"Critical response time: {metrics.response_time:.1f}ms")
            scores.append(30)
        elif metrics.response_time >= self.thresholds['response_time_warning']:
            issues.append(f"High response time: {metrics.response_time:.1f}ms")
            scores.append(70)
        else:
            scores.append(100)
        
        # Availability Assessment
        if metrics.availability <= self.thresholds['availability_critical']:
            issues.append(f"Low availability: {metrics.availability:.1f}%")
            scores.append(10)
        elif metrics.availability <= self.thresholds['availability_warning']:
            issues.append(f"Degraded availability: {metrics.availability:.1f}%")
            scores.append(60)
        else:
            scores.append(100)
        
        # Calculate overall health score
        health_score = np.mean(scores)
        
        # Determine status
        if health_score >= 80:
            status = "HEALTHY"
        elif health_score >= 60:
            status = "DEGRADED"
        elif health_score >= 30:
            status = "UNHEALTHY"
        else:
            status = "CRITICAL"
        
        health_status = HealthStatus(
            service_name=metrics.service_name,
            namespace=metrics.namespace,
            timestamp=metrics.timestamp,
            health_score=health_score,
            status=status,
            issues=issues,
            metrics=metrics
        )
        
        # Store in history
        service_key = f"{metrics.namespace}/{metrics.service_name}"
        if service_key not in self.health_history:
            self.health_history[service_key] = []
        
        self.health_history[service_key].append(health_status)
        
        # Keep only last 100 records per service
        if len(self.health_history[service_key]) > 100:
            self.health_history[service_key] = self.health_history[service_key][-100:]
        
        return health_status
    
    async def monitor_services(self, services: List[Tuple[str, str]]) -> Dict[str, HealthStatus]:
        """
        Monitor multiple services concurrently
        Args:
            services: List of (service_name, namespace) tuples
        Returns:
            Dictionary mapping service keys to health status
        """
        tasks = []
        for service_name, namespace in services:
            task = asyncio.create_task(self.collect_service_metrics(service_name, namespace))
            tasks.append((service_name, namespace, task))
        
        results = {}
        for service_name, namespace, task in tasks:
            try:
                metrics = await task
                if metrics:
                    health_status = self.assess_health(metrics)
                    service_key = f"{namespace}/{service_name}"
                    results[service_key] = health_status
                    
                    self.logger.info(f"Health check for {service_key}: {health_status.status} (Score: {health_status.health_score:.1f})")
                    if health_status.issues:
                        self.logger.warning(f"Issues found: {', '.join(health_status.issues)}")
                        
            except Exception as e:
                self.logger.error(f"Error monitoring {service_name} in {namespace}: {e}")
        
        return results
    
    def get_health_history(self, service_name: str, namespace: str, hours: int = 24) -> List[HealthStatus]:
        """
        Get health history for a service
        """
        service_key = f"{namespace}/{service_name}"
        if service_key not in self.health_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            status for status in self.health_history[service_key]
            if status.timestamp >= cutoff_time
        ]
    
    def get_metrics_dataframe(self, service_name: str, namespace: str, hours: int = 24) -> pd.DataFrame:
        """
        Get metrics as pandas DataFrame for ML processing
        """
        history = self.get_health_history(service_name, namespace, hours)
        
        if not history:
            return pd.DataFrame()
        
        data = []
        for status in history:
            data.append({
                'timestamp': status.timestamp,
                'service_name': status.service_name,
                'namespace': status.namespace,
                'health_score': status.health_score,
                'cpu_usage': status.metrics.cpu_usage,
                'memory_usage': status.metrics.memory_usage,
                'error_rate': status.metrics.error_rate,
                'response_time': status.metrics.response_time,
                'request_rate': status.metrics.request_rate,
                'network_io': status.metrics.network_io,
                'availability': status.metrics.availability,
                'status': status.status
            })
        
        return pd.DataFrame(data)


# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize health monitor
    monitor = HealthMonitor("http://localhost:9090")
    
    # Example services to monitor
    services = [
        ("user-service", "default"),
        ("order-service", "default"),
        ("payment-service", "default"),
        ("inventory-service", "default")
    ]
    
    async def main():
        # Monitor services
        results = await monitor.monitor_services(services)
        
        # Print results
        for service_key, health_status in results.items():
            print(f"\n{service_key}:")
            print(f"  Status: {health_status.status}")
            print(f"  Health Score: {health_status.health_score:.1f}")
            print(f"  Issues: {', '.join(health_status.issues) if health_status.issues else 'None'}")
            print(f"  CPU: {health_status.metrics.cpu_usage:.1f}%")
            print(f"  Memory: {health_status.metrics.memory_usage:.1f}%")
            print(f"  Error Rate: {health_status.metrics.error_rate:.1f}%")
            print(f"  Response Time: {health_status.metrics.response_time:.1f}ms")
    
    # Run monitoring
    asyncio.run(main())