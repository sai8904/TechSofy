"""
Main Application
Orchestrates the health monitoring and healing process
"""

import asyncio
import logging
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd
from modules.health_monitor import HealthMonitor
from modules.failure_detector import FailureDetector
from modules.decision_engine import DecisionEngine
from modules.orchestrator import Orchestrator

class HealthOrchestrator:
    """
    Main orchestrator class that ties all modules together
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Configure logging
        self._configure_logging()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize modules
        self.health_monitor = HealthMonitor(
            prometheus_url=self.config.get('prometheus_url', 'http://localhost:9090')
        )
        self.failure_detector = FailureDetector()
        self.decision_engine = DecisionEngine(config_path)
        self.orchestrator = Orchestrator(config_path)
        
        # Load ML models if available
        self.failure_detector.load_models()
        
        # Services to monitor
        self.services = self.config.get('services', [])
        
        # Track action history
        self.action_history: List[Dict] = []
    
    def _configure_logging(self):
        """Configure logging from file"""
        try:
            logging.config.fileConfig('config/logging.conf')
        except:
            # Fallback basic config
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}
    
    async def run_monitoring_cycle(self):
        """Run one complete monitoring and healing cycle"""
        try:
            self.logger.info("Starting monitoring cycle")
            start_time = datetime.now()
            
            # 1. Collect metrics and assess health
            health_status = await self.health_monitor.monitor_services(self.services)
            
            # 2. Get metrics history for ML processing
            metrics_history = self._get_metrics_history()
            
            # 3. Detect and predict failures
            immediate_failures = self.failure_detector.detect_immediate_failures(metrics_history)
            failure_predictions = self.failure_detector.predict_failure(metrics_history)
            anomalies = self.failure_detector.detect_anomalies(metrics_history)
            
            # Log findings
            self._log_findings(immediate_failures, failure_predictions, anomalies)
            
            # 4. Determine healing actions
            actions = self.decision_engine.determine_action(
                failure_predictions + immediate_failures,
                metrics_history.to_dict('records'),
                health_status
            )
            
            # 5. Execute actions
            self._execute_actions(actions)
            
            # 6. Train/update models periodically
            if start_time.minute % 30 == 0:  # Every 30 minutes
                self._train_models(metrics_history)
            
            cycle_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Completed monitoring cycle in {cycle_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error in monitoring cycle: {e}")
    
    def _get_metrics_history(self) -> pd.DataFrame:
        """Get metrics history for all services"""
        all_metrics = []
        
        for service_name, namespace in self.services:
            history = self.health_monitor.get_metrics_dataframe(
                service_name, 
                namespace,
                hours=self.config.get('history_hours', 24)
            )
            if not history.empty:
                all_metrics.append(history)
        
        if all_metrics:
            return pd.concat(all_metrics, ignore_index=True)
        return pd.DataFrame()
    
    def _log_findings(self, 
                     immediate_failures: List[Dict],
                     failure_predictions: List[Dict],
                     anomalies: List[Dict]):
        """Log all detected issues"""
        if immediate_failures:
            self.logger.warning(f"Detected {len(immediate_failures)} immediate failures")
            for failure in immediate_failures:
                self.logger.warning(
                    f"Service {failure['service_name']} failures: "
                    f"{', '.join([f['type'] for f in failure['failures']])}"
                )
        
        if failure_predictions:
            self.logger.warning(f"Generated {len(failure_predictions)} failure predictions")
            for pred in failure_predictions:
                self.logger.warning(
                    f"Service {pred.service_name} predicted {pred.failure_type} "
                    f"with probability {pred.failure_probability:.2f}"
                )
        
        if anomalies:
            anomaly_count = sum(1 for a in anomalies if a.is_anomaly)
            if anomaly_count > 0:
                self.logger.warning(f"Detected {anomaly_count} anomalies")
                for anomaly in anomalies:
                    if anomaly.is_anomaly:
                        self.logger.warning(
                            f"Service {anomaly.service_name} anomaly detected "
                            f"(score: {anomaly.anomaly_score:.2f})"
                        )
    
    def _execute_actions(self, actions: List[Dict]):
        """Execute healing actions with rate limiting"""
        max_actions = self.config.get('max_actions_per_cycle', 3)
        executed = 0
        
        for action in actions[:max_actions]:
            try:
                # Convert action to dict if it's a dataclass
                if hasattr(action, '__dataclass_fields__'):
                    action_dict = {
                        'action_type': action.action_type,
                        'service_name': action.service_name,
                        'namespace': action.namespace,
                        'parameters': action.parameters
                    }
                else:
                    action_dict = action
                
                # Execute action
                success = self.orchestrator.execute_action(action_dict)
                
                # Record action
                self.action_history.append({
                    'timestamp': datetime.now(),
                    'action': action_dict,
                    'success': success
                })
                
                if success:
                    executed += 1
                    self.logger.info(
                        f"Successfully executed {action_dict['action_type']} "
                        f"on {action_dict['service_name']}"
                    )
                else:
                    self.logger.error(
                        f"Failed to execute {action_dict['action_type']} "
                        f"on {action_dict['service_name']}"
                    )
                
            except Exception as e:
                self.logger.error(f"Error executing action: {e}")
        
        if executed > 0:
            self.logger.info(f"Executed {executed} healing actions this cycle")
    
    def _train_models(self, metrics_history: pd.DataFrame):
        """Train/update ML models"""
        try:
            self.logger.info("Training/updating ML models...")
            
            # Train failure detection models
            self.failure_detector.train_classification_model(metrics_history)
            self.failure_detector.train_time_series_model(metrics_history)
            self.failure_detector.train_anomaly_detector(metrics_history)
            
            # Save models
            self.failure_detector.save_models()
            
            # Train action recommendation model with historical data
            if self.action_history:
                self.decision_engine.train_action_model(self.action_history)
            
            self.logger.info("Model training completed")
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
    
    async def run_continuously(self, interval: int = 60):
        """Run monitoring continuously with specified interval (seconds)"""
        self.logger.info(f"Starting health orchestrator with {interval} second interval")
        
        while True:
            await self.run_monitoring_cycle()
            await asyncio.sleep(interval)


# Example configuration
DEFAULT_CONFIG = """
# Prometheus configuration
prometheus_url: "http://localhost:9090"

# Monitoring interval in seconds
monitoring_interval: 60

# Number of hours of history to keep
history_hours: 24

# Maximum actions to take per cycle
max_actions_per_cycle: 3

# Services to monitor (service_name, namespace)
services:
  - ["user-service", "default"]
  - ["order-service", "default"]
  - ["payment-service", "default"]

# Service dependencies
service_dependencies:
  order-service: ["user-service", "payment-service"]
  payment-service: ["user-service"]
"""

if __name__ == "__main__":
    import os
    from pathlib import Path
    
    # Create config directory if not exists
    os.makedirs("config", exist_ok=True)
    
    # Write default config if not exists
    config_path = "config/config.yaml"
    if not Path(config_path).exists():
        with open(config_path, "w") as f:
            f.write(DEFAULT_CONFIG)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Initialize and run orchestrator
    orchestrator = HealthOrchestrator(config_path)
    
    try:
        asyncio.run(orchestrator.run_continuously(
            interval=orchestrator.config.get('monitoring_interval', 60)
        ))
    except KeyboardInterrupt:
        print("\nShutting down health orchestrator...")