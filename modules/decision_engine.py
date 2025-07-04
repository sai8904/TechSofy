"""
Decision Engine Module
Determines the optimal healing action based on failure predictions
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import yaml
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

@dataclass
class HealingAction:
    """Data class for healing actions"""
    action_type: str  # RESTART, SCALE_UP, SCALE_DOWN, REROUTE, ROLLBACK
    service_name: str
    namespace: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float
    parameters: Dict  # Action-specific parameters


class DecisionEngine:
    """
    Decision Engine Module
    Determines the best healing action based on failure predictions and service dependencies
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.action_model = None
        self.service_dependencies = self.config.get('service_dependencies', {})
        
        # Initialize action model
        self._initialize_action_model()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}
    
    def _initialize_action_model(self):
        """Initialize the action recommendation model"""
        # This is a simple model that can be trained with historical data
        # For now, we'll use a basic implementation
        self.action_model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
    
    def determine_action(self, 
                        failure_predictions: List[Dict],
                        current_metrics: Dict,
                        service_status: Dict) -> List[HealingAction]:
        """
        Determine the best healing actions based on multiple inputs
        Args:
            failure_predictions: List of failure predictions from FailureDetector
            current_metrics: Current service metrics from HealthMonitor
            service_status: Current status of all services
        Returns:
            List of recommended healing actions
        """
        actions = []
        
        for prediction in failure_predictions:
            service_name = prediction['service_name']
            namespace = prediction['namespace']
            
            # Get default action based on failure type
            default_action = self._get_default_action(prediction)
            
            # Consider service dependencies
            dependent_services = self.service_dependencies.get(service_name, [])
            dependent_status = [
                service_status.get(f"{namespace}/{dep}", {}).get('status', 'HEALTHY')
                for dep in dependent_services
            ]
            
            # Adjust action based on dependencies
            action = self._adjust_action_for_dependencies(
                default_action,
                dependent_services,
                dependent_status
            )
            
            # Add to actions list
            if action:
                actions.append(action)
        
        # Prioritize actions
        return self._prioritize_actions(actions)
    
    def _get_default_action(self, prediction: Dict) -> HealingAction:
        """Get default action based on failure type"""
        failure_type = prediction['failure_type']
        severity = self._determine_severity(prediction)
        
        action_map = {
            'RESOURCE_EXHAUSTION': {
                'action_type': 'SCALE_UP',
                'parameters': {'scale_factor': 1.5}
            },
            'HIGH_ERROR_RATE': {
                'action_type': 'RESTART',
                'parameters': {}
            },
            'PERFORMANCE_DEGRADATION': {
                'action_type': 'SCALE_UP',
                'parameters': {'scale_factor': 2.0}
            },
            'AVAILABILITY_LOSS': {
                'action_type': 'REROUTE',
                'parameters': {'timeout': '30s'}
            }
        }
        
        action_config = action_map.get(failure_type, {
            'action_type': 'RESTART',
            'parameters': {}
        })
        
        return HealingAction(
            action_type=action_config['action_type'],
            service_name=prediction['service_name'],
            namespace=prediction['namespace'],
            severity=severity,
            confidence=prediction['confidence'],
            parameters=action_config['parameters']
        )
    
    def _determine_severity(self, prediction: Dict) -> str:
        """Determine severity based on prediction confidence and probability"""
        prob = prediction['failure_probability']
        conf = prediction['confidence']
        
        if prob > 0.8 and conf > 0.8:
            return 'CRITICAL'
        elif prob > 0.6 and conf > 0.7:
            return 'HIGH'
        elif prob > 0.4 and conf > 0.6:
            return 'MEDIUM'
        return 'LOW'
    
    def _adjust_action_for_dependencies(self, 
                                      action: HealingAction,
                                      dependent_services: List[str],
                                      dependent_status: List[str]) -> Optional[HealingAction]:
        """
        Adjust action based on dependent services status
        Returns None if action should be skipped
        """
        # Skip if critical dependencies are unhealthy
        if any(status in ['UNHEALTHY', 'CRITICAL'] for status in dependent_status):
            if action.action_type in ['SCALE_DOWN', 'RESTART']:
                self.logger.warning(
                    f"Skipping {action.action_type} for {action.service_name} "
                    f"due to unhealthy dependencies"
                )
                return None
        
        # For scale-up actions, check if dependencies can handle increased load
        if action.action_type == 'SCALE_UP':
            for dep, status in zip(dependent_services, dependent_status):
                if status in ['DEGRADED', 'UNHEALTHY']:
                    action.parameters['scale_factor'] = min(
                        1.2, 
                        action.parameters.get('scale_factor', 1.5)
                    )
                    self.logger.info(
                        f"Reducing scale factor for {action.service_name} "
                        f"due to degraded dependency {dep}"
                    )
        
        return action
    
    def _prioritize_actions(self, actions: List[HealingAction]) -> List[HealingAction]:
        """Prioritize actions based on severity and confidence"""
        severity_order = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        
        return sorted(
            actions,
            key=lambda x: (severity_order[x.severity], x.confidence),
            reverse=True
        )
    
    def train_action_model(self, historical_data: List[Dict]):
        """
        Train the action recommendation model with historical data
        Args:
            historical_data: List of dicts containing:
                - features: metrics and failure predictions
                - label: action taken
                - outcome: success/failure of action
        """
        try:
            # Prepare features and labels
            X = []
            y = []
            
            for record in historical_data:
                features = self._extract_features(record['features'])
                label = self._encode_action(record['label'])
                X.append(features)
                y.append(label)
            
            # Train model
            self.action_model.fit(X, y)
            self.logger.info("Action recommendation model trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training action model: {e}")
    
    def _extract_features(self, features: Dict) -> List[float]:
        """Extract numerical features from metrics and predictions"""
        # This should be customized based on your actual data
        return [
            features.get('cpu_usage', 0),
            features.get('memory_usage', 0),
            features.get('error_rate', 0),
            features.get('response_time', 0),
            features.get('failure_probability', 0),
            features.get('confidence', 0)
        ]
    
    def _encode_action(self, action: Dict) -> int:
        """Encode action type as integer"""
        action_types = ['RESTART', 'SCALE_UP', 'SCALE_DOWN', 'REROUTE', 'ROLLBACK']
        return action_types.index(action.get('action_type', 'RESTART'))


# Example usage
if __name__ == "__main__":
    import logging
    from datetime import datetime
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize decision engine
    engine = DecisionEngine()
    
    # Create sample failure prediction
    sample_prediction = {
        'service_name': 'user-service',
        'namespace': 'default',
        'failure_type': 'RESOURCE_EXHAUSTION',
        'failure_probability': 0.85,
        'confidence': 0.9,
        'timestamp': datetime.now()
    }
    
    # Create sample current metrics
    sample_metrics = {
        'default/user-service': {
            'cpu_usage': 92.5,
            'memory_usage': 88.3,
            'status': 'CRITICAL'
        },
        'default/order-service': {
            'status': 'HEALTHY'
        }
    }
    
    # Determine action
    actions = engine.determine_action(
        [sample_prediction],
        sample_metrics,
        sample_metrics
    )
    
    print("\nRecommended Actions:")
    for action in actions:
        print(f"  Service: {action.service_name}")
        print(f"    Action: {action.action_type}")
        print(f"    Severity: {action.severity}")
        print(f"    Parameters: {action.parameters}")