"""
Failure Detection & Prediction Module
Detects failures using rule-based thresholds and predicts future failures using ML
"""

import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import warnings
warnings.filterwarnings('ignore')


@dataclass
class FailurePrediction:
    """Data class for failure prediction results"""
    service_name: str
    namespace: str
    timestamp: datetime
    failure_probability: float
    failure_type: str  # RESOURCE_EXHAUSTION, HIGH_ERROR_RATE, PERFORMANCE_DEGRADATION, AVAILABILITY_LOSS
    predicted_failure_time: Optional[datetime]
    confidence: float
    contributing_factors: List[str]


@dataclass
class AnomalyDetection:
    """Data class for anomaly detection results"""
    service_name: str
    namespace: str
    timestamp: datetime
    is_anomaly: bool
    anomaly_score: float
    anomalous_metrics: List[str]


class FailureDetector:
    """
    Failure Detection & Prediction Module
    Implements both rule-based detection and ML-based prediction
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Rule-based thresholds for immediate failure detection
        self.failure_thresholds = {
            'cpu_critical': 90.0,
            'memory_critical': 90.0,
            'error_rate_critical': 15.0,
            'response_time_critical': 5000.0,  # ms
            'availability_critical': 90.0,
            'consecutive_failures': 3  # Number of consecutive unhealthy checks
        }
        
        # ML Models
        self.classification_model = None
        self.time_series_model = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        
        # Feature columns for ML
        self.feature_columns = [
            'cpu_usage', 'memory_usage', 'error_rate', 'response_time',
            'request_rate', 'network_io', 'availability'
        ]
        
        # Historical failure data
        self.failure_history: Dict[str, List[bool]] = {}
        
    def detect_immediate_failures(self, metrics_data: pd.DataFrame) -> List[Dict]:
        """
        Detect immediate failures using rule-based thresholds
        """
        failures = []
        
        for _, row in metrics_data.iterrows():
            service_key = f"{row['namespace']}/{row['service_name']}"
            detected_failures = []
            
            # CPU threshold check
            if row['cpu_usage'] >= self.failure_thresholds['cpu_critical']:
                detected_failures.append({
                    'type': 'RESOURCE_EXHAUSTION',
                    'severity': 'CRITICAL',
                    'message': f"CPU usage {row['cpu_usage']:.1f}% exceeds critical threshold",
                    'metric': 'cpu_usage',
                    'value': row['cpu_usage']
                })
            
            # Memory threshold check
            if row['memory_usage'] >= self.failure_thresholds['memory_critical']:
                detected_failures.append({
                    'type': 'RESOURCE_EXHAUSTION',
                    'severity': 'CRITICAL',
                    'message': f"Memory usage {row['memory_usage']:.1f}% exceeds critical threshold",
                    'metric': 'memory_usage',
                    'value': row['memory_usage']
                })
            
            # Error rate threshold check
            if row['error_rate'] >= self.failure_thresholds['error_rate_critical']:
                detected_failures.append({
                    'type': 'HIGH_ERROR_RATE',
                    'severity': 'CRITICAL',
                    'message': f"Error rate {row['error_rate']:.1f}% exceeds critical threshold",
                    'metric': 'error_rate',
                    'value': row['error_rate']
                })
            
            # Response time threshold check
            if row['response_time'] >= self.failure_thresholds['response_time_critical']:
                detected_failures.append({
                    'type': 'PERFORMANCE_DEGRADATION',
                    'severity': 'CRITICAL',
                    'message': f"Response time {row['response_time']:.1f}ms exceeds critical threshold",
                    'metric': 'response_time',
                    'value': row['response_time']
                })
            
            # Availability threshold check
            if row['availability'] <= self.failure_thresholds['availability_critical']:
                detected_failures.append({
                    'type': 'AVAILABILITY_LOSS',
                    'severity': 'CRITICAL',
                    'message': f"Availability {row['availability']:.1f}% below critical threshold",
                    'metric': 'availability',
                    'value': row['availability']
                })
            
            if detected_failures:
                failures.append({
                    'service_name': row['service_name'],
                    'namespace': row['namespace'],
                    'timestamp': row['timestamp'],
                    'failures': detected_failures
                })
        
        return failures
    
    def prepare_training_data(self, metrics_history: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for ML models
        """
        # Create failure labels based on health status
        def create_failure_label(row):
            return 1 if row['status'] in ['UNHEALTHY', 'CRITICAL'] else 0
        
        # Sort by timestamp
        metrics_history = metrics_history.sort_values(['service_name', 'timestamp'])
        
        # Create features and labels
        features = metrics_history[self.feature_columns].fillna(0)
        labels = metrics_history.apply(create_failure_label, axis=1)
        
        return features.values, labels.values
    
    def train_classification_model(self, metrics_history: pd.DataFrame):
        """
        Train Random Forest classifier for failure prediction
        """
        try:
            X, y = self.prepare_training_data(metrics_history)
            
            if len(X) < 10:
                self.logger.warning("Not enough data to train classification model")
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest
            self.classification_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            self.classification_model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.classification_model.predict(X_test_scaled)
            
            self.logger.info("Classification Model Performance:")
            self.logger.info(f"Training completed with {len(X_train)} samples")
            self.logger.info(f"Test accuracy: {self.classification_model.score(X_test_scaled, y_test):.3f}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.classification_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.logger.info("Feature Importance:")
            for _, row in feature_importance.head().iterrows():
                self.logger.info(f"  {row['feature']}: {row['importance']:.3f}")
                
        except Exception as e:
            self.logger.error(f"Error training classification model: {e}")
    
    def train_time_series_model(self, metrics_history: pd.DataFrame, sequence_length: int = 10):
        """
        Train LSTM model for time series failure prediction
        """
        try:
            # Group by service for time series
            services = metrics_history.groupby(['service_name', 'namespace'])
            
            all_sequences = []
            all_labels = []
            
            for (service_name, namespace), group in services:
                group = group.sort_values('timestamp')
                
                if len(group) < sequence_length + 1:
                    continue
                
                # Prepare features
                features = group[self.feature_columns].fillna(0).values
                labels = (group['status'].isin(['UNHEALTHY', 'CRITICAL'])).astype(int).values
                
                # Create sequences
                for i in range(len(features) - sequence_length):
                    seq = features[i:i + sequence_length]
                    label = labels[i + sequence_length]
                    
                    all_sequences.append(seq)
                    all_labels.append(label)
            
            if len(all_sequences) < 10:
                self.logger.warning("Not enough data to train time series model")
                return
            
            X = np.array(all_sequences)
            y = np.array(all_labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Build LSTM model
            self.time_series_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, len(self.feature_columns))),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            self.time_series_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            history = self.time_series_model.fit(
                X_train, y_train,
                epochs=20,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Evaluate
            test_loss, test_accuracy = self.time_series_model.evaluate(X_test, y_test, verbose=0)
            
            self.logger.info(f"Time Series Model Performance:")
            self.logger.info(f"Training completed with {len(X_train)} sequences")
            self.logger.info(f"Test accuracy: {test_accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error training time series model: {e}")
    
    def train_anomaly_detector(self, metrics_history: pd.DataFrame):
        """
        Train Isolation Forest for anomaly detection
        """
        try:
            features = metrics_history[self.feature_columns].fillna(0)
            
            if len(features) < 10:
                self.logger.warning("Not enough data to train anomaly detector")
                return
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train Isolation Forest
            self.anomaly_detector = IsolationForest(
                contamination=0.1,  # Expect 10% anomalies
                random_state=42,
                n_jobs=-1
            )
            
            self.anomaly_detector.fit(features_scaled)
            
            self.logger.info("Anomaly detector trained successfully")
            
        except Exception as e:
            self.logger.error(f"Error training anomaly detector: {e}")
    
    def predict_failure(self, current_metrics: pd.DataFrame) -> List[FailurePrediction]:
        """
        Predict failures using trained ML models
        """
        predictions = []
        
        try:
            features = current_metrics[self.feature_columns].fillna(0)
            
            if self.classification_model is None:
                self.logger.warning("Classification model not trained")
                return predictions
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get predictions
            failure_probabilities = self.classification_model.predict_proba(features_scaled)[:, 1]
            feature_importance = self.classification_model.feature_importances_
            
            for idx, (_, row) in enumerate(current_metrics.iterrows()):
                prob = failure_probabilities[idx]
                
                # Determine failure type based on dominant metrics
                failure_type = self._determine_failure_type(row, feature_importance)
                
                # Calculate confidence
                confidence = max(prob, 1 - prob)
                
                # Predict failure time (simplified estimation)
                predicted_time = None
                if prob > 0.7:
                    predicted_time = datetime.now() + timedelta(minutes=15)
                elif prob > 0.5:
                    predicted_time = datetime.now() + timedelta(hours=1)
                
                # Contributing factors
                contributing_factors = self._get_contributing_factors(row, feature_importance)
                
                prediction = FailurePrediction(
                    service_name=row['service_name'],
                    namespace=row['namespace'],
                    timestamp=datetime.now(),
                    failure_probability=prob,
                    failure_type=failure_type,
                    predicted_failure_time=predicted_time,
                    confidence=confidence,
                    contributing_factors=contributing_factors
                )
                
                predictions.append(prediction)
                
        except Exception as e:
            self.logger.error(f"Error predicting failures: {e}")
        
        return predictions
    
    def detect_anomalies(self, current_metrics: pd.DataFrame) -> List[AnomalyDetection]:
        """
        Detect anomalies using trained anomaly detector
        """
        anomalies = []
        
        try:
            if self.anomaly_detector is None:
                self.logger.warning("Anomaly detector not trained")
                return anomalies
            
            features = current_metrics[self.feature_columns].fillna(0)
            features_scaled = self.scaler.transform(features)
            
            # Detect anomalies
            anomaly_predictions = self.anomaly_detector.predict(features_scaled)
            anomaly_scores = self.anomaly_detector.decision_function(features_scaled)
            
            for idx, (_, row) in enumerate(current_metrics.iterrows()):
                is_anomaly = anomaly_predictions[idx] == -1
                score = anomaly_scores[idx]
                
                # Identify anomalous metrics
                anomalous_metrics = []
                if is_anomaly:
                    anomalous_metrics = self._identify_anomalous_metrics(row)
                
                anomaly = AnomalyDetection(
                    service_name=row['service_name'],
                    namespace=row['namespace'],
                    timestamp=datetime.now(),
                    is_anomaly=is_anomaly,
                    anomaly_score=score,
                    anomalous_metrics=anomalous_metrics
                )
                
                anomalies.append(anomaly)
                
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
        
        return anomalies
    
    def _determine_failure_type(self, row: pd.Series, feature_importance: np.ndarray) -> str:
        """
        Determine the most likely failure type based on metrics
        """
        # Map features to failure types
        feature_to_type = {
            'cpu_usage': 'RESOURCE_EXHAUSTION',
            'memory_usage': 'RESOURCE_EXHAUSTION',
            'error_rate': 'HIGH_ERROR_RATE',
            'response_time': 'PERFORMANCE_DEGRADATION',
            'availability': 'AVAILABILITY_LOSS'
        }
        
        # Weight by feature importance and current values
        type_scores = {}
        for i, feature in enumerate(self.feature_columns):
            if feature in feature_to_type:
                failure_type = feature_to_type[feature]
                score = feature_importance[i] * (row[feature] / 100.0)  # Normalize
                
                if failure_type not in type_scores:
                    type_scores[failure_type] = 0
                type_scores[failure_type] += score
        
        # Return the failure type with highest score
        if type_scores:
            return max(type_scores, key=type_scores.get)
        return 'UNKNOWN'
    
    def _get_contributing_factors(self, row: pd.Series, feature_importance: np.ndarray) -> List[str]:
        """
        Get contributing factors for failure prediction
        """
        factors = []
        
        # Sort features by importance
        feature_importance_pairs = list(zip(self.feature_columns, feature_importance))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for feature, importance in feature_importance_pairs[:3]:  # Top 3 factors
            value = row[feature]
            if importance > 0.1:  # Only include significant factors
                factors.append(f"{feature}: {value:.1f} (importance: {importance:.3f})")
        
        return factors
    
    def _identify_anomalous_metrics(self, row: pd.Series) -> List[str]:
        """
        Identify which metrics are anomalous
        """
        anomalous = []
        
        # Simple heuristic: identify metrics that are significantly different from normal ranges
        normal_ranges = {
            'cpu_usage': (0, 80),
            'memory_usage': (0, 80),
            'error_rate': (0, 5),
            'response_time': (0, 1000),
            'availability': (95, 100)
        }
        
        for metric, (min_val, max_val) in normal_ranges.items():
            if metric in row:
                value = row[metric]
                if value < min_val or value > max_val:
                    anomalous.append(f"{metric}: {value:.1f}")
        
        return anomalous
    
    def save_models(self, model_path: str = "models/"):
        """
        Save trained models to disk
        """
        try:
            import os
            os.makedirs(model_path, exist_ok=True)
            
            if self.classification_model:
                joblib.dump(self.classification_model, f"{model_path}/classification_model.pkl")
            
            if self.time_series_model:
                self.time_series_model.save(f"{model_path}/time_series_model.h5")
            
            if self.anomaly_detector:
                joblib.dump(self.anomaly_detector, f"{model_path}/anomaly_detector.pkl")
            
            joblib.dump(self.scaler, f"{model_path}/scaler.pkl")
            
            self.logger.info(f"Models saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def load_models(self, model_path: str = "models/"):
        """
        Load trained models from disk
        """
        try:
            import os
            
            if os.path.exists(f"{model_path}/classification_model.pkl"):
                self.classification_model = joblib.load(f"{model_path}/classification_model.pkl")
            
            if os.path.exists(f"{model_path}/time_series_model.h5"):
                self.time_series_model = tf.keras.models.load_model(f"{model_path}/time_series_model.h5")
            
            if os.path.exists(f"{model_path}/anomaly_detector.pkl"):
                self.anomaly_detector = joblib.load(f"{model_path}/anomaly_detector.pkl")
            
            if os.path.exists(f"{model_path}/scaler.pkl"):
                self.scaler = joblib.load(f"{model_path}/scaler.pkl")
            
            self.logger.info(f"Models loaded from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")


# Example usage
if __name__ == "__main__":
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data for testing
    def create_sample_data():
        np.random.seed(42)
        
        services = [
            ("user-service", "default"),
            ("order-service", "default"),
            ("payment-service", "default")
        ]
        
        data = []
        base_time = datetime.now() - timedelta(hours=24)
        
        for service_name, namespace in services:
            for i in range(100):  # 100 data points per service
                timestamp = base_time + timedelta(minutes=i * 15)
                
                # Simulate normal and failure conditions
                if i > 80:  # Last 20 points simulate failure
                    cpu_usage = np.random.normal(85, 10)
                    memory_usage = np.random.normal(80, 15)
                    error_rate = np.random.normal(12, 5)
                    response_time = np.random.normal(2000, 500)
                    availability = np.random.normal(92, 3)
                    status = 'CRITICAL' if i > 90 else 'UNHEALTHY'
                else:
                    cpu_usage = np.random.normal(40, 10)
                    memory_usage = np.random.normal(50, 10)
                    error_rate = np.random.normal(2, 1)
                    response_time = np.random.normal(200, 50)
                    availability = np.random.normal(99, 1)
                    status = 'HEALTHY'
                
                data.append({
                    'timestamp': timestamp,
                    'service_name': service_name,
                    'namespace': namespace,
                    'cpu_usage': max(0, min(100, cpu_usage)),
                    'memory_usage': max(0, min(100, memory_usage)),
                    'error_rate': max(0, error_rate),
                    'response_time': max(0, response_time),
                    'request_rate': np.random.normal(100, 20),
                    'network_io': np.random.normal(1000, 200),
                    'availability': max(0, min(100, availability)),
                    'status': status
                })
        
        return pd.DataFrame(data)
    
    # Initialize failure detector
    detector = FailureDetector()
    
    # Create sample data
    print("Creating sample data...")
    sample_data = create_sample_data()
    
    # Train models
    print("Training models...")
    detector.train_classification_model(sample_data)
    detector.train_time_series_model(sample_data)
    detector.train_anomaly_detector(sample_data)
    
    # Test predictions on recent data
    print("\nTesting predictions...")
    recent_data = sample_data.tail(10)
    
    # Test failure prediction
    predictions = detector.predict_failure(recent_data)
    print(f"\nFailure Predictions:")
    for pred in predictions:
        print(f"  {pred.service_name}: {pred.failure_probability:.3f} probability")
        print(f"    Type: {pred.failure_type}")
        print(f"    Confidence: {pred.confidence:.3f}")
        if pred.predicted_failure_time:
            print(f"    Predicted failure time: {pred.predicted_failure_time}")
    
    # Test anomaly detection
    anomalies = detector.detect_anomalies(recent_data)
    print(f"\nAnomaly Detection:")
    for anomaly in anomalies:
        if anomaly.is_anomaly:
            print(f"  {anomaly.service_name}: ANOMALY (score: {anomaly.anomaly_score:.3f})")
            print(f"    Anomalous metrics: {', '.join(anomaly.anomalous_metrics)}")
    
    # Test rule-based failure detection
    failures = detector.detect_immediate_failures(recent_data)
    print(f"\nImmediate Failures:")
    for failure in failures:
        print(f"  {failure['service_name']}:")
        for fail in failure['failures']:
            print(f"    - {fail['type']}: {fail['message']}")
    
    # Save models
    detector.save_models()
    print("\nModels saved successfully!")