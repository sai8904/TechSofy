import logging
import subprocess
from typing import Dict, List, Optional
import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import docker
from docker.errors import APIError

class Orchestrator:
    """
    Orchestration Module
    Handles execution of healing actions on Kubernetes/Docker
    """
    
    def __init__(self, kube_config: str = None, docker_socket: str = "unix://var/run/docker.sock"):
        self.logger = logging.getLogger(__name__)
        
        # Initialize Kubernetes client
        try:
            if kube_config:
                config.load_kube_config(config_file=kube_config)
            else:
                config.load_incluster_config()  # For running inside a cluster
                
            self.k8s_apps_v1 = client.AppsV1Api()
            self.k8s_core_v1 = client.CoreV1Api()
            self.k8s_networking_v1 = client.NetworkingV1Api()
            self.k8s_available = True
        except Exception as e:
            self.logger.warning(f"Kubernetes initialization failed: {e}")
            self.k8s_available = False
        
        # Initialize Docker client
        try:
            self.docker_client = docker.DockerClient(base_url=docker_socket)
            self.docker_available = True
        except Exception as e:
            self.logger.warning(f"Docker initialization failed: {e}")
            self.docker_available = False
    
    def execute_action(self, action: Dict) -> Dict:
        """
        Execute a healing action
        Returns dict with status and details
        """
        action_type = action.get('action_type')
        service_name = action.get('service_name')
        namespace = action.get('namespace', 'default')
        
        try:
            if action_type == 'RESTART':
                return self._restart_service(service_name, namespace, action.get('parameters', {}))
            elif action_type == 'SCALE_UP':
                return self._scale_service(service_name, namespace, 'up', action.get('parameters', {}))
            elif action_type == 'SCALE_DOWN':
                return self._scale_service(service_name, namespace, 'down', action.get('parameters', {}))
            elif action_type == 'REROUTE':
                return self._reroute_traffic(service_name, namespace, action.get('parameters', {}))
            elif action_type == 'ROLLBACK':
                return self._rollback_deployment(service_name, namespace, action.get('parameters', {}))
            else:
                return {
                    'status': 'error',
                    'message': f'Unknown action type: {action_type}'
                }
        except Exception as e:
            self.logger.error(f"Error executing action {action_type} on {service_name}: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _restart_service(self, service_name: str, namespace: str, parameters: Dict) -> Dict:
        """
        Restart a service by deleting pods (Kubernetes) or restarting containers (Docker)
        """
        timeout = parameters.get('timeout', 30)
        retries = parameters.get('retries', 3)
        
        if self.k8s_available:
            try:
                # Get the deployment
                deployment = self.k8s_apps_v1.read_namespaced_deployment(
                    name=service_name,
                    namespace=namespace
                )
                
                # Trigger rolling restart by updating an annotation
                patch = {
                    'spec': {
                        'template': {
                            'metadata': {
                                'annotations': {
                                    'kubectl.kubernetes.io/restartedAt': datetime.now().isoformat()
                                }
                            }
                        }
                    }
                }
                
                self.k8s_apps_v1.patch_namespaced_deployment(
                    name=service_name,
                    namespace=namespace,
                    body=patch
                )
                
                return {
                    'status': 'success',
                    'message': f'Triggered rolling restart for {service_name} in {namespace}',
                    'details': {
                        'method': 'kubernetes_rolling_restart',
                        'timeout': timeout,
                        'retries': retries
                    }
                }
                
            except ApiException as e:
                self.logger.error(f"Kubernetes API error restarting {service_name}: {e}")
                return {
                    'status': 'error',
                    'message': f'Kubernetes API error: {e}'
                }
        
        elif self.docker_available:
            try:
                containers = self.docker_client.containers.list(
                    filters={'name': service_name}
                )
                
                if not containers:
                    return {
                        'status': 'error',
                        'message': f'No containers found for service {service_name}'
                    }
                
                for container in containers:
                    container.restart(timeout=timeout)
                
                return {
                    'status': 'success',
                    'message': f'Restarted {len(containers)} containers for {service_name}',
                    'details': {
                        'method': 'docker_restart',
                        'containers': [c.name for c in containers],
                        'timeout': timeout
                    }
                }
                
            except APIError as e:
                self.logger.error(f"Docker API error restarting {service_name}: {e}")
                return {
                    'status': 'error',
                    'message': f'Docker API error: {e}'
                }
        
        else:
            return {
                'status': 'error',
                'message': 'Neither Kubernetes nor Docker API is available'
            }
    
    def _scale_service(self, service_name: str, namespace: str, direction: str, parameters: Dict) -> Dict:
        """
        Scale a service up or down
        """
        if not self.k8s_available:
            return {
                'status': 'error',
                'message': 'Kubernetes API not available'
            }
        
        try:
            # Get current deployment
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=service_name,
                namespace=namespace
            )
            
            current_replicas = deployment.spec.replicas or 1
            
            if direction == 'up':
                increment = parameters.get('increment', 1)
                max_replicas = parameters.get('max_replicas', 5)
                new_replicas = min(current_replicas + increment, max_replicas)
            else:
                decrement = parameters.get('decrement', 1)
                min_replicas = parameters.get('min_replicas', 1)
                new_replicas = max(current_replicas - decrement, min_replicas)
            
            if new_replicas == current_replicas:
                return {
                    'status': 'no_change',
                    'message': f'Replica count already at {current_replicas}'
                }
            
            # Update the deployment
            patch = {'spec': {'replicas': new_replicas}}
            
            self.k8s_apps_v1.patch_namespaced_deployment(
                name=service_name,
                namespace=namespace,
                body=patch
            )
            
            return {
                'status': 'success',
                'message': f'Scaled {service_name} from {current_replicas} to {new_replicas} replicas',
                'details': {
                    'direction': direction,
                    'current_replicas': current_replicas,
                    'new_replicas': new_replicas
                }
            }
            
        except ApiException as e:
            self.logger.error(f"Kubernetes API error scaling {service_name}: {e}")
            return {
                'status': 'error',
                'message': f'Kubernetes API error: {e}'
            }
    
    def _reroute_traffic(self, service_name: str, namespace: str, parameters: Dict) -> Dict:
        """
        Reroute traffic to healthy instances or versions
        """
        if not self.k8s_available:
            return {
                'status': 'error',
                'message': 'Kubernetes API not available'
            }
        
        try:
            percentage = parameters.get('percentage', 50)
            timeout = parameters.get('timeout', 60)
            
            # This is a simplified example - in practice you'd need to:
            # 1. Identify healthy instances/versions
            # 2. Update Istio VirtualService or Ingress resources
            
            # Example for Istio VirtualService
            vs_name = f"{service_name}-virtualservice"
            
            try:
                virtual_service = self.k8s_networking_v1.read_namespaced_virtual_service(
                    name=vs_name,
                    namespace=namespace
                )
                
                # Modify the virtual service to shift traffic
                # (This is a simplified example - actual implementation would vary)
                for http_route in virtual_service.spec.http:
                    for route in http_route.route:
                        if 'healthy' in route.destination.subset:
                            route.weight = percentage
                        else:
                            route.weight = 100 - percentage
                
                self.k8s_networking_v1.patch_namespaced_virtual_service(
                    name=vs_name,
                    namespace=namespace,
                    body=virtual_service
                )
                
                return {
                    'status': 'success',
                    'message': f'Rerouted {percentage}% traffic to healthy instances of {service_name}',
                    'details': {
                        'method': 'istio_virtualservice_update',
                        'timeout': timeout
                    }
                }
                
            except ApiException:
                # Fallback to simple pod deletion if VirtualService not found
                self.logger.warning(f"VirtualService {vs_name} not found, falling back to pod deletion")
                return self._restart_service(service_name, namespace, parameters)
            
        except ApiException as e:
            self.logger.error(f"Kubernetes API error rerouting traffic for {service_name}: {e}")
            return {
                'status': 'error',
                'message': f'Kubernetes API error: {e}'
            }
    
    def _rollback_deployment(self, service_name: str, namespace: str, parameters: Dict) -> Dict:
        """
        Rollback a deployment to a previous version
        """
        if not self.k8s_available:
            return {
                'status': 'error',
                'message': 'Kubernetes API not available'
            }
        
        try:
            revision = parameters.get('revision')
            
            if revision:
                # Rollback to specific revision
                body = {
                    'name': service_name,
                    'rollback_to': {
                        'revision': revision
                    }
                }
            else:
                # Rollback to previous revision
                deployments = self.k8s_apps_v1.list_namespaced_deployment(namespace)
                rollout_history = subprocess.check_output(
                    f"kubectl rollout history deployment/{service_name} -n {namespace}",
                    shell=True
                ).decode()
                
                # Parse rollout history to get previous revision
                # (This is simplified - in practice you'd want better parsing)
                revisions = [line.split()[0] for line in rollout_history.split('\n') if line.strip()]
                if len(revisions) < 2:
                    return {
                        'status': 'error',
                        'message': 'No previous revision available'
                    }
                
                revision = revisions[1]  # Assuming first is current, second is previous
                body = {
                    'name': service_name,
                    'rollback_to': {
                        'revision': int(revision)
                    }
                }
            
            self.k8s_apps_v1.create_namespaced_deployment_rollback(
                name=service_name,
                namespace=namespace,
                body=body
            )
            
            return {
                'status': 'success',
                'message': f'Rolled back {service_name} to revision {revision}',
                'details': {
                    'revision': revision
                }
            }
            
        except ApiException as e:
            self.logger.error(f"Kubernetes API error rolling back {service_name}: {e}")
            return {
                'status': 'error',
                'message': f'Kubernetes API error: {e}'
            }
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error getting rollout history for {service_name}: {e}")
            return {
                'status': 'error',
                'message': f'Error getting rollout history: {e}'
            }


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # Test actions
    actions = [
        {
            'action_type': 'RESTART',
            'service_name': 'user-service',
            'namespace': 'default',
            'parameters': {
                'timeout': 30,
                'retries': 3
            }
        },
        {
            'action_type': 'SCALE_UP',
            'service_name': 'payment-service',
            'namespace': 'default',
            'parameters': {
                'increment': 1,
                'max_replicas': 5
            }
        },
        {
            'action_type': 'REROUTE',
            'service_name': 'order-service',
            'namespace': 'default',
            'parameters': {
                'percentage': 50,
                'timeout': 60
            }
        }
    ]
    
    for action in actions:
        print(f"\nExecuting action: {action['action_type']} on {action['service_name']}")
        result = orchestrator.execute_action(action)
        print("Result:", result)