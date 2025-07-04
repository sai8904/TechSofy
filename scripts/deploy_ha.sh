#!/bin/bash
# Deploy Health Orchestrator in High Availability mode

set -e

echo "Deploying Health Orchestrator in HA mode..."

# Deploy primary instance
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/rbac.yaml

# Deploy multiple replicas
sed 's/replicas: 2/replicas: 3/' kubernetes/deployment.yaml | kubectl apply -f -

# Create load balancer service
cat > kubernetes/service-lb.yaml << 'EOL'
apiVersion: v1
kind: Service
metadata:
  name: health-orchestrator-lb
  namespace: default
spec:
  selector:
    app: health-orchestrator
  ports:
  - port: 80
    targetPort: 8080
    name: http
  type: LoadBalancer
  sessionAffinity: ClientIP