#!/bin/bash
# Complete production deployment script

set -e

echo "Starting Health Orchestrator production deployment..."

# Step 1: Pre-deployment checks
echo "Running pre-deployment checks..."
./scripts/check_prerequisites.sh

# Step 2: Build and push images
echo "Building and pushing container images..."
docker build -t health-orchestrator:$(git rev-parse --short HEAD) .
docker tag health-orchestrator:$(git rev-parse --short HEAD) health-orchestrator:latest

# Step 3: Deploy to Kubernetes
echo "Deploying to Kubernetes..."
kubectl apply -f kubernetes/rbac.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml

# Step 4: Wait for deployment
echo "Waiting for deployment to be ready..."
kubectl rollout status deployment/health-orchestrator --timeout=600s

# Step 5: Run health checks
echo "Running health checks..."
./scripts/health_check.sh

# Step 6: Setup monitoring
echo "Setting up monitoring..."
./scripts/setup_monitoring.sh

# Step 7: Create initial backup
echo "Creating initial backup..."
./scripts/backup.sh

echo "Production deployment completed successfully!"
echo "Dashboard available at: http://$(kubectl get svc health-orchestrator -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):8080"