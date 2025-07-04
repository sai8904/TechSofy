#!/bin/bash
# Check deployment prerequisites

set -e

echo "Checking prerequisites..."

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found"
    exit 1
fi
echo "✅ kubectl found"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found"
    exit 1
fi
echo "✅ Docker found"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi
echo "✅ Python 3 found"

# Check cluster connectivity
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Cannot connect to Kubernetes cluster"
    exit 1
fi
echo "✅ Kubernetes cluster accessible"

# Check required namespaces
if ! kubectl get namespace default &> /dev/null; then
    echo "❌ Default namespace not found"
    exit 1
fi
echo "✅ Default namespace exists"

# Check storage
REQUIRED_SPACE=5000000  # 5GB in KB
AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    echo "❌ Insufficient disk space (need 5GB)"
    exit 1
fi
echo "✅ Sufficient disk space available"

echo "All prerequisites satisfied!"