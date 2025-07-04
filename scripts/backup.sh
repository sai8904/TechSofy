#!/bin/bash
# Backup Health Orchestrator data and configuration

set -e

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Starting backup to $BACKUP_DIR..."

# Backup configuration files
cp -r config/ "$BACKUP_DIR/"
echo "Configuration backed up"

# Backup data and models
cp -r data/ "$BACKUP_DIR/"
cp -r models/ "$BACKUP_DIR/"
echo "Data and models backed up"

# Backup logs (last 7 days)
find logs/ -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/" \;
echo "Recent logs backed up"

# Backup Kubernetes resources
kubectl get all -n default -o yaml > "$BACKUP_DIR/k8s-resources.yaml"
kubectl get configmaps -n default -o yaml > "$BACKUP_DIR/k8s-configmaps.yaml"
kubectl get secrets -n default -o yaml > "$BACKUP_DIR/k8s-secrets.yaml"
echo "Kubernetes resources backed up"

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR/"
rm -rf "$BACKUP_DIR"

echo "Backup completed: $BACKUP_DIR.tar.gz"

# Clean up old backups (keep last 30 days)
find backups/ -name "*.tar.gz" -mtime +30 -delete
echo "Old backups cleaned up"
EOF

chmod +x scripts/backup.sh

# Create restore script
cat > scripts/restore.sh << EOF
#!/bin/bash
# Restore Health Orchestrator from backup

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

BACKUP_FILE="$1"
RESTORE_DIR="restore_$(date +%Y%m%d_%H%M%S)"

echo "Restoring from $BACKUP_FILE..."

# Extract backup
tar -xzf "$BACKUP_FILE" -C /tmp/
BACKUP_EXTRACTED=$(basename "$BACKUP_FILE" .tar.gz)

# Stop current services
kubectl scale deployment health-orchestrator --replicas=0
echo "Stopped current services"

# Restore configuration
cp -r "/tmp/$BACKUP_EXTRACTED/config/"* config/
echo "Configuration restored"

# Restore data and models
cp -r "/tmp/$BACKUP_EXTRACTED/data/"* data/
cp -r "/tmp/$BACKUP_EXTRACTED/models/"* models/
echo "Data and models restored"

# Restore Kubernetes resources if needed
if [ -f "/tmp/$BACKUP_EXTRACTED/k8s-resources.yaml" ]; then
    kubectl apply -f "/tmp/$BACKUP_EXTRACTED/k8s-resources.yaml"
    echo "Kubernetes resources restored"
fi

# Restart services
kubectl scale deployment health-orchestrator --replicas=2
echo "Services restarted"

# Clean up
rm -rf "/tmp/$BACKUP_EXTRACTED"
echo "Restore completed successfully!"