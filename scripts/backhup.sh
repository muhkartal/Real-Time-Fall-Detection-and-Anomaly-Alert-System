#!/bin/bash
# Backup script for EdgeVision-Guard
# This script creates a backup of models, configuration, and logs

set -e

# Configuration
DEFAULT_OUTPUT="./backup/edgevision-guard-backup-$(date +%Y%m%d-%H%M%S).tar.gz"
BACKUP_DIRS=("models" "logs" "config")
BACKUP_FILES=(".env" "models/registry.json")
TEMP_DIR="./backup/temp"

# Parse command-line arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -o|--output)
      OUTPUT="$2"
      shift 2
      ;;
    -e|--exclude)
      EXCLUDE+=("$2")
      shift 2
      ;;
    -i|--include-data)
      INCLUDE_DATA=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  -o, --output FILE       Output backup file (default: $DEFAULT_OUTPUT)"
      echo "  -e, --exclude PATTERN   Exclude files/directories matching pattern"
      echo "  -i, --include-data      Include data directory in backup (can be large)"
      echo "  -h, --help              Show this help message"
      echo ""
      echo "Example:"
      echo "  $0 --output my-backup.tar.gz --exclude \"*.onnx\" --include-data"
      exit 0
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

# Set default output if not specified
OUTPUT=${OUTPUT:-$DEFAULT_OUTPUT}

# Create backup directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT")"

# Create temporary directory
TEMP_BACKUP_DIR="$TEMP_DIR/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$TEMP_BACKUP_DIR"

echo "Creating backup of EdgeVision-Guard..."
echo "Output file: $OUTPUT"

# Copy directories to backup
for dir in "${BACKUP_DIRS[@]}"; do
  if [ -d "$dir" ]; then
    echo "Backing up directory: $dir"
    mkdir -p "$TEMP_BACKUP_DIR/$dir"
    
    # Build rsync exclude pattern
    RSYNC_EXCLUDE=""
    for pattern in "${EXCLUDE[@]}"; do
      RSYNC_EXCLUDE="$RSYNC_EXCLUDE --exclude=$pattern"
    done
    
    # Copy files with rsync
    rsync -av --quiet $RSYNC_EXCLUDE "$dir/" "$TEMP_BACKUP_DIR/$dir/"
  else
    echo "Warning: Directory not found: $dir"
  fi
done

# Copy individual files to backup
for file in "${BACKUP_FILES[@]}"; do
  if [ -f "$file" ]; then
    echo "Backing up file: $file"
    # Create directory structure if needed
    mkdir -p "$TEMP_BACKUP_DIR/$(dirname "$file")"
    cp "$file" "$TEMP_BACKUP_DIR/$file"
  else
    echo "Warning: File not found: $file"
  fi
done

# Include data directory if requested
if [ "$INCLUDE_DATA" = "1" ]; then
  if [ -d "data" ]; then
    echo "Backing up data directory..."
    mkdir -p "$TEMP_BACKUP_DIR/data"
    
    # Build rsync exclude pattern
    RSYNC_EXCLUDE=""
    for pattern in "${EXCLUDE[@]}"; do
      RSYNC_EXCLUDE="$RSYNC_EXCLUDE --exclude=$pattern"
    done
    
    # Copy files with rsync
    rsync -av --quiet $RSYNC_EXCLUDE "data/" "$TEMP_BACKUP_DIR/data/"
  else
    echo "Warning: Data directory not found"
  fi
fi

# Add metadata
echo "Creating backup metadata..."
METADATA_FILE="$TEMP_BACKUP_DIR/backup-metadata.json"
cat > "$METADATA_FILE" << EOL
{
  "backup_date": "$(date --iso-8601=seconds)",
  "backup_version": "1.0",
  "hostname": "$(hostname)",
  "user": "$(whoami)",
  "edgevision_version": "$(grep -oP '(?<=__version__ = ")[^"]+' src/__init__.py || echo 'unknown')",
  "contents": {
    "directories": $(printf '%s\n' "${BACKUP_DIRS[@]}" | jq -R . | jq -s .),
    "files": $(printf '%s\n' "${BACKUP_FILES[@]}" | jq -R . | jq -s .),
    "include_data": $([ "$INCLUDE_DATA" = "1" ] && echo "true" || echo "false")
  }
}
EOL

# Create a manifest of all backed up files
echo "Creating file manifest..."
MANIFEST_FILE="$TEMP_BACKUP_DIR/file-manifest.txt"
find "$TEMP_BACKUP_DIR" -type f -not -path "*/\.*" -not -path "*/backup-metadata.json" -not -path "*/file-manifest.txt" | sort > "$MANIFEST_FILE"

# Create the archive
echo "Creating backup archive..."
tar -czf "$OUTPUT" -C "$(dirname "$TEMP_BACKUP_DIR")" "$(basename "$TEMP_BACKUP_DIR")"

# Clean up temporary directory
echo "Cleaning up temporary files..."
rm -rf "$TEMP_BACKUP_DIR"

# Summary
BACKUP_SIZE=$(du -h "$OUTPUT" | cut -f1)
echo ""
echo "Backup completed successfully!"
echo "Backup file: $OUTPUT"
echo "Backup size: $BACKUP_SIZE"
echo ""
echo "To restore this backup, run:"
echo "./scripts/restore.sh --input $OUTPUT"

exit 0