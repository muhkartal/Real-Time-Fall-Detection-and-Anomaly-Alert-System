#!/bin/bash
# Restore script for EdgeVision-Guard
# This script restores a backup created by backup.sh

set -e

# Configuration
TEMP_DIR="./backup/temp"
RESTORE_DIR="./restore"

# Parse command-line arguments
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--input)
      INPUT="$2"
      shift 2
      ;;
    -d|--restore-dir)
      RESTORE_DIR="$2"
      shift 2
      ;;
    -f|--force)
      FORCE=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  -i, --input FILE        Input backup file (required)"
      echo "  -d, --restore-dir DIR   Directory to restore to (default: ./restore)"
      echo "  -f, --force             Force overwrite of existing files"
      echo "  -h, --help              Show this help message"
      echo ""
      echo "Example:"
      echo "  $0 --input backup.tar.gz --restore-dir ./my-restore"
      exit 0
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

# Validate input
if [ -z "$INPUT" ]; then
  echo "Error: Input backup file is required"
  echo "Use -h or --help for usage information"
  exit 1
fi

if [ ! -f "$INPUT" ]; then
  echo "Error: Input backup file not found: $INPUT"
  exit 1
fi

# Create temporary directory
TEMP_EXTRACT_DIR="$TEMP_DIR/extract-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$TEMP_EXTRACT_DIR"

echo "Restoring EdgeVision-Guard backup..."
echo "Input file: $INPUT"
echo "Restore directory: $RESTORE_DIR"

# Extract the backup archive to temp directory
echo "Extracting backup archive..."
tar -xzf "$INPUT" -C "$TEMP_EXTRACT_DIR"

# Find the backup directory (should be only one subdirectory)
BACKUP_DIR=$(find "$TEMP_EXTRACT_DIR" -mindepth 1 -maxdepth 1 -type d | head -n1)
if [ -z "$BACKUP_DIR" ]; then
  echo "Error: Could not find backup directory in archive"
  exit 1
fi

# Check metadata
METADATA_FILE="$BACKUP_DIR/backup-metadata.json"
if [ -f "$METADATA_FILE" ]; then
  echo "Backup metadata:"
  cat "$METADATA_FILE" | jq '.'
else
  echo "Warning: No metadata file found in backup"
fi

# Check file manifest
MANIFEST_FILE="$BACKUP_DIR/file-manifest.txt"
if [ -f "$MANIFEST_FILE" ]; then
  FILE_COUNT=$(wc -l < "$MANIFEST_FILE")
  echo "Backup contains $FILE_COUNT files"
else
  echo "Warning: No file manifest found in backup"
fi

# Ask for confirmation if not forced
if [ "$FORCE" != "1" ]; then
  read -p "Proceed with restore? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Restore cancelled"
    rm -rf "$TEMP_EXTRACT_DIR"
    exit 1
  fi
fi

# Create restore directory
mkdir -p "$RESTORE_DIR"

# Restore files
echo "Restoring files..."
rsync -av --quiet "$BACKUP_DIR/" "$RESTORE_DIR/"

# Remove metadata and manifest files from restore dir
rm -f "$RESTORE_DIR/backup-metadata.json"
rm -f "$RESTORE_DIR/file-manifest.txt"

# Clean up temporary directory
echo "Cleaning up temporary files..."
rm -rf "$TEMP_EXTRACT_DIR"

# Provide instructions for integrating the restored files
echo ""
echo "Restore completed successfully!"
echo ""
echo "To use the restored files, you can:"
echo "1. Replace specific directories/files in your main installation:"
echo "   cp -R $RESTORE_DIR/models /path/to/your/installation/"
echo ""
echo "2. Or activate a restored model:"
echo "   ./scripts/model-manager.sh list --models-dir $RESTORE_DIR/models"
echo "   ./scripts/model-manager.sh activate --version <version> --models-dir $RESTORE_DIR/models"
echo ""
echo "3. Or copy the entire restore to replace your installation:"
echo "   cp -R $RESTORE_DIR/* /path/to/your/installation/"
echo ""
echo "Important: Review the restored .env file before using it!"

exit 0