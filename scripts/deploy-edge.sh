#!/bin/bash
# Edge deployment script for EdgeVision-Guard
# This script deploys the EdgeVision-Guard to an edge device like Raspberry Pi or Jetson Nano

set -e

# Default values
TARGET=""
USERNAME="pi"
PRIVATE_KEY=""
PORT=22
DEPLOY_DIR="/home/pi/edgevision-guard"
INSTALL_DOCKER=0
INSTALL_DEPENDENCIES=1
SETUP_SERVICE=1
COPY_MODELS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --target)
      TARGET="$2"
      shift 2
      ;;
    --username)
      USERNAME="$2"
      shift 2
      ;;
    --key)
      PRIVATE_KEY="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --deploy-dir)
      DEPLOY_DIR="$2"
      shift 2
      ;;
    --install-docker)
      INSTALL_DOCKER=1
      shift
      ;;
    --skip-dependencies)
      INSTALL_DEPENDENCIES=0
      shift
      ;;
    --skip-service)
      SETUP_SERVICE=0
      shift
      ;;
    --skip-models)
      COPY_MODELS=0
      shift
      ;;
    --help)
      echo "EdgeVision-Guard Edge Deployment Script"
      echo ""
      echo "Usage: $0 [options]"
      echo ""
      echo "Options:"
      echo "  --target IP          Target device IP address (required)"
      echo "  --username USER      SSH username (default: pi)"
      echo "  --key PATH           Path to SSH private key"
      echo "  --port PORT          SSH port (default: 22)"
      echo "  --deploy-dir DIR     Deployment directory (default: /home/pi/edgevision-guard)"
      echo "  --install-docker     Install Docker if not present"
      echo "  --skip-dependencies  Skip installing dependencies"
      echo "  --skip-service       Skip setting up systemd service"
      echo "  --skip-models        Skip copying model files"
      echo ""
      echo "Example:"
      echo "  $0 --target 192.168.1.100 --username pi --key ~/.ssh/id_rsa"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [ -z "$TARGET" ]; then
  echo "Error: Target IP address is required"
  echo "Use --help for usage information"
  exit 1
fi

# Build SSH command
SSH_CMD="ssh"
if [ -n "$PRIVATE_KEY" ]; then
  SSH_CMD="$SSH_CMD -i $PRIVATE_KEY"
fi
SSH_CMD="$SSH_CMD -p $PORT $USERNAME@$TARGET"

# Build SCP command
SCP_CMD="scp"
if [ -n "$PRIVATE_KEY" ]; then
  SCP_CMD="$SCP_CMD -i $PRIVATE_KEY"
fi
SCP_CMD="$SCP_CMD -P $PORT"

# Check SSH connection
echo "Testing SSH connection to $TARGET..."
if ! $SSH_CMD "echo Connection successful"; then
  echo "Error: Could not connect to $TARGET"
  exit 1
fi

echo "Deploying EdgeVision-Guard to $TARGET ($DEPLOY_DIR)..."

# Install Docker if requested
if [ $INSTALL_DOCKER -eq 1 ]; then
  echo "Installing Docker on target device..."
  $SSH_CMD "
    if ! command -v docker &> /dev/null; then
      echo 'Installing Docker...'
      curl -fsSL https://get.docker.com -o get-docker.sh
      sudo sh get-docker.sh
      sudo usermod -aG docker $USERNAME
      echo 'Docker installed successfully'
    else
      echo 'Docker is already installed'
    fi

    if ! command -v docker-compose &> /dev/null; then
      echo 'Installing Docker Compose...'
      sudo apt-get update
      sudo apt-get install -y docker-compose
      echo 'Docker Compose installed successfully'
    else
      echo 'Docker Compose is already installed'
    fi
  "
fi

# Install dependencies if requested
if [ $INSTALL_DEPENDENCIES -eq 1 ]; then
  echo "Installing dependencies on target device..."
  $SSH_CMD "
    echo 'Updating package lists...'
    sudo apt-get update

    echo 'Installing required packages...'
    sudo apt-get install -y \
      git \
      python3 \
      python3-pip \
      libgl1-mesa-glx \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender-dev

    echo 'Dependencies installed successfully'
  "
fi

# Create deployment directory
echo "Creating deployment directory..."
$SSH_CMD "mkdir -p $DEPLOY_DIR/models $DEPLOY_DIR/data $DEPLOY_DIR/logs"

# Prepare local files for copying
echo "Preparing local files for deployment..."
TEMP_DIR=$(mktemp -d)
mkdir -p "$TEMP_DIR/docker" "$TEMP_DIR/deployment"

# Copy required files
cp docker/Dockerfile.edge "$TEMP_DIR/docker/"
cp docker-compose.edge.yml "$TEMP_DIR/docker-compose.yml"
cp .env.example "$TEMP_DIR/.env"

# Create systemd service file
cat > "$TEMP_DIR/deployment/edgevision-guard.service" << EOF
[Unit]
Description=EdgeVision-Guard Fall Detection Service
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=$USERNAME
WorkingDirectory=$DEPLOY_DIR
ExecStart=/usr/bin/docker-compose -f docker-compose.yml up
ExecStop=/usr/bin/docker-compose -f docker-compose.yml down
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Copy files to target device
echo "Copying files to target device..."
$SCP_CMD -r "$TEMP_DIR/"* "$USERNAME@$TARGET:$DEPLOY_DIR/"

# Copy models if requested
if [ $COPY_MODELS -eq 1 ]; then
  echo "Copying model files..."
  if [ -d "models" ] && [ -n "$(ls -A models)" ]; then
    $SCP_CMD -r models/* "$USERNAME@$TARGET:$DEPLOY_DIR/models/"
  else
    echo "Warning: No model files found in ./models directory"
  fi
fi

# Set up systemd service if requested
if [ $SETUP_SERVICE -eq 1 ]; then
  echo "Setting up systemd service..."
  $SSH_CMD "
    sudo cp $DEPLOY_DIR/deployment/edgevision-guard.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable edgevision-guard.service
  "
fi

# Clean up temporary files
rm -rf "$TEMP_DIR"

echo "Deployment completed successfully!"
echo ""
echo "To start the service on the edge device:"
echo "  $SSH_CMD 'sudo systemctl start edgevision-guard'"
echo ""
echo "To check the status:"
echo "  $SSH_CMD 'sudo systemctl status edgevision-guard'"
echo ""
echo "To view logs:"
echo "  $SSH_CMD 'sudo journalctl -u edgevision-guard -f'"

exit 0