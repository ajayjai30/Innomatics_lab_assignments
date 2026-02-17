#!/bin/bash

# AWS EC2 Deployment Script for Sentiment Analysis Application
# This script sets up the environment and deploys the Streamlit/Flask application on EC2

set -e

echo "=========================================="
echo "Sentiment Analysis App - EC2 Deployment"
echo "=========================================="

# Update system packages
echo "Step 1: Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and dependencies
echo "Step 2: Installing Python and pip..."
sudo apt-get install -y python3.9 python3.9-venv python3-pip build-essential

# Install system dependencies for NLP
echo "Step 3: Installing system dependencies..."
sudo apt-get install -y libssl-dev libffi-dev python3-dev git wget

# Create application directory
echo "Step 4: Setting up application directory..."
APP_DIR="/home/ubuntu/sentiment-analysis-app"
sudo mkdir -p $APP_DIR
sudo chown -R ubuntu:ubuntu $APP_DIR

# Clone or copy the application code
echo "Step 5: Copying application files..."
# Replace with your repository URL if using git
cd $APP_DIR

# Create virtual environment
echo "Step 6: Creating Python virtual environment..."
python3.9 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Step 7: Installing Python packages..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Create systemd service for Streamlit app
echo "Step 8: Setting up Streamlit service..."
sudo tee /etc/systemd/system/streamlit.service > /dev/null <<EOF
[Unit]
Description=Streamlit Sentiment Analysis App
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
ExecStart=$APP_DIR/venv/bin/streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create systemd service for Flask app
echo "Step 9: Setting up Flask service..."
sudo tee /etc/systemd/system/flask.service > /dev/null <<EOF
[Unit]
Description=Flask Sentiment Analysis API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
ExecStart=$APP_DIR/venv/bin/python app/flask_app.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start services
echo "Step 10: Enabling services..."
sudo systemctl daemon-reload
sudo systemctl enable streamlit
sudo systemctl enable flask

# Install and configure Nginx as reverse proxy
echo "Step 11: Installing and configuring Nginx..."
sudo apt-get install -y nginx

# Configure Nginx for Streamlit and Flask
sudo tee /etc/nginx/sites-available/default > /dev/null <<'EOF'
upstream streamlit {
    server localhost:8501;
}

upstream flask {
    server localhost:5000;
}

server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    # Streamlit proxy
    location / {
        proxy_pass http://streamlit;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Flask API proxy
    location /api/ {
        proxy_pass http://flask;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

# Enable Nginx
sudo systemctl restart nginx

echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Services are configured and ready to start."
echo ""
echo "To start the services, run:"
echo "  sudo systemctl start streamlit"
echo "  sudo systemctl start flask"
echo ""
echo "To check service status, run:"
echo "  sudo systemctl status streamlit"
echo "  sudo systemctl status flask"
echo ""
echo "The application will be accessible at:"
echo "  Streamlit: http://your-ec2-public-ip:8501"
echo "  Flask API: http://your-ec2-public-ip:5000"
echo "  Nginx (Port 80): http://your-ec2-public-ip"
echo ""
echo "=========================================="
