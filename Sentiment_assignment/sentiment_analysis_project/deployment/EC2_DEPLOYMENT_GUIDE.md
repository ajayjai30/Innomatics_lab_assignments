# AWS EC2 Deployment Instructions

## Prerequisites

1. AWS Account with EC2 access
2. Security group allowing inbound traffic on ports:
   - 22 (SSH)
   - 80 (HTTP)
   - 8501 (Streamlit)
   - 5000 (Flask)
3. EC2 Key pair downloaded

## Step 1: Launch EC2 Instance

### Using AWS Console:
1. Go to EC2 Dashboard
2. Click "Launch Instance"
3. Select Ubuntu Server 20.04 LTS (or later) AMI
4. Choose instance type: t3.medium (minimum recommended)
5. Configure instance details: 
   - Ensure "Assign Public IP" is enabled
6. Add storage: 30 GB minimum
7. Add security group with required ports open
8. Review and launch
9. Select or create key pair

### Or using AWS CLI:
```bash
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.medium \
  --key-name your-key-name \
  --security-groups sentiment-analysis-sg
```

## Step 2: Connect to EC2 Instance

```bash
# SSH into the instance
ssh -i /path/to/your/key.pem ubuntu@your-ec2-public-ip

# Update system
sudo apt-get update
sudo apt-get upgrade -y
```

## Step 3: Deploy Application

### Option A: Using Deployment Script

```bash
# Download the deployment script
cd ~
wget https://your-repo-url/deploy_to_ec2.sh
chmod +x deploy_to_ec2.sh

# Run the deployment script
./deploy_to_ec2.sh
```

### Option B: Manual Deployment

```bash
# Clone repository or upload files
git clone https://your-repo-url/sentiment-analysis.git
cd sentiment-analysis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy model files to the instance (from local machine)
# scp -i key.pem -r models/* ubuntu@your-ip:/home/ubuntu/sentiment-analysis/models/
```

## Step 4: Configure and Start Services

### Start Streamlit Application
```bash
cd /home/ubuntu/sentiment-analysis
source venv/bin/activate
streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

### Start Flask API (in separate terminal)
```bash
cd /home/ubuntu/sentiment-analysis
source venv/bin/activate
python app/flask_app.py
```

## Step 5: Configure Nginx (Optional but Recommended)

```bash
# Install Nginx
sudo apt-get install nginx -y

# Create Nginx configuration
sudo nano /etc/nginx/sites-available/default
```

Add the configuration from `nginx_config.conf`

```bash
# Test and enable Nginx
sudo nginx -t
sudo systemctl restart nginx
```

## Step 6: Setup Auto-start Services

```bash
# Install systemd services
sudo cp deployment/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable streamlit flask
sudo systemctl start streamlit flask

# Check service status
sudo systemctl status streamlit
sudo systemctl status flask
```

## Step 7: Setup HTTPS (SSL Certificate)

### Using Let's Encrypt with Certbot:

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx -y

# Obtain certificate
sudo certbot certonly --nginx -d your-domain.com

# Configure Nginx for HTTPS
sudo nano /etc/nginx/sites-available/default
# Update with SSL certificate paths

# Restart Nginx
sudo systemctl restart nginx
```

## Step 8: Monitor Application

### Check logs
```bash
# Streamlit logs
sudo journalctl -u streamlit -f

# Flask logs
sudo journalctl -u flask -f

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Monitor system resources
```bash
# Install htop
sudo apt-get install htop -y
htop
```

## Step 9: Backup Model Files

```bash
# Create regular backups
sudo crontab -e

# Add this line to run daily backup at 2 AM
0 2 * * * tar -czf /backups/model-backup-$(date +\%Y\%m\%d).tar.gz /home/ubuntu/sentiment-analysis/models/
```

## Access the Application

Once deployed:

### Streamlit App
- URL: `http://your-ec2-public-ip:8501`
- Direct: `http://your-ec2-public-ip`

### Flask API
- Base URL: `http://your-ec2-public-ip:5000`
- API Docs: `http://your-ec2-public-ip:5000/api`

### Health Check
```bash
curl http://your-ec2-public-ip:5000/api/health
```

## API Examples

### Single Prediction
```bash
curl -X POST http://your-ec2-public-ip:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "This product is amazing!"}'
```

### Batch Prediction
```bash
curl -X POST http://your-ec2-public-ip:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"reviews": ["Great product!", "Poor quality"]}'
```

## Troubleshooting

### Port Already in Use
```bash
# Find process using port
sudo lsof -i :8501
sudo lsof -i :5000

# Kill process
sudo kill -9 <PID>
```

### Permission Denied
```bash
# Fix ownership
sudo chown -R ubuntu:ubuntu /home/ubuntu/sentiment-analysis/
```

### Memory Issues
```bash
# Increase swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Connection Refused
```bash
# Check security group rules in AWS Console
# Ensure required ports are open:
# - 22 (SSH)
# - 80 (HTTP)
# - 443 (HTTPS)
# - 8501 (Streamlit)
# - 5000 (Flask)
```

## Maintenance

### Update packages
```bash
sudo apt-get update
sudo apt-get upgrade -y
pip install --upgrade -r requirements.txt
```

### Restart services
```bash
sudo systemctl restart streamlit flask nginx
```

### View service logs
```bash
sudo journalctl -u streamlit -n 50
sudo journalctl -u flask -n 50
```

## Scaling Considerations

1. **Load Balancing**: Use AWS ELB/ALB for multiple instances
2. **Database**: Consider RDS for storing predictions
3. **Caching**: Implement Redis for faster predictions
4. **Model Updates**: Use S3 for model versioning
5. **Monitoring**: Setup CloudWatch for metrics and alerts

## Cost Optimization

- Use t3.medium for development/testing
- Use t3.large for production
- Set up auto-shutdown for idle instances
- Use reserved instances for long-term deployments
- Monitor CloudWatch for cost analysis

For detailed AWS EC2 documentation, visit: https://docs.aws.amazon.com/ec2/
