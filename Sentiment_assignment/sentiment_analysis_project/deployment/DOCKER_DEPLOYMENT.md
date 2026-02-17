# Docker Deployment for Sentiment Analysis Application

## Dockerfile for Streamlit Application

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Dockerfile for Flask API

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK CMD curl --fail http://localhost:5000/api/health || exit 1

# Run Flask
CMD ["python", "app/flask_app.py"]
```

## Docker Compose Configuration

```yaml
version: '3.8'

services:
  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    container_name: sentiment-streamlit
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_BROWSER_GATHERUSAGE=false
    restart: unless-stopped
    networks:
      - sentiment-network

  flask:
    build:
      context: .
      dockerfile: Dockerfile.flask
    container_name: sentiment-flask
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - FLASK_ENV=production
    restart: unless-stopped
    networks:
      - sentiment-network

  nginx:
    image: nginx:latest
    container_name: sentiment-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./deployment/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./deployment/ssl:/etc/nginx/ssl:ro
    depends_on:
      - streamlit
      - flask
    restart: unless-stopped
    networks:
      - sentiment-network

networks:
  sentiment-network:
    driver: bridge
```

## Building and Running with Docker

### Build Images
```bash
# Build both images
docker build -f Dockerfile.streamlit -t sentiment-streamlit:latest .
docker build -f Dockerfile.flask -t sentiment-flask:latest .

# Or use docker-compose
docker-compose build
```

### Run Containers
```bash
# Using docker-compose (recommended)
docker-compose up -d

# Or run individual containers
docker run -d -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --name sentiment-streamlit \
  sentiment-streamlit:latest

docker run -d -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --name sentiment-flask \
  sentiment-flask:latest
```

### View Logs
```bash
# View logs from docker-compose
docker-compose logs -f streamlit
docker-compose logs -f flask

# Or from individual containers
docker logs -f sentiment-streamlit
docker logs -f sentiment-flask
```

### Stop Containers
```bash
# Using docker-compose
docker-compose down

# Or stop individual containers
docker stop sentiment-streamlit sentiment-flask
docker rm sentiment-streamlit sentiment-flask
```

## Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Tag images
docker tag sentiment-streamlit:latest your-username/sentiment-streamlit:latest
docker tag sentiment-flask:latest your-username/sentiment-flask:latest

# Push images
docker push your-username/sentiment-streamlit:latest
docker push your-username/sentiment-flask:latest
```

## Deploy to AWS ECR

```bash
# Create ECR repository
aws ecr create-repository --repository-name sentiment-analysis

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789012.dkr.ecr.us-east-1.amazonaws.com

# Tag images
docker tag sentiment-streamlit:latest \
  123456789012.dkr.ecr.us-east-1.amazonaws.com/sentiment-analysis:streamlit

docker tag sentiment-flask:latest \
  123456789012.dkr.ecr.us-east-1.amazonaws.com/sentiment-analysis:flask

# Push to ECR
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/sentiment-analysis:streamlit
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/sentiment-analysis:flask
```

## Kubernetes Deployment

### Create Kubernetes Manifests

**streamlit-deployment.yaml**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-streamlit
spec:
  replicas: 2
  selector:
    matchLabels:
      app: sentiment-streamlit
  template:
    metadata:
      labels:
        app: sentiment-streamlit
    spec:
      containers:
      - name: streamlit
        image: your-username/sentiment-streamlit:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        volumeMounts:
        - name: models
          mountPath: /app/models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: sentiment-models-pvc
```

### Deploy to Kubernetes
```bash
# Apply manifests
kubectl apply -f streamlit-deployment.yaml
kubectl apply -f flask-deployment.yaml
kubectl apply -f service.yaml

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/sentiment-streamlit
kubectl logs -f deployment/sentiment-flask

# Scale deployment
kubectl scale deployment sentiment-streamlit --replicas=3
```

## Monitoring with Docker/Kubernetes

### Docker Stats
```bash
# Monitor resource usage
docker stats sentiment-streamlit sentiment-flask

# Check health
docker inspect --format='{{json .State.Health}}' sentiment-streamlit
```

### Kubernetes Metrics
```bash
# View resource usage
kubectl top pods
kubectl top nodes

# Setup Prometheus for detailed monitoring
helm install prometheus prometheus-community/kube-prometheus-stack
```

## Database Integration (Optional)

Add PostgreSQL service to track predictions:

```yaml
  postgres:
    image: postgres:13
    container_name: sentiment-postgres
    environment:
      POSTGRES_DB: sentiment_db
      POSTGRES_USER: sentiment
      POSTGRES_PASSWORD: secure_password
    volumes:
      - sentiment-db:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - sentiment-network

volumes:
  sentiment-db:
```

## Backup and Recovery

```bash
# Backup model files
docker run --rm -v sentiment-models:/models \
  -v $(pwd)/backups:/backup \
  ubuntu tar czf /backup/models-backup.tar.gz -C /models .

# Backup database
docker exec sentiment-postgres pg_dump -U sentiment sentiment_db > backup.sql

# Restore from backup
cat backup.sql | docker exec -i sentiment-postgres psql -U sentiment sentiment_db
```
