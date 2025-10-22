# Deployment Guide - B2B LLM Chatbot (Sahabat Chatbot)

This guide provides comprehensive instructions for deploying the B2B LLM Chatbot application to various environments.

## Table of Contents

- [Production Deployment Checklist](#production-deployment-checklist)
- [Server Requirements](#server-requirements)
- [Deployment Methods](#deployment-methods)
  - [Traditional Server Deployment](#traditional-server-deployment)
  - [Docker Deployment](#docker-deployment)
  - [Cloud Platform Deployment](#cloud-platform-deployment)
- [Post-Deployment Configuration](#post-deployment-configuration)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Troubleshooting](#troubleshooting)

## Production Deployment Checklist

Before deploying to production, ensure you have completed the following:

- [ ] Changed `DEBUG = False` in settings.py
- [ ] Set a strong, unique `SECRET_KEY`
- [ ] Configured proper `ALLOWED_HOSTS`
- [ ] Set up environment variables for sensitive data
- [ ] Configured database (PostgreSQL recommended for production)
- [ ] Set up static file serving (via nginx or CDN)
- [ ] Configured media file storage
- [ ] Set up HTTPS/SSL certificates
- [ ] Configured logging
- [ ] Set up backup strategy
- [ ] Configured monitoring tools
- [ ] Reviewed security settings
- [ ] Tested all functionality in staging environment

## Server Requirements

### Minimum Hardware Requirements

- **CPU**: 2 cores (4+ cores recommended)
- **RAM**: 4GB (8GB+ recommended for RAG operations)
- **Storage**: 20GB (+ space for documents and database)
- **Network**: Stable internet connection for LLM API calls

### Software Requirements

- **OS**: Ubuntu 20.04+ / CentOS 7+ / Debian 10+
- **Python**: 3.8 or higher
- **Database**: SQLite (development) / PostgreSQL 12+ (production recommended)
- **Web Server**: Nginx or Apache
- **Application Server**: Gunicorn or uWSGI
- **Supervisor**: For process management (optional but recommended)

## Deployment Methods

### Traditional Server Deployment

#### 1. Server Preparation

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3.10 python3.10-venv python3-pip
sudo apt install -y postgresql postgresql-contrib
sudo apt install -y nginx
sudo apt install -y supervisor

# Install system dependencies for document processing
sudo apt install -y libmagic1
```

#### 2. Create Application User

```bash
# Create dedicated user for the application
sudo useradd -m -s /bin/bash chatbot
sudo usermod -aG www-data chatbot
```

#### 3. Setup Application

```bash
# Switch to application user
sudo su - chatbot

# Clone the repository
git clone http://10.45.127.199:8082/b2b-services-delivery/sahabat_chatbot.git
cd sahabat_chatbot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn psycopg2-binary
```

#### 4. Configure Environment Variables

```bash
# Create .env file
cd chatbot
nano .env
```

Add the following to `.env`:

```env
# Django Settings
SECRET_KEY=your-very-long-random-secret-key-here
DEBUG=False
ALLOWED_HOSTS=your-domain.com,www.your-domain.com,server-ip

# Database (PostgreSQL)
DB_ENGINE=django.db.backends.postgresql
DB_NAME=chatbot_db
DB_USER=chatbot_user
DB_PASSWORD=strong-database-password
DB_HOST=localhost
DB_PORT=5432

# LLM Configuration
LLM_API_KEY=your_llm_api_key
LLM_API_URL=https://your-llm-endpoint.com/api

# Email Configuration (optional)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your-email@example.com
EMAIL_HOST_PASSWORD=your-email-password
EMAIL_USE_TLS=True

# Security
SECURE_SSL_REDIRECT=True
SESSION_COOKIE_SECURE=True
CSRF_COOKIE_SECURE=True
```

#### 5. Update settings.py for Production

```bash
nano chatbot/settings.py
```

Add/Update the following:

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Security Settings
DEBUG = os.getenv('DEBUG', 'False') == 'True'
SECRET_KEY = os.getenv('SECRET_KEY')
ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', '').split(',')

# Database Configuration
DATABASES = {
    'default': {
        'ENGINE': os.getenv('DB_ENGINE', 'django.db.backends.sqlite3'),
        'NAME': os.getenv('DB_NAME', BASE_DIR / 'db.sqlite3'),
        'USER': os.getenv('DB_USER', ''),
        'PASSWORD': os.getenv('DB_PASSWORD', ''),
        'HOST': os.getenv('DB_HOST', ''),
        'PORT': os.getenv('DB_PORT', ''),
    }
}

# Static files
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Security settings for production
if not DEBUG:
    SECURE_SSL_REDIRECT = True
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    X_FRAME_OPTIONS = 'DENY'
```

#### 6. Setup PostgreSQL Database

```bash
# Switch to postgres user
sudo -u postgres psql

# In PostgreSQL console
CREATE DATABASE chatbot_db;
CREATE USER chatbot_user WITH PASSWORD 'strong-database-password';
ALTER ROLE chatbot_user SET client_encoding TO 'utf8';
ALTER ROLE chatbot_user SET default_transaction_isolation TO 'read committed';
ALTER ROLE chatbot_user SET timezone TO 'Asia/Jakarta';
GRANT ALL PRIVILEGES ON DATABASE chatbot_db TO chatbot_user;
\q
```

#### 7. Run Migrations and Collect Static Files

```bash
# As chatbot user with venv activated
cd /home/chatbot/sahabat_chatbot/chatbot
python manage.py migrate
python manage.py collectstatic --noinput
python manage.py createsuperuser
```

#### 8. Configure Gunicorn

Create Gunicorn configuration file:

```bash
nano /home/chatbot/sahabat_chatbot/gunicorn_config.py
```

Add:

```python
import multiprocessing

bind = "127.0.0.1:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 120
keepalive = 5

# Logging
accesslog = "/home/chatbot/logs/gunicorn-access.log"
errorlog = "/home/chatbot/logs/gunicorn-error.log"
loglevel = "info"

# Process naming
proc_name = "chatbot_gunicorn"

# Server mechanics
daemon = False
pidfile = "/home/chatbot/gunicorn.pid"
```

Create log directory:

```bash
mkdir -p /home/chatbot/logs
```

#### 9. Configure Supervisor

```bash
sudo nano /etc/supervisor/conf.d/chatbot.conf
```

Add:

```ini
[program:chatbot]
directory=/home/chatbot/sahabat_chatbot/chatbot
command=/home/chatbot/sahabat_chatbot/venv/bin/gunicorn chatbot.wsgi:application -c /home/chatbot/sahabat_chatbot/gunicorn_config.py
user=chatbot
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/home/chatbot/logs/supervisor-chatbot.log
stderr_logfile=/home/chatbot/logs/supervisor-chatbot-error.log
environment=PATH="/home/chatbot/sahabat_chatbot/venv/bin"
```

Update and start supervisor:

```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start chatbot
sudo supervisorctl status chatbot
```

#### 10. Configure Nginx

```bash
sudo nano /etc/nginx/sites-available/chatbot
```

Add:

```nginx
upstream chatbot_app {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com www.your-domain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/ssl/certs/your-certificate.crt;
    ssl_certificate_key /etc/ssl/private/your-private-key.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security headers
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    client_max_body_size 100M;

    location /static/ {
        alias /home/chatbot/sahabat_chatbot/chatbot/staticfiles/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    location /media/ {
        alias /home/chatbot/sahabat_chatbot/chatbot/media/;
        expires 30d;
    }

    location / {
        proxy_pass http://chatbot_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_redirect off;

        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts for streaming responses
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/chatbot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### 11. Setup SSL with Let's Encrypt (Optional but Recommended)

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

### Docker Deployment

#### 1. Create Dockerfile

```dockerfile
# /Users/supriyadi/Projects/LLM/B2B-LLM-Project/Dockerfile
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install gunicorn psycopg2-binary

# Copy project
COPY chatbot /app/

# Collect static files
RUN python manage.py collectstatic --noinput

# Create non-root user
RUN useradd -m -u 1000 chatbot && chown -R chatbot:chatbot /app
USER chatbot

# Expose port
EXPOSE 8000

# Run gunicorn
CMD ["gunicorn", "chatbot.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "3"]
```

#### 2. Create docker-compose.yml

```yaml
version: '3.8'

services:
  db:
    image: postgres:14
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=chatbot_db
      - POSTGRES_USER=chatbot_user
      - POSTGRES_PASSWORD=strong-password
    restart: always

  web:
    build: .
    command: gunicorn chatbot.wsgi:application --bind 0.0.0.0:8000 --workers 3
    volumes:
      - ./chatbot:/app
      - static_volume:/app/staticfiles
      - media_volume:/app/media
    ports:
      - "8000:8000"
    env_file:
      - ./chatbot/.env
    depends_on:
      - db
    restart: always

  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - static_volume:/app/staticfiles
      - media_volume:/app/media
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - web
    restart: always

volumes:
  postgres_data:
  static_volume:
  media_volume:
```

#### 3. Deploy with Docker

```bash
# Build and start containers
docker-compose up -d --build

# Run migrations
docker-compose exec web python manage.py migrate

# Create superuser
docker-compose exec web python manage.py createsuperuser

# View logs
docker-compose logs -f
```

### Cloud Platform Deployment

#### AWS EC2

1. Launch EC2 instance (Ubuntu 20.04 LTS, t3.medium or higher)
2. Configure security groups (ports 80, 443, 22)
3. Follow [Traditional Server Deployment](#traditional-server-deployment) steps
4. Configure AWS RDS for PostgreSQL (optional)
5. Use AWS S3 for media file storage (optional)

#### Google Cloud Platform

1. Create Compute Engine VM
2. Follow [Traditional Server Deployment](#traditional-server-deployment) steps
3. Configure Cloud SQL for PostgreSQL (optional)
4. Use Google Cloud Storage for media files (optional)

#### Azure

1. Create Virtual Machine
2. Follow [Traditional Server Deployment](#traditional-server-deployment) steps
3. Configure Azure Database for PostgreSQL (optional)
4. Use Azure Blob Storage for media files (optional)

## Post-Deployment Configuration

### 1. Setup Backup Strategy

```bash
# Create backup script
sudo nano /home/chatbot/backup.sh
```

Add:

```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/home/chatbot/backups"
mkdir -p $BACKUP_DIR

# Backup database
pg_dump -U chatbot_user chatbot_db | gzip > $BACKUP_DIR/db_$DATE.sql.gz

# Backup media files
tar -czf $BACKUP_DIR/media_$DATE.tar.gz /home/chatbot/sahabat_chatbot/chatbot/media/

# Backup documents
tar -czf $BACKUP_DIR/documents_$DATE.tar.gz /home/chatbot/sahabat_chatbot/chatbot/documents/

# Keep only last 7 days of backups
find $BACKUP_DIR -type f -mtime +7 -delete
```

Make executable and schedule:

```bash
chmod +x /home/chatbot/backup.sh
crontab -e
# Add: 0 2 * * * /home/chatbot/backup.sh
```

### 2. Configure Logging

Update `settings.py`:

```python
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/home/chatbot/logs/django.log',
            'maxBytes': 1024*1024*15,  # 15MB
            'backupCount': 10,
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['file'],
        'level': 'INFO',
    },
}
```

### 3. Setup Monitoring

Install and configure monitoring tools:

```bash
# Install monitoring tools
pip install sentry-sdk

# In settings.py
import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration

if not DEBUG:
    sentry_sdk.init(
        dsn="your-sentry-dsn",
        integrations=[DjangoIntegration()],
        traces_sample_rate=1.0,
        send_default_pii=True
    )
```

## Monitoring and Maintenance

### Application Monitoring

```bash
# Check application status
sudo supervisorctl status chatbot

# View logs
tail -f /home/chatbot/logs/gunicorn-error.log
tail -f /home/chatbot/logs/django.log

# Restart application
sudo supervisorctl restart chatbot
```

### Database Maintenance

```bash
# Optimize database
sudo -u postgres psql chatbot_db -c "VACUUM ANALYZE;"

# Check database size
sudo -u postgres psql chatbot_db -c "SELECT pg_size_pretty(pg_database_size('chatbot_db'));"
```

### Update Application

```bash
# As chatbot user
cd /home/chatbot/sahabat_chatbot
source venv/bin/activate
git pull origin main
pip install -r requirements.txt
cd chatbot
python manage.py migrate
python manage.py collectstatic --noinput
sudo supervisorctl restart chatbot
```

## Troubleshooting

### Common Issues

#### Static Files Not Loading

```bash
python manage.py collectstatic --clear --noinput
sudo systemctl restart nginx
```

#### Database Connection Error

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql -U chatbot_user -d chatbot_db -h localhost
```

#### High Memory Usage

```bash
# Check memory usage
free -h

# Restart application
sudo supervisorctl restart chatbot
```

#### Application Not Starting

```bash
# Check logs
sudo supervisorctl tail chatbot stderr
tail -f /home/chatbot/logs/gunicorn-error.log

# Check configuration
cd /home/chatbot/sahabat_chatbot/chatbot
source ../venv/bin/activate
python manage.py check --deploy
```

### Performance Optimization

1. **Enable Database Connection Pooling**
2. **Configure Redis for Caching** (optional)
3. **Use CDN for Static Files**
4. **Optimize Database Queries**
5. **Enable Gzip Compression in Nginx**

## Security Checklist

- [ ] Changed default SECRET_KEY
- [ ] DEBUG = False in production
- [ ] HTTPS enabled with valid SSL certificate
- [ ] Firewall configured (UFW or iptables)
- [ ] Regular security updates applied
- [ ] Strong database passwords
- [ ] Restricted admin panel access
- [ ] Regular backups configured
- [ ] Monitoring and logging enabled
- [ ] Rate limiting configured (optional)

## Support

For deployment assistance, contact the B2B Services Delivery team.
