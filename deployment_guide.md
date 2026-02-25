# Deployment Guide for Ubuntu VPS

This guide explains how to deploy the LLM Gateway application to your Ubuntu 24.04 LTS VPS.

## Prerequisites on VPS

1.  **Access**: SSH access to your VPS.
2.  **Docker**: Installed on the VPS.
    ```bash
    # Quick install for Ubuntu
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    ```

## Deployment Steps

### 1. Build and Push (Local)
Run the build script on your local machine to push the image to Docker Hub:
```bash
# Make sure you are logged in: docker login
./build_and_push.sh
```

### 2. Prepare Files on VPS
You need to copy the following files from your local machine to your VPS (e.g., to `~/llm-gateway/`):
- `docker-compose.yml`
- `.env` (Make sure this contains your production API keys!)
- `deploy_on_vps.sh` (Optional, for convenience)

**Example using `scp` (run from your local machine):**
```bash
# Replace user@your-vps-ip with your actual VPS credentials
scp docker-compose.yml .env deploy_on_vps.sh user@your-vps-ip:~/llm-gateway/
```

### 3. Run Deployment (Remote)
SSH into your VPS and navigate to the directory:
```bash
ssh user@your-vps-ip
cd ~/llm-gateway

# Copy the new nginx config
mkdir -p nginx
# (You might need to copy nginx.conf from your local machine if not using git)
# scp nginx/nginx.conf user@your-vps-ip:~/llm-gateway/nginx/

chmod +x deploy_on_vps.sh
./deploy_on_vps.sh
```

Alternatively, without the script:
```bash
docker compose up -d
```

### 4. Verify
The application should be running on subdomain **llm-gateway.techxbytes.com** (Port 80).
Test it:
```bash
curl http://llm-gateway.techxbytes.com/health
```


