import os
import time
import asyncio
import logging
import requests
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

load_dotenv()

# Configuration
SALAD_API_KEY = os.getenv("SALAD_API_KEY", "your_salad_api_key_here")
SALAD_ORG_NAME = os.getenv("SALAD_ORG_NAME", "your_org_name")
SALAD_PROJECT_NAME = os.getenv("SALAD_PROJECT_NAME", "your_project_name")
SALAD_CONTAINER_GROUP_NAME = os.getenv("SALAD_CONTAINER_GROUP_NAME", "llmchataiapp")
SALAD_CONTAINER_URL = os.getenv("SALAD_CONTAINER_URL", "https://your-salad-container-dns") 
IDLE_TIMEOUT_SECONDS = int(os.getenv("IDLE_TIMEOUT_SECONDS", 300)) # 5 minutes

SALAD_API_BASE = f"https://api.salad.com/api/public/organizations/{SALAD_ORG_NAME}/projects/{SALAD_PROJECT_NAME}/containers/{SALAD_CONTAINER_GROUP_NAME}"

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ollama", tags=["Ollama SaladCloud"])

# State
def is_salad_configured():
    return SALAD_API_KEY and SALAD_API_KEY != 'your_salad_api_key_here'

last_request_time = time.time()
container_lock = asyncio.Lock()

def get_container_status():
    headers = {"Salad-Api-Key": SALAD_API_KEY}
    response = requests.get(SALAD_API_BASE, headers=headers)
    if not response.ok:
        logger.error(f"Failed to get status. Response: {response.text}")
    response.raise_for_status()
    return response.json()

def start_container():
    headers = {"Salad-Api-Key": SALAD_API_KEY}
    response = requests.post(f"{SALAD_API_BASE}/start", headers=headers)
    response.raise_for_status()
    logger.info("Sent start request to Salad container.")

def stop_container():
    headers = {"Salad-Api-Key": SALAD_API_KEY}
    response = requests.post(f"{SALAD_API_BASE}/stop", headers=headers)
    response.raise_for_status()

async def check_and_start_container():
    if not is_salad_configured():
        logger.info("Salad API Key not configured. Skipping container status check.")
        return
    async with container_lock:
        status = get_container_status()
        
        target_status = status.get("state", {}).get("status", "")
        if target_status == "stopped":
            logger.info("Container is stopped. Calling Start API...")
            start_container()
            
        # Poll until replicas_ready or running_count >= 1
        while True:
            status = get_container_status()
            current_state = status.get("current_state", {})
            current_status = current_state.get("status", "")
            
            # Look for running replicas or replicas_ready based on SaladCloud's schema logic
            instances = current_state.get("instance_status_counts", {})
            running_instances = instances.get("running_count", 0)
            replicas_ready = status.get("replicas_ready", 0) # sometimes provided at root
            
            if current_status == "running" and (running_instances >= 1 or replicas_ready >= 1):
                logger.info("Container is fully running and ready (replicas_ready >= 1).")
                break
            
            logger.info(f"Waiting for container to be ready... Current status: {current_status}")
            await asyncio.sleep(5)

async def idle_monitor():
    global last_request_time
    logger.info("Idle monitor started.")
    while True:
        await asyncio.sleep(10) # check every 10 seconds
        idle_time = time.time() - last_request_time
        
        if idle_time > IDLE_TIMEOUT_SECONDS:
            try:
                status = get_container_status()
                target_status = status.get("state", {}).get("status", "")
                
                # Only stop if it's not already stopped
                if target_status != "stopped":
                    logger.info(f"Idle detected: Stopping GPU... (Idle for {int(idle_time)}s)")
                    stop_container()
            except Exception as e:
                logger.error(f"Error checking status in idle monitor: {e}")

def init_idle_monitor():
    if not is_salad_configured():
        logger.info("Salad API Key not configured. Disabling Salad idle monitor.")
        return
    # Reset last_request_time on startup
    global last_request_time
    last_request_time = time.time()
    asyncio.create_task(idle_monitor())

@router.post("/chat")
async def chat_endpoint(request: Request):
    global last_request_time
    body = await request.body()
    
    try:
        await check_and_start_container()
    except Exception as e:
        logger.error(f"Failed to start/verify container: {e}")
        raise HTTPException(status_code=500, detail="Failed to wake up container")
    
    # Forward the request to the actual container
    headers = {
        "Content-Type": "application/json",
    }
    # Add Salad-Api-Key if necessary for auth to the endpoint itself
    if SALAD_API_KEY:
        headers["Salad-Api-Key"] = SALAD_API_KEY
        
    try:
        url = f"{SALAD_CONTAINER_URL}/api/chat"
        # Use stream=True to support streaming responses (e.g., SSE from Ollama)
        response = requests.post(url, data=body, headers=headers, stream=True)
        
        # Reset the idle timer to 0 strictly on a successful request dispatch
        last_request_time = time.time()
        
        def generate():
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
                    
        return StreamingResponse(
            generate(), 
            status_code=response.status_code, 
            media_type=response.headers.get("content-type")
        )
        
    except Exception as e:
        logger.error(f"Error forwarding request to container: {e}")
        raise HTTPException(status_code=500, detail="Error forwarding request")


