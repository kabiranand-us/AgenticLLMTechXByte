from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
from contextlib import asynccontextmanager
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from config import settings
from llm_service import LLMFactory, SELECTABLE_FALLBACK_MODELS, invoke_with_fallback, CHAT_SYSTEM_PROMPT
from ollama_manager import router as ollama_router, init_idle_monitor

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    message: str
    provider: str = "google"  # Default to Google (Free Tier)
    model_name: Optional[str] = None
    temperature: Optional[float] = 0.7
    # Optional: when using provider=google + model_name=gemini-2.5-flash-lite, start fallback from this model
    prefer_model: Optional[str] = None  # One of SELECTABLE_FALLBACK_MODELS, or "auto" for full chain

class ChatResponse(BaseModel):
    response: str
    llm_provider: str
    model: str
    metadata: Optional[Dict[str, Any]] = None

class BlogRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    topic: str
    provider: str = "google"
    model_name: Optional[str] = None
    temperature: Optional[float] = 0.7
    prefer_model: Optional[str] = None

class BlogResponse(BaseModel):
    blog_content: str
    llm_provider: str
    model: str
    metadata: Optional[Dict[str, Any]] = None

# --- Lifespan Events ---
@asynccontextmanager
async def lifespan(application: FastAPI):
    # Startup logic: check if keys are present (optional warning)
    if not settings.GOOGLE_API_KEY:
        print("WARNING: GOOGLE_API_KEY not found. Gemini calls will fail.")
        
    # Start the Ollama SaladCloud idle monitor
    init_idle_monitor()
    
    yield
    # Shutdown logic if needed

# --- App Setup ---
app = FastAPI(
    title="Unified LLM Gateway",
    description="API Gateway for multiple LLM providers (Gemini, Groq, Claude, DeepSeek)",
    version="1.0.1",
    lifespan=lifespan
)

# Include Ollama API router
app.include_router(ollama_router)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://techxbytes.com",
        "https://www.techxbytes.com",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoints ---

@app.get("/health")
async def health_check():
    return {"status": "ok", "env": settings.APP_ENV}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/models")
async def list_fallback_models():
    """Return the list of model IDs that can be used with prefer_model for the flash-lite fallback chain."""
    return {"prefer_model_options": SELECTABLE_FALLBACK_MODELS}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. Normalize inputs
        model_name = request.model_name
        if model_name == "string":
            model_name = None
        prefer_model = getattr(request, "prefer_model", None)
        if prefer_model == "string":
            prefer_model = None

        # 2. Invoke, cascading through the full fallback chain on real rate-limit errors.
        #    Send the chat system prompt so architectural answers come back as Mermaid.
        ai_message, provider_used, model_used = invoke_with_fallback(
            request.message, request.provider, model_name, prefer_model,
            system_prompt=CHAT_SYSTEM_PROMPT,
        )

        # 3. Extract content
        response_text = ai_message.content
        if isinstance(response_text, list):
            # Handle multi-modal response where content is a list of blocks
            # Usually only text blocks for chat models
            response_text = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block) 
                for block in response_text
            )
        
        # 4. Return structured response (provider_used/model_used reflect fallback to Groq when applicable)
        return ChatResponse(
            response=response_text,
            llm_provider=provider_used,
            model=model_used,
            metadata=ai_message.response_metadata if hasattr(ai_message, 'response_metadata') else {}
        )

    except ValueError as e:
        # Handle known errors (e.g. unsupported provider, missing key)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle unexpected errors
        import sys, traceback
        sys.stderr.write(f"Error processing request: {e}\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise HTTPException(status_code=500, detail="Internal Server Error processing LLM request.")

@app.post("/api/create/blog", response_model=BlogResponse)
async def create_blog_endpoint(request: BlogRequest):
    try:
        model_name = request.model_name
        if model_name == "string":
            model_name = None
        prefer_model = getattr(request, "prefer_model", None)
        if prefer_model == "string":
            prefer_model = None

        from content_engineer import ContentEngineer
        import json
        import httpx
        
        # We run this synchronously; since LLMFactory and invoke are mostly synchronous here
        payload_str, provider_used, model_used, metadata = ContentEngineer.generate_blog_payload(
            topic=request.topic,
            provider=request.provider,
            model_name=model_name,
            prefer_model=prefer_model
        )
        
        # Finally Insert into the DB
        try:
            payload_dict = json.loads(payload_str)
            headers = {}
            if settings.TECHXBYTES_API_KEY:
                headers["Authorization"] = f"Bearer {settings.TECHXBYTES_API_KEY}"
            
            async with httpx.AsyncClient() as client:
                db_response = await client.post(
                    "https://techxbytes.com/api/blogs", 
                    json=payload_dict, 
                    headers=headers,
                    timeout=10.0
                )
                db_response.raise_for_status()
                # Optional: Add the creation status to metadata
                metadata["db_insertion"] = {"status": db_response.status_code, "success": True}
        except Exception as e:
            import sys
            sys.stderr.write(f"Warning: Failed to insert blog into database: {e}\n")
            metadata["db_insertion"] = {"status": "failed", "error": str(e), "success": False}
        
        return BlogResponse(
            blog_content=payload_str,
            llm_provider=provider_used,
            model=model_used,
            metadata=metadata
        ), 200

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import sys, traceback
        sys.stderr.write(f"Error processing blog request: {e}\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise HTTPException(status_code=500, detail="Internal Error")

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=settings.PORT
    )
