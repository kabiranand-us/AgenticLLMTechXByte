from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
from contextlib import asynccontextmanager

from config import settings
from llm_service import LLMFactory, SELECTABLE_FALLBACK_MODELS

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

# --- Lifespan Events ---
@asynccontextmanager
async def lifespan(application: FastAPI):
    # Startup logic: check if keys are present (optional warning)
    if not settings.GOOGLE_API_KEY:
        print("WARNING: GOOGLE_API_KEY not found. Gemini calls will fail.")
    yield
    # Shutdown logic if needed

# --- App Setup ---
app = FastAPI(
    title="Unified LLM Gateway",
    description="API Gateway for multiple LLM providers (Gemini, Groq, Claude, DeepSeek)",
    version="1.0.0",
    lifespan=lifespan
)

# --- CORS Middleware ---
# Allow all origins for development convenience. 
# In production, restrict this to the specific React app domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoints ---

@app.get("/health")
async def health_check():
    return {"status": "ok", "env": settings.APP_ENV}


@app.get("/api/models")
async def list_fallback_models():
    """Return the list of model IDs that can be used with prefer_model for the flash-lite fallback chain."""
    return {"prefer_model_options": SELECTABLE_FALLBACK_MODELS}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # 1. Get the appropriate LLM from the factory
        model_name = request.model_name
        if model_name == "string":
            model_name = None
        prefer_model = getattr(request, "prefer_model", None)
        if prefer_model == "string":
            prefer_model = None

        llm, provider_used, model_used = LLMFactory.get_llm(
            request.provider, model_name, prefer_model=prefer_model
        )
        
        # 2. Invoke the model
        # LangChain's invoke method returns an AIMessage object
        try:
            ai_message = llm.invoke(request.message)
        except Exception as e:
            error_msg = str(e).lower()
            if "resource_exhausted" in error_msg or "429" in error_msg:
                print(f"Rate limit hit for {provider_used} ({model_used}). Falling back to Groq...")
                # Fallback directly to a reliable Groq model with high rate limits
                llm, provider_used, model_used = LLMFactory.get_llm(
                    "groq", 
                    model_name="llama-3.1-8b-instant"
                )
                ai_message = llm.invoke(request.message)
            else:
                raise e
        
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

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=settings.PORT
    )
