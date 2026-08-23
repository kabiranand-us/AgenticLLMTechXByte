import os
import time
import logging
import threading
from typing import Tuple, Optional, Any
from datetime import datetime
from zoneinfo import ZoneInfo
from prometheus_client import Counter

from google import genai
from google.genai import types
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from config import settings

logger = logging.getLogger(__name__)

LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total LLM invocations via invoke_with_fallback",
    ["provider", "model", "status"],
)

CHAT_SYSTEM_PROMPT = (
    "You are a helpful software engineering assistant. Answer in clear, well-structured "
    "Markdown (short paragraphs, bullet lists, and `inline code` where useful).\n\n"
    "When the answer involves system design, architecture, request/data flow, sequences, "
    "state machines, or entity relationships, include a diagram as a fenced ```mermaid code "
    "block. Rules for diagrams:\n"
    "- Pick the fitting type: flowchart, sequenceDiagram, stateDiagram-v2, erDiagram, or C4Context.\n"
    "- Keep node labels short and use plain ASCII only. Do NOT put parentheses (), angle "
    "brackets <>, quotes, or emojis inside labels — they break the Mermaid parser.\n"
    "- Place the diagram alongside the explanation, not only at the very end.\n"
    "- Only include a diagram when it genuinely aids understanding.\n"
    "- Never output an image, image link, or base64 image — diagrams must be Mermaid text.\n\n"
    "For cloud infrastructure or service-topology diagrams, prefer an `architecture-beta` "
    "diagram with icons:\n"
    "- Syntax: `service <id>(<icon>)[<Label>] in <group>`; connect nodes with edges like "
    "`a:R --> L:b` (sides are T/B/L/R).\n"
    "- The `(icon)` parenthesis is REQUIRED architecture-beta syntax and is allowed here — the "
    "no-parentheses rule above applies only to text inside the `[ ]` label.\n"
    "- Icons: built-in `cloud`, `database`, `disk`, `internet`, `server`, or brand icons from the "
    "`logos` pack, e.g. `logos:aws`, `logos:aws-s3`, `logos:aws-lambda`, `logos:aws-ec2`, "
    "`logos:aws-dynamodb`, `logos:aws-cloudformation`, `logos:docker`, `logos:kubernetes`, "
    "`logos:redis`, `logos:mongodb`, `logos:react`.\n"
    "- Example:\n"
    "```mermaid\n"
    "architecture-beta\n"
    "  group cloud(logos:aws)[AWS]\n"
    "  service s3(logos:aws-s3)[Storage] in cloud\n"
    "  service fn(logos:aws-lambda)[API] in cloud\n"
    "  service db(logos:aws-dynamodb)[Database] in cloud\n"
    "  s3:R --> L:fn\n"
    "  fn:R --> L:db\n"
    "```"
)

# Active fallback chain: 1. Vertex AI ($300 GCP Credits) -> 2. Google AI Studio -> 3. Groq
FALLBACK_CHAIN = (
    ("vertex", "gemini-2.5-flash"),
    ("vertex", "gemini-2.5-pro"),
    ("groq", "allam-2-7b"),
    ("groq", "qwen/qwen3.6-27b"),
    ("groq", "groq/compound"),
    ("openrouter", "qwen/qwen3-coder:free"),
    ("mistral", "mistral-small-latest"),
)
SELECTABLE_FALLBACK_MODELS = [model_id for _, model_id in FALLBACK_CHAIN]

_GROQ_FALLBACK_CHAIN = (
    ("allam-2-7b", 60, 7000),
    ("qwen/qwen3.6-27b", 60, 1000),
    ("groq/compound", 30, 250),
)
_groq_chain_state: dict = {}
_limiter_lock = threading.Lock()


def _pacific_date() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")


def _is_groq_model_over_limit(model_id: str, rpm: int, rpd: int) -> bool:
    with _limiter_lock:
        now = time.time()
        pacific_date = _pacific_date()
        global _groq_chain_state
        if model_id not in _groq_chain_state:
            return False
        state = _groq_chain_state[model_id]
        state["timestamps"] = [t for t in state["timestamps"] if now - t < 60]
        if len(state["timestamps"]) >= rpm:
            return True
        if state["daily"].get(pacific_date, 0) >= rpd:
            return True
        return False


def _record_groq_model_usage(model_id: str) -> None:
    with _limiter_lock:
        pacific_date = _pacific_date()
        global _groq_chain_state
        if model_id not in _groq_chain_state:
            _groq_chain_state[model_id] = {"timestamps": [], "daily": {}}
        state = _groq_chain_state[model_id]
        state["timestamps"].append(time.time())
        state["daily"][pacific_date] = state["daily"].get(pacific_date, 0) + 1


def _pick_groq_fallback_model(groq_start_index: int = 0) -> Tuple[str, bool]:
    chain_slice = _GROQ_FALLBACK_CHAIN[groq_start_index:]
    for model_id, rpm, rpd in chain_slice:
        if not _is_groq_model_over_limit(model_id, rpm, rpd):
            _record_groq_model_usage(model_id)
            return (model_id, True)
    if not chain_slice:
        return (_GROQ_FALLBACK_CHAIN[0][0], False)
    return (chain_slice[0][0], False)


class NativeGenAIWrapper:
    """Fast, direct Google GenAI SDK wrapper supporting Vertex AI ($300 credits) and AI Studio."""
    def __init__(self, model_name: str, vertexai: bool = False):
        self.model_name = model_name
        self.vertexai = vertexai
        if vertexai:
            self.client = genai.Client(
                vertexai=True,
                project=settings.GCP_PROJECT_ID,
                location=settings.GCP_LOCATION
            )
        else:
            key = settings.resolved_google_api_key
            if not key:
                raise ValueError("Google API key missing.")
            self.client = genai.Client(api_key=key)

    def invoke(self, payload: Any) -> AIMessage:
        if isinstance(payload, list):
            # Combine System & Human messages
            sys_text = ""
            user_text = ""
            for msg in payload:
                if isinstance(msg, SystemMessage):
                    sys_text += msg.content + "\n\n"
                elif isinstance(msg, HumanMessage):
                    user_text += msg.content
                elif hasattr(msg, "content"):
                    user_text += str(msg.content)
            prompt = (sys_text + user_text).strip()
        else:
            prompt = str(payload)

        config = types.GenerateContentConfig(
            temperature=0.7,
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        usage = getattr(response, "usage_metadata", None)
        token_meta = {}
        if usage:
            token_meta = {
                "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                "completion_tokens": getattr(usage, "candidates_token_count", 0),
                "total_tokens": getattr(usage, "total_token_count", 0),
            }
        return AIMessage(
            content=response.text or "",
            response_metadata={"token_usage": token_meta, "model_name": self.model_name}
        )


def _get_openrouter_llm(model_name: str = "qwen/qwen3-coder:free") -> Tuple[BaseChatModel, str]:
    llm = ChatOpenAI(
        model=model_name,
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
        temperature=0.7,
        timeout=5.0,
        max_retries=0,
    )
    return (llm, model_name)


def _get_mistral_llm(model_name: str = "mistral-small-latest") -> Tuple[BaseChatModel, str]:
    llm = ChatMistralAI(
        model=model_name,
        mistral_api_key=settings.MISTRAL_API_KEY,
        temperature=0.7,
        timeout=5.0,
        max_retries=0,
    )
    return (llm, model_name)


class LLMFactory:
    @staticmethod
    def get_llm(
        provider: str = "google",
        model_name: str = None,
        prefer_model: str = None,
    ) -> Tuple[Any, str, str]:
        provider = provider.lower()
        model_name = (model_name or "").strip() or None
        prefer_model = (prefer_model or "").strip() or None
        if prefer_model and prefer_model.lower() == "auto":
            prefer_model = None

        # 1. Vertex AI ($300 GCP Credits)
        if provider == "vertex" or (provider in ("google", "default") and settings.USE_VERTEX_AI and settings.GCP_PROJECT_ID):
            try:
                vmodel = model_name or "gemini-2.5-flash"
                wrapper = NativeGenAIWrapper(model_name=vmodel, vertexai=True)
                return (wrapper, "vertex", vmodel)
            except Exception as ex:
                logger.warning(f"Vertex AI initialization failed ({ex}), falling back to Google AI Studio / Groq.")

        # 2. Google AI Studio (Free Tier)
        if provider == "google" and settings.resolved_google_api_key:
            try:
                gmodel = model_name or "gemini-2.5-flash"
                wrapper = NativeGenAIWrapper(model_name=gmodel, vertexai=False)
                return (wrapper, "google", gmodel)
            except Exception as ex:
                logger.warning(f"Google AI Studio initialization failed ({ex}).")

        # 3. Groq (Ultra-Fast ~0.5s)
        if (provider in ("groq", "google", "default")) and settings.GROQ_API_KEY:
            groq_model, _ = _pick_groq_fallback_model(0)
            llm = ChatGroq(
                model_name=groq_model,
                groq_api_key=settings.GROQ_API_KEY,
                temperature=0.7,
                timeout=5.0,
                max_retries=0,
            )
            return (llm, "groq", groq_model)

        if provider == "anthropic":
            if not settings.ANTHROPIC_API_KEY:
                raise ValueError("Anthropic API Key is not set.")
            model = model_name or "claude-3-opus-20240229"
            llm = ChatAnthropic(
                model=model,
                anthropic_api_key=settings.ANTHROPIC_API_KEY,
                temperature=0.7,
                timeout=8.0,
                max_retries=0,
            )
            return (llm, "anthropic", model)

        if provider == "deepseek":
            if not settings.DEEPSEEK_API_KEY:
                raise ValueError("DeepSeek API Key is not set.")
            model = model_name or "deepseek-chat"
            llm = ChatOpenAI(
                model=model,
                api_key=settings.DEEPSEEK_API_KEY,
                base_url=settings.DEEPSEEK_BASE_URL,
                temperature=0.7,
                timeout=8.0,
                max_retries=0,
            )
            return (llm, "deepseek", model)

        if provider == "mistral":
            if not settings.MISTRAL_API_KEY:
                raise ValueError("Mistral API Key is not set.")
            model = model_name or "mistral-small-latest"
            llm, _ = _get_mistral_llm(model)
            return (llm, "mistral", model)

        if provider == "openrouter":
            if not settings.OPENROUTER_API_KEY:
                raise ValueError("OpenRouter API Key is not set.")
            model = model_name or "qwen/qwen3-coder:free"
            llm, _ = _get_openrouter_llm(model)
            return (llm, "openrouter", model)

        raise ValueError(f"Unsupported provider: {provider}.")


def invoke_with_fallback(message: str, provider: str = "google", model_name: str = None, prefer_model: str = None, system_prompt: str = None):
    payload = message if not system_prompt else [
        SystemMessage(content=system_prompt),
        HumanMessage(content=message),
    ]

    last_exc = None
    # 1. Try primary configured model (Vertex AI with $300 Credits)
    try:
        llm, provider_used, model_used = LLMFactory.get_llm(provider, model_name, prefer_model=prefer_model)
        ai_message = llm.invoke(payload)
        LLM_REQUESTS_TOTAL.labels(provider=provider_used, model=model_used, status="success").inc()
        return (ai_message, provider_used, model_used)
    except Exception as e:
        last_exc = e
        LLM_REQUESTS_TOTAL.labels(provider=provider or "vertex", model=model_name or "gemini-2.5-flash", status="error").inc()
        logger.warning(f"Primary ({provider}:{model_name}) failed with {type(e).__name__}: {e}. Cascading fallback...")

    # 2. Cascade down the fallback tiers
    for prov, mid in FALLBACK_CHAIN:
        try:
            llm, provider_used, model_used = LLMFactory.get_llm(prov, mid)
            ai_message = llm.invoke(payload)
            LLM_REQUESTS_TOTAL.labels(provider=provider_used, model=model_used, status="success").inc()
            return (ai_message, provider_used, model_used)
        except Exception as e:
            last_exc = e
            LLM_REQUESTS_TOTAL.labels(provider=prov, model=mid, status="error").inc()
            logger.warning(f"Fallback ({prov}:{mid}) failed with {type(e).__name__}: {e}.")
            continue

    raise last_exc or RuntimeError("All LLM providers and fallback tiers failed.")
