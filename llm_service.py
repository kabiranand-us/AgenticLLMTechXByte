from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain_google_vertexai import ChatVertexAI
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from config import settings
import os
import time
import logging
import threading
from typing import Tuple, Optional
from datetime import datetime
from zoneinfo import ZoneInfo
from prometheus_client import Counter

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

# Active, verified models ordered by speed and capability
FALLBACK_CHAIN = (
    ("google", "gemini-2.5-flash"),
    ("groq", "allam-2-7b"),
    ("groq", "qwen/qwen3.6-27b"),
    ("groq", "groq/compound"),
    ("groq", "groq/compound-mini"),
    ("openrouter", "qwen/qwen3-coder:free"),
    ("mistral", "mistral-small-latest"),
)
SELECTABLE_FALLBACK_MODELS = [model_id for _, model_id in FALLBACK_CHAIN]

# Groq active sub-chain
_GROQ_FALLBACK_CHAIN = (
    ("allam-2-7b", 60, 7000),
    ("qwen/qwen3.6-27b", 60, 1000),
    ("groq/compound", 30, 250),
    ("groq/compound-mini", 30, 250),
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


def _get_vertex_llm(model_name: str = "gemini-2.5-flash") -> Tuple[BaseChatModel, str]:
    if not (settings.USE_VERTEX_AI and settings.GCP_PROJECT_ID and VERTEX_AVAILABLE):
        raise ValueError("Vertex AI is disabled, unconfigured, or package not available.")
    
    # Fast timeout to prevent ADC discovery hangs
    llm = ChatVertexAI(
        model_name=model_name,
        project=settings.GCP_PROJECT_ID,
        location=settings.GCP_LOCATION,
        temperature=0.7,
        timeout=4.0,
        max_retries=0,
    )
    return (llm, model_name)


def _fallback_chain_start_index(prefer_model: str | None) -> int:
    if not (prefer_model and (prefer_model := prefer_model.strip())):
        return 0
    for i, (_prov, model_id) in enumerate(FALLBACK_CHAIN):
        if model_id == prefer_model:
            return i
    return 0


class LLMFactory:
    @staticmethod
    def get_llm(
        provider: str = "google",
        model_name: str = None,
        prefer_model: str = None,
    ) -> Tuple[BaseChatModel, str, str]:
        provider = provider.lower()
        model_name = (model_name or "").strip() or None
        prefer_model = (prefer_model or "").strip() or None
        if prefer_model and prefer_model.lower() == "auto":
            prefer_model = None

        google_key = settings.resolved_google_api_key

        # 1. Direct Vertex AI request (Fast timeout)
        if provider == "vertex":
            model = model_name or "gemini-2.5-flash"
            llm, vertex_model = _get_vertex_llm(model)
            return (llm, "vertex", vertex_model)

        # 2. Google / Default chain
        if provider in ("google", "default") and (model_name is None or "gemini" in model_name):
            start_index = _fallback_chain_start_index(prefer_model)
            for i in range(start_index, len(FALLBACK_CHAIN)):
                prov, mid = FALLBACK_CHAIN[i]

                # Google AI Studio (Fast 4s timeout, max_retries 0)
                if prov == "google" and google_key:
                    try:
                        llm = ChatGoogleGenerativeAI(
                            model=mid,
                            google_api_key=google_key,
                            temperature=0.7,
                            timeout=4.0,
                            max_retries=0,
                            convert_system_message_to_human=True,
                        )
                        return (llm, "google", mid)
                    except Exception as ex:
                        logger.warning(f"Google AI Studio model {mid} init failed ({ex}).")
                    continue

                # Groq Active Models (Lightning Fast ~0.5s)
                if prov == "groq" and settings.GROQ_API_KEY:
                    groq_model, _ = _pick_groq_fallback_model(0)
                    llm = ChatGroq(
                        model_name=groq_model,
                        groq_api_key=settings.GROQ_API_KEY,
                        temperature=0.7,
                        timeout=5.0,
                        max_retries=0,
                    )
                    return (llm, "groq", groq_model)

                # OpenRouter
                if prov == "openrouter" and settings.OPENROUTER_API_KEY:
                    llm, or_model = _get_openrouter_llm(mid)
                    return (llm, "openrouter", or_model)

                # Mistral
                if prov == "mistral" and settings.MISTRAL_API_KEY:
                    llm, mistral_model = _get_mistral_llm()
                    return (llm, "mistral", mistral_model)

            # Fallback to Groq if Google key missing
            if settings.GROQ_API_KEY:
                groq_model, _ = _pick_groq_fallback_model(0)
                llm = ChatGroq(
                    model_name=groq_model,
                    groq_api_key=settings.GROQ_API_KEY,
                    temperature=0.7,
                    timeout=5.0,
                    max_retries=0,
                )
                return (llm, "groq", groq_model)

        if provider == "google":
            if not google_key:
                raise ValueError("Google API Key is not configured.")
            model = model_name or "gemini-2.5-flash"
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=google_key,
                temperature=0.7,
                timeout=4.0,
                max_retries=0,
                convert_system_message_to_human=True,
            )
            return (llm, "google", model)

        if provider == "groq":
            if not settings.GROQ_API_KEY:
                raise ValueError("Groq API Key is not set.")
            model = model_name or "allam-2-7b"
            llm = ChatGroq(
                model_name=model,
                groq_api_key=settings.GROQ_API_KEY,
                temperature=0.7,
                timeout=5.0,
                max_retries=0,
            )
            return (llm, "groq", model)

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
    try:
        llm, provider_used, model_used = LLMFactory.get_llm(provider, model_name, prefer_model=prefer_model)
        ai_message = llm.invoke(payload)
        LLM_REQUESTS_TOTAL.labels(provider=provider_used, model=model_used, status="success").inc()
        return (ai_message, provider_used, model_used)
    except Exception as e:
        last_exc = e
        LLM_REQUESTS_TOTAL.labels(provider=provider or "google", model=model_name or "default", status="error").inc()
        logger.warning(f"Primary ({provider}:{model_name}) failed with {type(e).__name__}: {e}. Cascading fallback...")

    # Fast fallback through active tiers
    start_index = _fallback_chain_start_index(prefer_model)
    for i in range(start_index, len(FALLBACK_CHAIN)):
        _prov, mid = FALLBACK_CHAIN[i]
        try:
            llm, provider_used, model_used = LLMFactory.get_llm("google", None, prefer_model=mid)
            ai_message = llm.invoke(payload)
            LLM_REQUESTS_TOTAL.labels(provider=provider_used, model=model_used, status="success").inc()
            return (ai_message, provider_used, model_used)
        except Exception as e:
            last_exc = e
            LLM_REQUESTS_TOTAL.labels(provider=_prov, model=mid, status="error").inc()
            logger.warning(f"Fallback ({_prov}:{mid}) failed with {type(e).__name__}: {e}.")
            continue

    raise last_exc or RuntimeError("All LLM providers and fallback tiers failed.")
