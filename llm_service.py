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
from typing import Tuple
from datetime import datetime
from zoneinfo import ZoneInfo
from prometheus_client import Counter

logger = logging.getLogger(__name__)

# Total LLM invocations, labeled by provider, model, and outcome (success/error).
# Powers the Grafana dashboard: totals-by-model, error percentage, and requests-over-time.
LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total LLM invocations via invoke_with_fallback",
    ["provider", "model", "status"],
)

# System prompt for the /api/chat surface (AI-Explain, Q&A answers). The frontend
# renders ```mermaid fenced blocks as SVG, so instruct the model to emit diagrams
# as text-based Mermaid when a question is architectural — never as an image.
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

# --- Gemini 2.5 Flash-Lite free tier: 15 RPM, 1000 RPD (resets midnight Pacific) ---
_FLASH_LITE_RPM = 15
_FLASH_LITE_RPD = 1000
_flash_lite_timestamps: list = []
_flash_lite_daily: dict = {}  # date_str (Pacific) -> count

# --- Full fallback chain: (provider, model_id). Prioritizes Vertex AI ($300 GCP Credits) -> Google AI Studio -> Groq -> OpenRouter -> Mistral ---
FALLBACK_CHAIN = (
    ("vertex", "gemini-2.5-flash"),
    ("google", "gemini-2.5-flash-lite"),
    ("groq", "llama-3.3-70b-versatile"),
    ("groq", "llama-3.1-8b-instant"),
    ("groq", "qwen/qwen3-32b"),
    ("groq", "moonshotai/kimi-k2-instruct-0905"),
    ("groq", "moonshotai/kimi-k2-instruct"),
    ("groq", "allam-2-7b"),
    ("groq", "groq/compound"),
    ("groq", "groq/compound-mini"),
    ("openrouter", "qwen/qwen3-coder:free"),
    ("mistral", "mistral-small-latest"),
)
# Selectable model IDs for prefer_model (same order as chain; for API docs/dropdown).
SELECTABLE_FALLBACK_MODELS = [model_id for _, model_id in FALLBACK_CHAIN]

# Groq sub-chain: (model_id, rpm, rpd) — indices align with groq entries in FALLBACK_CHAIN (2..9).
_GROQ_FALLBACK_CHAIN = (
    ("llama-3.3-70b-versatile", 30, 1000),
    ("llama-3.1-8b-instant", 30, 14400),
    ("qwen/qwen3-32b", 60, 1000),
    ("moonshotai/kimi-k2-instruct-0905", 60, 1000),
    ("moonshotai/kimi-k2-instruct", 60, 1000),
    ("allam-2-7b", 30, 7000),
    ("groq/compound", 30, 250),
    ("groq/compound-mini", 30, 250),
)
# Per-model state: model_id -> {"timestamps": [...], "daily": {date_str: count}}
_groq_chain_state: dict = {}

# --- OpenRouter free-tier fallback (after Groq chain exhausted). ---
_OPENROUTER_RPM = 20
_OPENROUTER_RPD = 1000
_openrouter_timestamps: list = []
_openrouter_daily: dict = {}

# --- Mistral AI fallback (after Groq chain exhausted). Free tier ~1 RPS, 500K TPM. ---
_MISTRAL_FALLBACK_RPM = 60
_MISTRAL_FALLBACK_RPD = 2000
_mistral_fallback_timestamps: list = []
_mistral_fallback_daily: dict = {}

_limiter_lock = threading.Lock()


def _pacific_date() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")


def _is_flash_lite_over_limit() -> bool:
    """True if we're at or over 15 RPM or 1000 RPD for Gemini 2.5 Flash-Lite."""
    with _limiter_lock:
        now = time.time()
        pacific_date = _pacific_date()
        global _flash_lite_timestamps, _flash_lite_daily
        _flash_lite_timestamps = [t for t in _flash_lite_timestamps if now - t < 60]
        if len(_flash_lite_timestamps) >= _FLASH_LITE_RPM:
            return True
        if _flash_lite_daily.get(pacific_date, 0) >= _FLASH_LITE_RPD:
            return True
        return False


def _record_flash_lite_usage() -> None:
    """Record one request for Gemini 2.5 Flash-Lite (call when using that model)."""
    with _limiter_lock:
        pacific_date = _pacific_date()
        _flash_lite_timestamps.append(time.time())
        _flash_lite_daily[pacific_date] = _flash_lite_daily.get(pacific_date, 0) + 1


def _is_groq_model_over_limit(model_id: str, rpm: int, rpd: int) -> bool:
    """True if this Groq model is at or over its RPM/RPD limit."""
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
    """Record one request for a Groq fallback model."""
    with _limiter_lock:
        pacific_date = _pacific_date()
        global _groq_chain_state
        if model_id not in _groq_chain_state:
            _groq_chain_state[model_id] = {"timestamps": [], "daily": {}}
        state = _groq_chain_state[model_id]
        state["timestamps"].append(time.time())
        state["daily"][pacific_date] = state["daily"].get(pacific_date, 0) + 1


def _pick_groq_fallback_model(groq_start_index: int = 0) -> Tuple[str, bool]:
    """
    Returns (model_id, should_record).
    First model in _GROQ_FALLBACK_CHAIN[groq_start_index:] that is under its limit; record usage.
    If all are over limit, returns last model in that slice and should_record=False.
    """
    chain_slice = _GROQ_FALLBACK_CHAIN[groq_start_index:]
    for model_id, rpm, rpd in chain_slice:
        if not _is_groq_model_over_limit(model_id, rpm, rpd):
            _record_groq_model_usage(model_id)
            return (model_id, True)
    if not chain_slice:
        return (_GROQ_FALLBACK_CHAIN[-1][0], False)
    return (chain_slice[-1][0], False)


def _is_openrouter_over_limit() -> bool:
    """True if OpenRouter free-tier usage is at or over 20 RPM / 1000 RPD."""
    with _limiter_lock:
        now = time.time()
        pacific_date = _pacific_date()
        global _openrouter_timestamps, _openrouter_daily
        _openrouter_timestamps = [t for t in _openrouter_timestamps if now - t < 60]
        if len(_openrouter_timestamps) >= _OPENROUTER_RPM:
            return True
        if _openrouter_daily.get(pacific_date, 0) >= _OPENROUTER_RPD:
            return True
        return False


def _record_openrouter_usage() -> None:
    """Record one request against the OpenRouter free-tier budget."""
    with _limiter_lock:
        pacific_date = _pacific_date()
        _openrouter_timestamps.append(time.time())
        _openrouter_daily[pacific_date] = _openrouter_daily.get(pacific_date, 0) + 1


def _get_openrouter_llm(model_name: str = "qwen/qwen3-coder:free") -> Tuple[BaseChatModel, str]:
    """Build an OpenRouter-backed ChatOpenAI instance (OpenAI-compatible API). Returns (llm, model_used)."""
    llm = ChatOpenAI(
        model=model_name,
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
        temperature=0.7,
    )
    return (llm, model_name)


def _is_mistral_fallback_over_limit() -> bool:
    """True if Mistral fallback usage is at or over 60 RPM / 2000 RPD."""
    with _limiter_lock:
        now = time.time()
        pacific_date = _pacific_date()
        global _mistral_fallback_timestamps, _mistral_fallback_daily
        _mistral_fallback_timestamps = [t for t in _mistral_fallback_timestamps if now - t < 60]
        if len(_mistral_fallback_timestamps) >= _MISTRAL_FALLBACK_RPM:
            return True
        if _mistral_fallback_daily.get(pacific_date, 0) >= _MISTRAL_FALLBACK_RPD:
            return True
        return False


def _record_mistral_fallback_usage() -> None:
    """Record one request for Mistral fallback."""
    with _limiter_lock:
        pacific_date = _pacific_date()
        _mistral_fallback_timestamps.append(time.time())
        _mistral_fallback_daily[pacific_date] = _mistral_fallback_daily.get(pacific_date, 0) + 1


def _get_mistral_llm(model_name: str = "mistral-small-latest") -> Tuple[BaseChatModel, str]:
    """Build ChatMistralAI instance. Returns (llm, model_used)."""
    llm = ChatMistralAI(
        model=model_name,
        mistral_api_key=settings.MISTRAL_API_KEY,
        temperature=0.7,
    )
    return (llm, model_name)


def _get_vertex_llm(model_name: str = "gemini-2.5-flash") -> Tuple[BaseChatModel, str]:
    """Build a Vertex AI-backed ChatVertexAI instance using GCP Project credentials ($300 credits)."""
    if not (settings.USE_VERTEX_AI and settings.GCP_PROJECT_ID):
        raise ValueError("Vertex AI is disabled or GCP_PROJECT_ID is not set.")
    if not VERTEX_AVAILABLE:
        raise ValueError("langchain_google_vertexai is not installed.")
    
    llm = ChatVertexAI(
        model_name=model_name,
        project=settings.GCP_PROJECT_ID,
        location=settings.GCP_LOCATION,
        temperature=0.7,
        max_retries=1,
    )
    return (llm, model_name)


def _fallback_chain_start_index(prefer_model: str | None) -> int:
    """Return index into FALLBACK_CHAIN to start from (0 = full auto). prefer_model is model_id."""
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
        """
        Factory method to get the appropriate LLM based on the provider.
        Prioritizes Vertex AI ($300 GCP Credits) -> Google AI Studio -> Groq -> OpenRouter -> Mistral.

        Args:
            provider: The provider ('google', 'vertex', 'anthropic', 'deepseek', 'groq', 'mistral', 'openrouter').
            model_name: The specific model name. If None, uses a default for the provider.
            prefer_model: Optional model id in FALLBACK_CHAIN to start from.

        Returns:
            Tuple[BaseChatModel, str, str]: (llm, provider_used, model_used).
        """
        provider = provider.lower()
        model_name = (model_name or "").strip() or None
        prefer_model = (prefer_model or "").strip() or None
        if prefer_model and prefer_model.lower() == "auto":
            prefer_model = None

        # Direct vertex request
        if provider == "vertex":
            model = model_name or "gemini-2.5-flash"
            llm, vertex_model = _get_vertex_llm(model)
            return (llm, "vertex", vertex_model)

        # Gemini flash / flash-lite flow: follow fallback chain starting from index
        if provider == "google" and (model_name in ("gemini-2.5-flash-lite", "gemini-2.5-flash", None)):
            start_index = _fallback_chain_start_index(prefer_model)
            for i in range(start_index, len(FALLBACK_CHAIN)):
                prov, mid = FALLBACK_CHAIN[i]
                
                # 1. Tier 1: Vertex AI (Consumes $300 GCP Credits)
                if prov == "vertex":
                    if settings.USE_VERTEX_AI and settings.GCP_PROJECT_ID and VERTEX_AVAILABLE:
                        try:
                            llm, vmodel = _get_vertex_llm(mid)
                            return (llm, "vertex", vmodel)
                        except Exception as ex:
                            logger.warning(f"Vertex AI initialization skipped ({ex}), falling back to Google AI Studio.")
                    continue

                # 2. Tier 2: Google AI Studio (Free Tier via API Key)
                if prov == "google":
                    if not _is_flash_lite_over_limit() and settings.GOOGLE_API_KEY:
                        _record_flash_lite_usage()
                        llm = ChatGoogleGenerativeAI(
                            model=mid,
                            google_api_key=settings.GOOGLE_API_KEY,
                            temperature=0.7,
                            max_retries=1,
                            convert_system_message_to_human=True,
                        )
                        return (llm, "google", mid)
                    continue

                # 3. Tier 3: Groq Free / High-Speed Sub-chain
                if prov == "groq" and settings.GROQ_API_KEY:
                    groq_start = max(0, i - 2)
                    groq_model, groq_ok = _pick_groq_fallback_model(groq_start)
                    if groq_ok:
                        llm = ChatGroq(
                            model_name=groq_model,
                            groq_api_key=settings.GROQ_API_KEY,
                            temperature=0.7,
                        )
                        return (llm, "groq", groq_model)
                    continue

                # 4. Tier 4: OpenRouter Free Models
                if prov == "openrouter" and settings.OPENROUTER_API_KEY:
                    if not _is_openrouter_over_limit():
                        _record_openrouter_usage()
                        llm, or_model = _get_openrouter_llm(mid)
                        return (llm, "openrouter", or_model)
                    continue

                # 5. Tier 5: Mistral AI Free Tier
                if prov == "mistral" and settings.MISTRAL_API_KEY:
                    if not _is_mistral_fallback_over_limit():
                        _record_mistral_fallback_usage()
                    llm, mistral_model = _get_mistral_llm()
                    return (llm, "mistral", mistral_model)

            # Fallbacks if loop exhausted
            if settings.USE_VERTEX_AI and settings.GCP_PROJECT_ID and VERTEX_AVAILABLE:
                try:
                    llm, vmodel = _get_vertex_llm("gemini-2.5-flash")
                    return (llm, "vertex", vmodel)
                except Exception:
                    pass

            if settings.GOOGLE_API_KEY:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-lite",
                    google_api_key=settings.GOOGLE_API_KEY,
                    temperature=0.7,
                    max_retries=1,
                    convert_system_message_to_human=True,
                )
                return (llm, "google", "gemini-2.5-flash-lite")

            if settings.GROQ_API_KEY:
                groq_model, _ = _pick_groq_fallback_model(0)
                llm = ChatGroq(
                    model_name=groq_model,
                    groq_api_key=settings.GROQ_API_KEY,
                    temperature=0.7,
                )
                return (llm, "groq", groq_model)

            raise ValueError("No viable LLM provider available. Check Vertex AI credentials or GOOGLE_API_KEY.")

        if provider == "google":
            # Direct non-flash google request: Try Vertex first if enabled
            if settings.USE_VERTEX_AI and settings.GCP_PROJECT_ID and VERTEX_AVAILABLE:
                try:
                    model = model_name or "gemini-2.5-flash"
                    llm, vertex_model = _get_vertex_llm(model)
                    return (llm, "vertex", vertex_model)
                except Exception as ex:
                    logger.warning(f"Vertex AI failed ({ex}), falling back to Google AI Studio.")

            if not settings.GOOGLE_API_KEY:
                raise ValueError("Google API Key is not set in environment or .env file.")
            model = model_name or "gemini-2.5-flash"
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0.7,
                max_retries=1,
                convert_system_message_to_human=True,
            )
            return (llm, "google", model)

        if provider == "groq":
            if not settings.GROQ_API_KEY:
                raise ValueError("Groq API Key is not set.")
            model = model_name or "llama-3.3-70b-versatile"
            llm = ChatGroq(
                model_name=model,
                groq_api_key=settings.GROQ_API_KEY,
                temperature=0.7,
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
            )
            return (llm, "deepseek", model)

        if provider == "mistral":
            if not settings.MISTRAL_API_KEY:
                raise ValueError("Mistral API Key is not set. Set MISTRAL_API_KEY in .env.")
            model = model_name or "mistral-small-latest"
            llm, _ = _get_mistral_llm(model)
            return (llm, "mistral", model)

        if provider == "openrouter":
            if not settings.OPENROUTER_API_KEY:
                raise ValueError("OpenRouter API Key is not set. Set OPENROUTER_API_KEY in .env.")
            model = model_name or "qwen/qwen3-coder:free"
            llm, _ = _get_openrouter_llm(model)
            return (llm, "openrouter", model)

        raise ValueError(f"Unsupported provider: {provider}. Supported: google, vertex, groq, anthropic, deepseek, mistral, openrouter.")


def _is_rate_limit_error(e: Exception) -> bool:
    """True on rate limits, quota limits, credit exhaustion, 403s, 429s, or 503s."""
    msg = str(e).lower()
    return (
        "resource_exhausted" in msg
        or "429" in msg
        or "rate limit" in msg
        or "503" in msg
        or "unavailable" in msg
        or "overloaded" in msg
        or "quota" in msg
        or "billing" in msg
        or "credit" in msg
        or "permission" in msg
        or "403" in msg
    )


def invoke_with_fallback(message: str, provider: str = "google", model_name: str = None, prefer_model: str = None, system_prompt: str = None):
    """
    Invoke an LLM, cascading through the *entire* remaining FALLBACK_CHAIN on real
    rate-limit / quota / billing failures.

    If ``system_prompt`` is provided, it is sent as a system message ahead of the
    user message.

    Returns (ai_message, provider_used, model_used).
    """
    llm, provider_used, model_used = LLMFactory.get_llm(provider, model_name, prefer_model=prefer_model)
    using_chain = (provider.lower() in ("google", "vertex") and (model_name in (None, "gemini-2.5-flash-lite", "gemini-2.5-flash")))

    # Build the invocation payload: a plain string, or system+human messages.
    payload = message if not system_prompt else [
        SystemMessage(content=system_prompt),
        HumanMessage(content=message),
    ]

    last_exc = None
    try:
        ai_message = llm.invoke(payload)
        LLM_REQUESTS_TOTAL.labels(provider=provider_used, model=model_used, status="success").inc()
        return (ai_message, provider_used, model_used)
    except Exception as e:
        last_exc = e
        LLM_REQUESTS_TOTAL.labels(provider=provider_used, model=model_used, status="error").inc()
        if not (using_chain and _is_rate_limit_error(e)):
            raise
        logger.warning(f"Primary invocation ({provider_used}:{model_used}) failed with {e}. Cascading fallback chain...")

    # Walk the rest of the chain, starting just after the model that failed.
    start_index = _fallback_chain_start_index(model_used) + 1
    for i in range(start_index, len(FALLBACK_CHAIN)):
        _prov, mid = FALLBACK_CHAIN[i]
        try:
            llm, provider_used, model_used = LLMFactory.get_llm(
                "google", "gemini-2.5-flash-lite", prefer_model=mid
            )
            ai_message = llm.invoke(payload)
            LLM_REQUESTS_TOTAL.labels(provider=provider_used, model=model_used, status="success").inc()
            return (ai_message, provider_used, model_used)
        except Exception as e:
            last_exc = e
            LLM_REQUESTS_TOTAL.labels(provider=provider_used, model=model_used, status="error").inc()
            if not _is_rate_limit_error(e):
                raise
            continue

    raise last_exc
