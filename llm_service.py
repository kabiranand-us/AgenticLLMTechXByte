from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_core.language_models.chat_models import BaseChatModel
from config import settings
import os
import time
import threading
from typing import Tuple
from datetime import datetime
from zoneinfo import ZoneInfo
from prometheus_client import Counter

# Total LLM invocations, labeled by provider, model, and outcome (success/error).
# Powers the Grafana dashboard: totals-by-model, error percentage, and requests-over-time.
LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total LLM invocations via invoke_with_fallback",
    ["provider", "model", "status"],
)

# --- Gemini 2.5 Flash-Lite free tier: 15 RPM, 1000 RPD (resets midnight Pacific) ---
_FLASH_LITE_RPM = 15
_FLASH_LITE_RPD = 1000
_flash_lite_timestamps: list = []
_flash_lite_daily: dict = {}  # date_str (Pacific) -> count

# --- Full fallback chain: (provider, model_id). User can optionally start from any via prefer_model. ---
FALLBACK_CHAIN = (
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

# Groq sub-chain: (model_id, rpm, rpd) — indices align with groq entries in FALLBACK_CHAIN (1..8).
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

# --- OpenRouter free-tier fallback (after Groq chain exhausted). Conservative defaults;
# actual per-model limits vary, see https://openrouter.ai/docs/limits. ---
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
        When using Google with gemini-2.5-flash-lite, uses the fallback chain; optional
        prefer_model starts the chain from that model (e.g. "llama-3.1-8b-instant").

        Args:
            provider: The provider ('google', 'anthropic', 'deepseek', 'groq', 'mistral').
            model_name: The specific model name. If None, uses a default for the provider.
            prefer_model: Optional. For flash-lite flow, start fallback from this model id
                (one of SELECTABLE_FALLBACK_MODELS). Use "auto" or omit for full chain from Flash-Lite.

        Returns:
            Tuple[BaseChatModel, str, str]: (llm, provider_used, model_used).
        """
        provider = provider.lower()
        model_name = (model_name or "").strip() or None
        prefer_model = (prefer_model or "").strip() or None
        if prefer_model and prefer_model.lower() == "auto":
            prefer_model = None

        # Gemini 2.5 Flash-Lite with optional prefer_model: follow fallback chain from start_index
        if provider == "google" and (model_name == "gemini-2.5-flash-lite" or not model_name):
            start_index = _fallback_chain_start_index(prefer_model)
            for i in range(start_index, len(FALLBACK_CHAIN)):
                prov, mid = FALLBACK_CHAIN[i]
                if prov == "google":
                    if not _is_flash_lite_over_limit() and settings.GOOGLE_API_KEY:
                        _record_flash_lite_usage()
                        llm = ChatGoogleGenerativeAI(
                            model="gemini-2.5-flash-lite",
                            google_api_key=settings.GOOGLE_API_KEY,
                            temperature=0.7,
                            max_retries=1,
                            convert_system_message_to_human=True,
                        )
                        return (llm, "google", "gemini-2.5-flash-lite")
                    continue
                if prov == "groq" and settings.GROQ_API_KEY:
                    groq_start = i - 1  # first groq in FALLBACK_CHAIN is at index 1
                    groq_model, groq_ok = _pick_groq_fallback_model(groq_start)
                    if groq_ok:
                        llm = ChatGroq(
                            model_name=groq_model,
                            groq_api_key=settings.GROQ_API_KEY,
                            temperature=0.7,
                        )
                        return (llm, "groq", groq_model)
                    # Groq slice exhausted; continue to Mistral if in chain
                    continue
                if prov == "openrouter" and settings.OPENROUTER_API_KEY:
                    if not _is_openrouter_over_limit():
                        _record_openrouter_usage()
                        llm, or_model = _get_openrouter_llm(mid)
                        return (llm, "openrouter", or_model)
                    continue
                if prov == "mistral" and settings.MISTRAL_API_KEY:
                    if not _is_mistral_fallback_over_limit():
                        _record_mistral_fallback_usage()
                    llm, mistral_model = _get_mistral_llm()
                    return (llm, "mistral", mistral_model)
            # No step succeeded: try Mistral if we have key (e.g. no Groq key)
            if settings.MISTRAL_API_KEY:
                if not _is_mistral_fallback_over_limit():
                    _record_mistral_fallback_usage()
                llm, mistral_model = _get_mistral_llm()
                return (llm, "mistral", mistral_model)
            if settings.GROQ_API_KEY:
                groq_model, _ = _pick_groq_fallback_model(0)
                llm = ChatGroq(
                    model_name=groq_model,
                    groq_api_key=settings.GROQ_API_KEY,
                    temperature=0.7,
                )
                return (llm, "groq", groq_model)
            if not settings.GOOGLE_API_KEY:
                raise ValueError("Google API Key is not set. Cannot use Gemini 2.5 Flash-Lite.")
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0.7,
                max_retries=1,
                convert_system_message_to_human=True,
            )
            return (llm, "google", "gemini-2.5-flash-lite")

        if provider == "google":
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
            model = model_name or "llama-3.3-70b-versatile"  # mixtral-8x7b-32768 was decommissioned
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

        raise ValueError(f"Unsupported provider: {provider}. Supported: google, groq, anthropic, deepseek, mistral, openrouter.")


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return (
        "resource_exhausted" in msg
        or "429" in msg
        or "rate limit" in msg
        or "503" in msg
        or "unavailable" in msg
        or "overloaded" in msg
    )


def invoke_with_fallback(message: str, provider: str = "google", model_name: str = None, prefer_model: str = None):
    """
    Invoke an LLM, cascading through the *entire* remaining FALLBACK_CHAIN on real
    rate-limit failures (not just one hardcoded retry step). Only meaningful when
    using the google/gemini-2.5-flash-lite entry point, since that's what drives
    the chain; other providers are tried once as-is.

    Returns (ai_message, provider_used, model_used).
    """
    llm, provider_used, model_used = LLMFactory.get_llm(provider, model_name, prefer_model=prefer_model)
    using_chain = (provider.lower() == "google" and (model_name in (None, "gemini-2.5-flash-lite")))

    last_exc = None
    try:
        ai_message = llm.invoke(message)
        LLM_REQUESTS_TOTAL.labels(provider=provider_used, model=model_used, status="success").inc()
        return (ai_message, provider_used, model_used)
    except Exception as e:
        last_exc = e
        LLM_REQUESTS_TOTAL.labels(provider=provider_used, model=model_used, status="error").inc()
        if not (using_chain and _is_rate_limit_error(e)):
            raise

    # Walk the rest of the chain, starting just after the model that just failed.
    start_index = _fallback_chain_start_index(model_used) + 1
    for i in range(start_index, len(FALLBACK_CHAIN)):
        _prov, mid = FALLBACK_CHAIN[i]
        try:
            llm, provider_used, model_used = LLMFactory.get_llm(
                "google", "gemini-2.5-flash-lite", prefer_model=mid
            )
            ai_message = llm.invoke(message)
            LLM_REQUESTS_TOTAL.labels(provider=provider_used, model=model_used, status="success").inc()
            return (ai_message, provider_used, model_used)
        except Exception as e:
            last_exc = e
            LLM_REQUESTS_TOTAL.labels(provider=provider_used, model=model_used, status="error").inc()
            if not _is_rate_limit_error(e):
                raise
            continue

    raise last_exc
