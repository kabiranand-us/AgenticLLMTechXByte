# Changelog

## 2026-06-24 — Fail fast on Gemini 503s instead of ~40s of internal SDK backoff

**Commit:** [`45501cd`](https://github.com/kabiranand-us/AgenticLLMTechXByte/commit/45501cd) (this entry documents that already-shipped change for the record)

### Problem

A production request to `/api/chat` took **9.96 seconds** to return a `200 OK` — unusually slow
for a single chat call. Checking `docker logs llm-gateway` for that window showed the real cause:

```
HTTP/1.1 503 Service Unavailable  (gemini-2.5-flash-lite "high demand")
  -> retry in 1.95s
HTTP/1.1 503 Service Unavailable
  -> retry in 2.5s
HTTP/1.1 503 Service Unavailable
  -> retry in 4.75s
HTTP/1.1 503 Service Unavailable
  -> retry in 8.87s
HTTP/1.1 200 OK   (5th attempt, ~40s after the first)
```

Gemini was transiently overloaded and returning `503`, which is unrelated to our own quota/rate
limiting. The underlying `google-genai` SDK has its own internal retry loop with exponential
backoff, and by default retried the *same* model up to 4 times — roughly 40 seconds — before
giving control back to our code. Our own `invoke_with_fallback()` chain (added in
[`7e74dfe`](https://github.com/kabiranand-us/AgenticLLMTechXByte/commit/7e74dfe)) never got a
chance to fail over to Groq or OpenRouter during that window, because from its point of view the
call to `llm.invoke()` simply hadn't returned yet.

This directly worked against the goal of maximizing successful free-tier completions: a free,
fast alternative (Groq, ~1-2s typical latency) was sitting unused while the request sat blocked
on Gemini's own retry loop for an overloaded model.

### Fix

In `llm_service.py`:

1. **Capped `max_retries=1`** on all three `ChatGoogleGenerativeAI` instantiations. The SDK no
   longer burns ~40s retrying a single overloaded model — it surfaces the error after one retry,
   letting our own fallback chain take over almost immediately.
2. **Widened `_is_rate_limit_error()`** to also match `503`, `unavailable`, and `overloaded` —
   previously it only treated `resource_exhausted`/`429`/`rate limit` as fallback-worthy, so a
   `503` overload error wasn't recognized as something `invoke_with_fallback()` should cascade
   past.

### Result

A Gemini `503` now surfaces in ~1-2s (one retry) instead of ~40s (four retries), and
`invoke_with_fallback()` immediately tries the next model in `FALLBACK_CHAIN` (Groq, then
OpenRouter, then Mistral) rather than waiting on Gemini's own backoff schedule.
