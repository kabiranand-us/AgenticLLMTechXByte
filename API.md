# API Documentation

This document explicitly defines all the API endpoints provided by the AgenticAI Orchestrator. The base URL runs on `http://127.0.0.1:8000/` or your assigned remote domain.

## How to Run

To start the FastAPI application locally, activate the virtual environment and run the uvicorn server:

```bash
source .venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 1. Create Blog
Executes the Autonomous Content Engineer pipeline. It generates HTML-formatted content by LLM, JSON structures it, and automatically injects it into the TechXBytes DB using a locally configured Authorization header.

**Endpoint:** `/api/create/blog`
**Method:** `POST`

### Request Body (`application/json`)
```json
{
  "topic": "What is Node.js and explain its event-driven architecture.",
  "provider": "google",
  "model_name": null,
  "temperature": 0.7,
  "prefer_model": null
}
```
*Note: `provider` defaults to "google" but can accept "groq", "anthropic", "deepseek", or "mistral".*

### Response Body (`application/json` or `201 Created` DB cascade)
```json
{
  "blog_content": "[raw generated html text formatted json string that was inserted]",
  "llm_provider": "google",
  "model": "gemini-2.5-flash-lite",
  "metadata": {
    "db_insertion": {
      "status": 201,
      "success": true
    }
  }
}
```

---

## 2. Core LLM Chat Gateway
Routes prompt messages to configured AI providers with built-in rate-limit fallback capabilities.

**Endpoint:** `/api/chat`
**Method:** `POST`

### Request Body (`application/json`)
```json
{
  "message": "Explain quantum computing briefly.",
  "provider": "google",
  "model_name": null,
  "temperature": 0.7,
  "prefer_model": null
}
```

### Response Body (`application/json`)
```json
{
  "response": "Quantum computing is an area of study...",
  "llm_provider": "google",
  "model": "gemini-2.5-flash",
  "metadata": {}
}
```

---

## 3. Ollama SaladCloud Manager
Dynamically provisions and routes chat prompt generations to a designated SaladCloud containerized Ollama deployment model. It intelligently handles idle delays.

**Endpoint:** `/api/ollama/chat`
**Method:** `POST`

### Request Body (`application/json`)
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Write a short poem."
    }
  ],
  "model": "llama3.2",
  "stream": false
}
```

### Response Body (`application/json`)
```json
{
  "model": "llama3.2",
  "created_at": "2026-03-11T12:00:00.0000Z",
  "message": {
    "role": "assistant",
    "content": "Roses are red..."
  },
  "done": true
}
```

---

## 4. Get Fallback Models
Returns the operational schema of models currently available in the system's `prefer_model` parameter cascading chain.

**Endpoint:** `/api/models`
**Method:** `GET`

### Request Body
*None*

### Response Body (`application/json`)
```json
{
  "prefer_model_options": [
    "gemini-2.5-flash-lite",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct-0905",
    "moonshotai/kimi-k2-instruct",
    "allam-2-7b",
    "groq/compound",
    "groq/compound-mini",
    "mistral-small-latest"
  ]
}
```

---

## 5. Health Check
Verifies that the API Gateway application is operational.

**Endpoint:** `/health`
**Method:** `GET`

### Request Body
*None*

### Response Body (`application/json`)
```json
{
  "status": "ok",
  "env": "development"
}
```
