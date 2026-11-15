import json
import requests

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODELS_URL = f"{GROQ_BASE_URL}/models"
GROQ_CHAT_URL = f"{GROQ_BASE_URL}/chat/completions"
# The previous default (llama-3.3-70b-versatile) was shut down by Groq in
# August 2026. Prefer its documented replacement, with current alternatives.
MODEL_PREFERENCE = (
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
)
MODEL = MODEL_PREFERENCE[0]


def _error_message(response, api_key=""):
    try:
        body = response.json()
        message = body.get("error", {}).get("message") or body.get("message") or str(body)
    except (ValueError, AttributeError):
        message = response.text[:1000]
    message = str(message).replace(api_key, "[redacted]") if api_key else str(message)
    return f"Groq returned HTTP {response.status_code}: {message[:1000]}"


def _models(api_key):
    try:
        response = requests.get(GROQ_MODELS_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
    except requests.RequestException as exc:
        raise ValueError(f"Could not connect to Groq models endpoint ({exc.__class__.__name__}).") from None
    if not response.ok:
        raise ValueError(_error_message(response, api_key))
    try:
        data = response.json().get("data", [])
        return {str(item["id"]) for item in data if isinstance(item, dict) and item.get("id")}
    except (ValueError, TypeError, KeyError):
        raise ValueError("Groq models endpoint returned an invalid response.") from None


def choose_model(api_key):
    models = _models(api_key)
    for candidate in MODEL_PREFERENCE:
        if candidate in models:
            return candidate
    # The models endpoint lists models enabled for this key. Exclude known
    # non-chat families before using a last-resort available ID.
    non_chat = ("whisper", "speech", "tts", "embedding", "guard", "moderation")
    alternatives = sorted(m for m in models if not any(tag in m.lower() for tag in non_chat))
    if alternatives:
        return alternatives[0]
    raise ValueError("Groq key is valid, but no chat-capable model is available to it.")


def groq_chat(api_key, system, user, temperature=0):
    global MODEL
    # Resolve against this key's actual model catalog. No key or key-derived
    # identifier is cached or written to disk.
    MODEL = choose_model(api_key)
    payload = {"model": MODEL, "temperature": temperature, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    try:
        response = requests.post(GROQ_CHAT_URL, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=payload, timeout=60)
    except requests.RequestException as exc:
        raise ValueError(f"Could not connect to Groq chat completions ({exc.__class__.__name__}).") from None
    if not response.ok:
        message = _error_message(response, api_key)
        if response.status_code == 429:
            raise ValueError("Groq rate limit reached. " + message)
        raise ValueError(message)
    try:
        return response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise ValueError("Groq chat completion returned an invalid response structure.") from None


def json_from_response(text):
    if not isinstance(text, str):
        raise ValueError("The AI response was not text.")
    text = text.strip().lstrip("\ufeff")
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for opening, closing in (("{", "}"), ("[", "]")):
            start, end = text.find(opening), text.rfind(closing)
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue
        raise ValueError("The AI returned an unreadable response. Please retry.") from None


def validate_groq_key(api_key):
    global MODEL
    try:
        model = choose_model(api_key)
        MODEL = model
        return True, f"Key validated; chat model available: {model}."
    except ValueError as exc:
        return False, str(exc)
