from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from urllib import error, request

from app.config import get_settings
from app.persistence import atomic_write_json, file_lock


_active_chats: set[str] = set()
_active_lock = threading.Lock()

ASSISTANT_INSTRUCTIONS = (
    "You are the private assistant for the owner of a forex and metals signal bot. "
    "Reply clearly and concisely for Telegram. You may explain, plan, analyze, and answer questions, "
    "but you cannot directly operate the server, deploy code, or claim that an external action was completed. "
    "When an action requires system access, say exactly what needs to be done. Never reveal credentials or secrets."
)


def _api_call(method: str, path: str, payload: dict | None = None) -> dict:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    api_request = request.Request(
        f"https://api.openai.com/v1{path}", data=body, headers=headers, method=method
    )
    try:
        with request.urlopen(api_request, timeout=45) as response:
            return json.loads(response.read())
    except error.HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("error", {}).get("message")
        except (json.JSONDecodeError, AttributeError):
            detail = None
        raise RuntimeError(detail or f"OpenAI API returned HTTP {exc.code}") from exc


def _state() -> dict:
    path = Path(get_settings().telegram_assistant_state_path)
    try:
        return json.loads(path.read_text()) if path.exists() else {"conversations": {}}
    except (OSError, json.JSONDecodeError):
        return {"conversations": {}}


def _conversation_id(chat_id: str) -> str:
    path = Path(get_settings().telegram_assistant_state_path)
    with file_lock(path):
        state = _state()
        existing = state.setdefault("conversations", {}).get(chat_id)
        if existing:
            return str(existing)
        conversation = _api_call("POST", "/conversations", {"metadata": {"channel": "telegram"}})
        conversation_id = str(conversation["id"])
        state["conversations"][chat_id] = conversation_id
        atomic_write_json(path, state)
        return conversation_id


def reset_conversation(chat_id: str) -> None:
    path = Path(get_settings().telegram_assistant_state_path)
    with file_lock(path):
        state = _state()
        state.setdefault("conversations", {}).pop(chat_id, None)
        atomic_write_json(path, state)


def _completed_text(response: dict) -> str:
    if response.get("output_text"):
        return str(response["output_text"])
    parts: list[str] = []
    for item in response.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text" and content.get("text"):
                parts.append(str(content["text"]))
    return "\n".join(parts).strip()


def _run(chat_id: str, prompt: str, send_text) -> None:
    settings = get_settings()
    try:
        send_text(chat_id, "⏳ Working on it. I’ll send the result here when it’s done.")
        conversation_id = _conversation_id(chat_id)
        response = _api_call(
            "POST",
            "/responses",
            {
                "model": settings.openai_model,
                "conversation": conversation_id,
                "input": prompt,
                "instructions": ASSISTANT_INSTRUCTIONS,
                "background": True,
                "max_output_tokens": 2000,
            },
        )
        response_id = str(response["id"])
        deadline = time.monotonic() + settings.openai_background_timeout_seconds
        while response.get("status") in {"queued", "in_progress"} and time.monotonic() < deadline:
            time.sleep(max(1, settings.openai_background_poll_seconds))
            response = _api_call("GET", f"/responses/{response_id}")

        status = response.get("status")
        if status == "completed":
            answer = _completed_text(response) or "The task completed without a text response."
            chunks = [answer[index:index + 3900] for index in range(0, len(answer), 3900)]
            for index, chunk in enumerate(chunks):
                prefix = "✅ Done\n\n" if index == 0 else ""
                send_text(chat_id, prefix + chunk)
        elif time.monotonic() >= deadline:
            send_text(chat_id, "⚠️ The request is still taking too long. Please try a smaller request.")
        else:
            api_error = (response.get("error") or {}).get("message")
            send_text(chat_id, f"❌ I couldn’t complete that request: {api_error or status or 'unknown error'}")
    except Exception as exc:
        print(json.dumps({"telegram_assistant_error": str(exc)}), flush=True)
        send_text(chat_id, f"❌ I couldn’t complete that request: {exc}")
    finally:
        with _active_lock:
            _active_chats.discard(chat_id)


def start_assistant_task(chat_id: str, prompt: str, send_text) -> bool:
    with _active_lock:
        if chat_id in _active_chats:
            return False
        _active_chats.add(chat_id)
    threading.Thread(
        target=_run,
        args=(chat_id, prompt, send_text),
        name=f"telegram-ai-{chat_id}",
        daemon=True,
    ).start()
    return True
