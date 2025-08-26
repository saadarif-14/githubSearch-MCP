from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI


@dataclass
class _TextBlock:
    type: str
    text: str


@dataclass
class _MessageLike:
    content: List[_TextBlock]
    stop_reason: str


class OpenAIProvider:
    def __init__(self, model: str):
        self.client = OpenAI()
        self.model = model

    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts: List[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content = "\n".join([p for p in text_parts if p])
            elif isinstance(content, dict):
                # Rare case: single text block dict
                if content.get("type") == "text":
                    content = content.get("text", "")

            normalized.append({"role": role, "content": content})
        return normalized

    def add_user_message(self, messages: list, message: Any):
        if isinstance(message, _MessageLike):
            text = "\n".join([b.text for b in message.content if b.type == "text"]) 
        else:
            text = message if isinstance(message, str) else str(message)
        messages.append({"role": "user", "content": text})

    def add_assistant_message(self, messages: list, message: Any):
        if isinstance(message, _MessageLike):
            text = "\n".join([b.text for b in message.content if b.type == "text"]) 
        else:
            text = message if isinstance(message, str) else str(message)
        messages.append({"role": "assistant", "content": text})

    def text_from_message(self, message: _MessageLike) -> str:
        return "\n".join([block.text for block in message.content if block.type == "text"])

    def chat(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        temperature: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        thinking: bool = False,
        thinking_budget: int = 1024,
    ) -> _MessageLike:
        del tools, thinking, thinking_budget  # Not supported in this shim

        normalized_messages = self._normalize_messages(messages)
        if system:
            normalized_messages = [{"role": "system", "content": system}] + normalized_messages

        response = self.client.chat.completions.create(
            model=self.model,
            messages=normalized_messages,
            temperature=temperature,
            stop=stop_sequences if stop_sequences else None,
        )
        text = response.choices[0].message.content or ""

        return _MessageLike(
            content=[_TextBlock(type="text", text=text)],
            stop_reason="end_turn",
        )


