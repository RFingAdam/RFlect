"""
Unified LLM Provider Abstraction for RFlect

Supports OpenAI (GPT-4/5), Anthropic (Claude), and Ollama (local models).
Each provider translates the unified message/tool format to its native API.

All providers are optional — import errors are caught gracefully.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Unified Data Types
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A tool/function call requested by the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMMessage:
    """Unified message format for all providers."""
    role: str  # "system", "user", "assistant", "tool"
    content: str = ""
    images: List[str] = field(default_factory=list)  # base64-encoded strings
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None


@dataclass
class ToolDefinition:
    """Unified tool/function definition."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema


@dataclass
class LLMResponse:
    """Unified response from any provider."""
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # "end_turn", "tool_use", "max_tokens"
    raw: Any = None


# ---------------------------------------------------------------------------
# Base Provider
# ---------------------------------------------------------------------------

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[ToolDefinition]] = None,
        max_tokens: int = 500,
        temperature: float = 0.2,
        **kwargs,
    ) -> LLMResponse:
        """Send messages and get a response."""
        ...

    @abstractmethod
    def supports_tools(self) -> bool:
        """Whether this provider/model supports tool calling."""
        ...

    @abstractmethod
    def supports_vision(self) -> bool:
        """Whether this provider/model supports image input."""
        ...

    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g. 'openai', 'anthropic', 'ollama')."""
        ...


# ---------------------------------------------------------------------------
# OpenAI Provider
# ---------------------------------------------------------------------------

class OpenAIProvider(BaseLLMProvider):
    """OpenAI Chat Completions API (GPT-4) and Responses API (GPT-5)."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", **kwargs):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._kwargs = kwargs

    def provider_name(self) -> str:
        return "openai"

    def supports_tools(self) -> bool:
        # O-series models don't support function calling
        incompatible = ("o1-preview", "o1-mini", "o3-mini", "o3", "o1")
        return not any(self.model.startswith(p) for p in incompatible)

    def supports_vision(self) -> bool:
        # GPT-4o, GPT-5 family support vision
        vision_models = ("gpt-4o", "gpt-5", "gpt-4-turbo", "gpt-4-vision")
        return any(self.model.startswith(p) for p in vision_models)

    def chat(self, messages, tools=None, max_tokens=500, temperature=0.2, **kwargs):
        is_gpt5 = self.model.startswith("gpt-5")
        if is_gpt5:
            return self._chat_responses_api(messages, tools, max_tokens, **kwargs)
        else:
            return self._chat_completions_api(messages, tools, max_tokens, temperature)

    def _chat_completions_api(self, messages, tools, max_tokens, temperature):
        """GPT-4 family: Chat Completions API."""
        api_messages = []
        for msg in messages:
            if msg.role == "tool":
                api_messages.append({
                    "role": "function",
                    "name": msg.tool_name or "",
                    "content": msg.content,
                })
            elif msg.tool_calls:
                # Assistant message with function_call
                tc = msg.tool_calls[0]
                api_messages.append({
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                })
            elif msg.images:
                # Vision message
                content_parts = []
                if msg.content:
                    content_parts.append({"type": "text", "text": msg.content})
                for img_b64 in msg.images:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    })
                api_messages.append({"role": msg.role, "content": content_parts})
            else:
                api_messages.append({"role": msg.role, "content": msg.content})

        api_params = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add function definitions if tools provided
        if tools:
            api_params["functions"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in tools
            ]
            api_params["function_call"] = "auto"

        response = self.client.chat.completions.create(**api_params)
        msg = response.choices[0].message

        # Check for function call
        if msg.function_call:
            try:
                args = json.loads(msg.function_call.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            return LLMResponse(
                content=msg.content or "",
                tool_calls=[ToolCall(
                    id=f"call_{msg.function_call.name}",
                    name=msg.function_call.name,
                    arguments=args,
                )],
                stop_reason="tool_use",
                raw=response,
            )

        return LLMResponse(
            content=(msg.content or "").strip(),
            stop_reason="end_turn",
            raw=response,
        )

    def _chat_responses_api(self, messages, tools, max_tokens, **kwargs):
        """GPT-5 family: Responses API."""
        input_messages = []
        for msg in messages:
            if msg.role == "tool":
                input_messages.append({
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id or "",
                    "output": msg.content,
                })
            elif msg.tool_calls:
                tc = msg.tool_calls[0]
                input_messages.append({
                    "type": "function_call",
                    "call_id": tc.id,
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                })
            elif msg.images:
                content_parts = []
                if msg.content:
                    content_parts.append({"type": "input_text", "text": msg.content})
                for img_b64 in msg.images:
                    content_parts.append({
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{img_b64}",
                    })
                input_messages.append({"role": msg.role, "content": content_parts})
            else:
                input_messages.append({"role": msg.role, "content": msg.content})

        api_params = {
            "model": self.model,
            "input": input_messages,
        }

        if tools:
            api_params["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "strict": False,
                }
                for t in tools
            ]

        # GPT-5 reasoning/verbosity settings from kwargs
        reasoning_effort = kwargs.get("reasoning_effort", "low")
        text_verbosity = kwargs.get("text_verbosity", "medium")
        api_params["reasoning"] = {"effort": reasoning_effort}
        api_params["text"] = {"verbosity": text_verbosity}

        response = self.client.responses.create(**api_params)

        # Parse response output
        tool_calls = []
        text_content = ""

        if hasattr(response, "output") and response.output:
            for item in response.output:
                item_type = getattr(item, "type", None)
                if item_type == "function_call":
                    call_id = getattr(item, "call_id", "")
                    name = getattr(item, "name", "")
                    args_str = getattr(item, "arguments", "{}")
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    tool_calls.append(ToolCall(id=call_id, name=name, arguments=args))
                elif item_type == "message":
                    for ci in getattr(item, "content", []):
                        if getattr(ci, "type", None) == "output_text":
                            text_content = getattr(ci, "text", "")

        if not text_content and hasattr(response, "output_text"):
            text_content = response.output_text or ""

        if tool_calls:
            return LLMResponse(
                content=text_content.strip(),
                tool_calls=tool_calls,
                stop_reason="tool_use",
                raw=response,
            )

        return LLMResponse(
            content=text_content.strip(),
            stop_reason="end_turn",
            raw=response,
        )


# ---------------------------------------------------------------------------
# Anthropic Provider
# ---------------------------------------------------------------------------

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Messages API (Claude models)."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", **kwargs):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self._kwargs = kwargs

    def provider_name(self) -> str:
        return "anthropic"

    def supports_tools(self) -> bool:
        return True

    def supports_vision(self) -> bool:
        return True

    def chat(self, messages, tools=None, max_tokens=500, temperature=0.2, **kwargs):
        # Extract system message (Anthropic uses top-level system= parameter)
        system_text = ""
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                system_text += msg.content + "\n"
            elif msg.role == "tool":
                # Tool results go as user messages with tool_result blocks
                api_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id or "",
                        "content": msg.content,
                    }],
                })
            elif msg.tool_calls:
                # Assistant message with tool_use blocks
                content_blocks = []
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                api_messages.append({"role": "assistant", "content": content_blocks})
            elif msg.images:
                # Vision message with image blocks
                content_blocks = []
                for img_b64 in msg.images:
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_b64,
                        },
                    })
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                api_messages.append({"role": msg.role, "content": content_blocks})
            else:
                api_messages.append({"role": msg.role, "content": msg.content})

        api_params = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }

        if system_text.strip():
            api_params["system"] = system_text.strip()

        if tools:
            api_params["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]

        response = self.client.messages.create(**api_params)

        # Parse response
        text_content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        stop = "tool_use" if response.stop_reason == "tool_use" else "end_turn"

        return LLMResponse(
            content=text_content.strip(),
            tool_calls=tool_calls,
            stop_reason=stop,
            raw=response,
        )


# ---------------------------------------------------------------------------
# Ollama Provider
# ---------------------------------------------------------------------------

class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""

    def __init__(self, model: str = "llama3.1", base_url: str = "http://localhost:11434", **kwargs):
        self.model = model
        self.base_url = base_url
        self._kwargs = kwargs

    def provider_name(self) -> str:
        return "ollama"

    def supports_tools(self) -> bool:
        # Most modern models support tools
        return True

    def supports_vision(self) -> bool:
        vision_keywords = ("llava", "llama3.2-vision", "gemma3", "qwen2.5-vl", "llama4")
        return any(v in self.model.lower() for v in vision_keywords)

    def chat(self, messages, tools=None, max_tokens=500, temperature=0.2, **kwargs):
        import ollama

        api_messages = []
        for msg in messages:
            m = {"role": msg.role, "content": msg.content}
            if msg.images:
                m["images"] = msg.images  # Ollama accepts base64 strings directly
            if msg.role == "tool":
                m["role"] = "tool"
                if msg.tool_name:
                    m["tool_name"] = msg.tool_name
            api_messages.append(m)

        call_kwargs = {
            "model": self.model,
            "messages": api_messages,
        }

        if tools:
            call_kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        response = ollama.chat(**call_kwargs)

        # Parse response
        text_content = response.message.content or ""
        tool_calls = []

        if response.message.tool_calls:
            for tc in response.message.tool_calls:
                tool_calls.append(ToolCall(
                    id=f"ollama_{tc.function.name}",
                    name=tc.function.name,
                    arguments=tc.function.arguments if isinstance(tc.function.arguments, dict) else {},
                ))

        stop = "tool_use" if tool_calls else "end_turn"

        return LLMResponse(
            content=text_content.strip(),
            tool_calls=tool_calls,
            stop_reason=stop,
            raw=response,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_provider(provider_name: str, **kwargs) -> BaseLLMProvider:
    """Create a provider instance by name.

    Args:
        provider_name: "openai", "anthropic", or "ollama"
        **kwargs: Provider-specific arguments (api_key, model, base_url, etc.)

    Returns:
        A BaseLLMProvider instance

    Raises:
        ValueError: If provider_name is unknown
        ImportError: If the required package is not installed
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider,
    }
    cls = providers.get(provider_name)
    if cls is None:
        raise ValueError(f"Unknown provider: {provider_name}. Choose from: {list(providers)}")
    return cls(**kwargs)


def get_available_providers() -> Dict[str, bool]:
    """Check which providers have their packages installed."""
    available = {}
    for name, pkg in [("openai", "openai"), ("anthropic", "anthropic"), ("ollama", "ollama")]:
        try:
            __import__(pkg)
            available[name] = True
        except ImportError:
            available[name] = False
    return available
