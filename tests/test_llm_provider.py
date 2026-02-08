"""
Tests for plot_antenna.llm_provider module.

Smoke tests verifying basic construction and capabilities, no real API calls.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys

from plot_antenna.llm_provider import (
    ToolCall,
    LLMMessage,
    ToolDefinition,
    LLMResponse,
    create_provider,
    get_available_providers,
)


# ---------------------------------------------------------------------------
# TestDataTypes
# ---------------------------------------------------------------------------


class TestDataTypes:
    """Test unified data type construction."""

    def test_tool_call_construction(self):
        """ToolCall should construct with id, name, arguments."""
        tc = ToolCall(id="call_123", name="get_weather", arguments={"location": "NYC"})
        assert tc.id == "call_123"
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "NYC"}

    def test_llm_message_construction(self):
        """LLMMessage should construct with role, content, images, tool_calls."""
        # Basic message
        msg = LLMMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.images == []
        assert msg.tool_calls == []

        # Message with images
        msg_vision = LLMMessage(role="user", content="What's this?", images=["base64data"])
        assert msg_vision.images == ["base64data"]

        # Message with tool calls
        tc = ToolCall(id="1", name="func", arguments={})
        msg_tool = LLMMessage(role="assistant", content="", tool_calls=[tc])
        assert len(msg_tool.tool_calls) == 1
        assert msg_tool.tool_calls[0].name == "func"

    def test_tool_definition_construction(self):
        """ToolDefinition should construct with name, description, parameters."""
        tool = ToolDefinition(
            name="calculate",
            description="Performs calculation",
            parameters={"type": "object", "properties": {}},
        )
        assert tool.name == "calculate"
        assert tool.description == "Performs calculation"
        assert "type" in tool.parameters

    def test_llm_response_construction_defaults(self):
        """LLMResponse should construct with defaults."""
        # Minimal response
        resp = LLMResponse(content="Hello world")
        assert resp.content == "Hello world"
        assert resp.tool_calls == []
        assert resp.stop_reason == "end_turn"
        assert resp.raw is None

        # Response with tool calls
        tc = ToolCall(id="1", name="func", arguments={})
        resp_tool = LLMResponse(content="", tool_calls=[tc], stop_reason="tool_use")
        assert len(resp_tool.tool_calls) == 1
        assert resp_tool.stop_reason == "tool_use"


# ---------------------------------------------------------------------------
# TestCreateProvider
# ---------------------------------------------------------------------------


class TestCreateProvider:
    """Test provider factory function."""

    def test_create_openai_provider(self):
        """create_provider('openai') should return OpenAIProvider."""
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = MagicMock()

        with patch.dict(sys.modules, {"openai": mock_openai}):
            from plot_antenna.llm_provider import OpenAIProvider

            provider = create_provider("openai", api_key="test-key", model="gpt-4o-mini")
            assert isinstance(provider, OpenAIProvider)
            assert provider.model == "gpt-4o-mini"
            assert provider.provider_name() == "openai"

    def test_create_anthropic_provider(self):
        """create_provider('anthropic') should return AnthropicProvider."""
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = MagicMock()

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            from plot_antenna.llm_provider import AnthropicProvider

            provider = create_provider(
                "anthropic", api_key="test-key", model="claude-sonnet-4-20250514"
            )
            assert isinstance(provider, AnthropicProvider)
            assert provider.model == "claude-sonnet-4-20250514"
            assert provider.provider_name() == "anthropic"

    def test_create_unknown_provider_raises(self):
        """create_provider with unknown name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("unknown-provider")


# ---------------------------------------------------------------------------
# TestProviderCapabilities
# ---------------------------------------------------------------------------


class TestProviderCapabilities:
    """Test provider capability methods."""

    def test_openai_supports_vision_and_tools(self):
        """OpenAIProvider should report vision/tool support correctly."""
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = MagicMock()

        with patch.dict(sys.modules, {"openai": mock_openai}):
            from plot_antenna.llm_provider import OpenAIProvider

            # GPT-4o supports both vision and tools
            provider = OpenAIProvider(api_key="test", model="gpt-4o-mini")
            assert provider.supports_vision() is True
            assert provider.supports_tools() is True

            # o1 models don't support tools
            provider_o1 = OpenAIProvider(api_key="test", model="o1-preview")
            assert provider_o1.supports_tools() is False

    def test_openai_provider_name(self):
        """OpenAIProvider.provider_name() should return 'openai'."""
        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = MagicMock()

        with patch.dict(sys.modules, {"openai": mock_openai}):
            from plot_antenna.llm_provider import OpenAIProvider

            provider = OpenAIProvider(api_key="test", model="gpt-4o-mini")
            assert provider.provider_name() == "openai"

    def test_ollama_provider_creation(self):
        """OllamaProvider should construct with model and base_url."""
        from plot_antenna.llm_provider import OllamaProvider

        provider = OllamaProvider(model="llama3.1", base_url="http://localhost:11434")
        assert provider.model == "llama3.1"
        assert provider.base_url == "http://localhost:11434"
        assert provider.provider_name() == "ollama"
        assert provider.supports_tools() is True


# ---------------------------------------------------------------------------
# TestGetAvailableProviders
# ---------------------------------------------------------------------------


class TestGetAvailableProviders:
    """Test get_available_providers function."""

    def test_get_available_providers_returns_dict(self):
        """get_available_providers() should return dict."""
        available = get_available_providers()
        assert isinstance(available, dict)

    def test_get_available_providers_contains_expected_keys(self):
        """get_available_providers() should contain openai, anthropic, ollama."""
        available = get_available_providers()
        assert "openai" in available
        assert "anthropic" in available
        assert "ollama" in available
        # Values should be bool (True if installed, False if not)
        assert isinstance(available["openai"], bool)
        assert isinstance(available["anthropic"], bool)
        assert isinstance(available["ollama"], bool)
