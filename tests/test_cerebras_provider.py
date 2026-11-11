"""Tests for Cerebras provider implementation."""

import os
from unittest.mock import patch

import pytest

from providers.cerebras import CerebrasModelProvider
from providers.shared import ProviderType


class TestCerebrasProvider:
    """Test Cerebras provider functionality."""

    def setup_method(self):
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    def teardown_method(self):
        import utils.model_restrictions

        utils.model_restrictions._restriction_service = None

    @patch.dict(os.environ, {"CEREBRAS_API_KEY": "test-key"})
    def test_initialization(self):
        provider = CerebrasModelProvider("test-key")
        assert provider.api_key == "test-key"
        assert provider.get_provider_type() == ProviderType.CEREBRAS
        assert provider.base_url == "https://api.cerebras.ai/v1"

    def test_initialization_with_custom_url(self):
        provider = CerebrasModelProvider("test-key", base_url="https://custom.cerebras.ai/v1")
        assert provider.api_key == "test-key"
        assert provider.base_url == "https://custom.cerebras.ai/v1"

    def test_model_validation_valid_models(self):
        provider = CerebrasModelProvider("test-key")

        assert provider.validate_model_name("qwen-3-235b-a22b-instruct-2507") is True
        assert provider.validate_model_name("gpt-oss-120b") is True
        assert provider.validate_model_name("zai-glm-4.7") is True
        assert provider.validate_model_name("llama3.1-8b") is True

    def test_model_validation_aliases(self):
        provider = CerebrasModelProvider("test-key")

        assert provider.validate_model_name("qwen-235b") is True
        assert provider.validate_model_name("cerebras-fast") is True
        assert provider.validate_model_name("glm-4.7") is True
        assert provider.validate_model_name("gpt-oss") is True

    def test_model_validation_rejects_unknown(self):
        provider = CerebrasModelProvider("test-key")

        assert provider.validate_model_name("gpt-4") is False
        assert provider.validate_model_name("grok-4") is False
        assert provider.validate_model_name("gemini-pro") is False

    def test_resolve_model_alias(self):
        provider = CerebrasModelProvider("test-key")

        assert provider._resolve_model_name("cerebras-fast") == "llama3.1-8b"
        assert provider._resolve_model_name("qwen-235b") == "qwen-3-235b-a22b-instruct-2507"
        assert provider._resolve_model_name("gpt-oss") == "gpt-oss-120b"
        assert provider._resolve_model_name("glm-4.7") == "zai-glm-4.7"

    def test_get_preferred_model_fast_response_prefers_8b(self):
        from tools.models import ToolModelCategory

        provider = CerebrasModelProvider("test-key")
        allowed = ["qwen-3-235b-a22b-instruct-2507", "llama3.1-8b", "gpt-oss-120b"]

        picked = provider.get_preferred_model(ToolModelCategory.FAST_RESPONSE, allowed)
        assert picked == "llama3.1-8b"

    def test_get_preferred_model_extended_reasoning_prefers_qwen235b(self):
        from tools.models import ToolModelCategory

        provider = CerebrasModelProvider("test-key")
        allowed = ["qwen-3-235b-a22b-instruct-2507", "llama3.1-8b"]

        picked = provider.get_preferred_model(ToolModelCategory.EXTENDED_REASONING, allowed)
        assert picked == "qwen-3-235b-a22b-instruct-2507"

    def test_get_preferred_model_empty_allowed_returns_none(self):
        from tools.models import ToolModelCategory

        provider = CerebrasModelProvider("test-key")
        assert provider.get_preferred_model(ToolModelCategory.FAST_RESPONSE, []) is None

    def test_registry_loads_all_four_models(self):
        provider = CerebrasModelProvider("test-key")
        models = provider.list_models(respect_restrictions=False)

        # Canonical names should be present (aliases are resolved separately).
        assert "qwen-3-235b-a22b-instruct-2507" in models
        assert "gpt-oss-120b" in models
        assert "zai-glm-4.7" in models
        assert "llama3.1-8b" in models
