"""Cerebras native model provider implementation.

Cerebras runs open-weight LLMs on wafer-scale hardware with sub-10ms
latency. The API is OpenAI-compatible at https://api.cerebras.ai/v1.
"""

import logging
from typing import TYPE_CHECKING, ClassVar, Optional

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from .openai_compatible import OpenAICompatibleProvider
from .registries.cerebras import CerebrasModelRegistry
from .registry_provider_mixin import RegistryBackedProviderMixin
from .shared import ModelCapabilities, ProviderType

logger = logging.getLogger(__name__)


class CerebrasModelProvider(RegistryBackedProviderMixin, OpenAICompatibleProvider):
    """Integration for Cerebras hosted models exposed over an OpenAI-style API."""

    FRIENDLY_NAME = "Cerebras"

    REGISTRY_CLASS = CerebrasModelRegistry
    MODEL_CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {}

    # Canonical model identifiers used for category routing.
    PRIMARY_MODEL = "qwen-3-235b-a22b-instruct-2507"  # best balanced/reasoning
    FAST_MODEL = "llama3.1-8b"  # fastest response

    def __init__(self, api_key: str, **kwargs):
        kwargs.setdefault("base_url", "https://api.cerebras.ai/v1")
        self._ensure_registry()
        super().__init__(api_key, **kwargs)
        self._invalidate_capability_cache()

    def get_provider_type(self) -> ProviderType:
        return ProviderType.CEREBRAS

    def get_preferred_model(self, category: "ToolModelCategory", allowed_models: list[str]) -> Optional[str]:
        """Pick a Cerebras-hosted model for a given tool category.

        Cerebras' selling point is latency, so FAST_RESPONSE prefers the 8B
        model, while EXTENDED_REASONING and BALANCED prefer the 235B Qwen.
        """
        from tools.models import ToolModelCategory

        if not allowed_models:
            return None

        if category == ToolModelCategory.FAST_RESPONSE:
            if self.FAST_MODEL in allowed_models:
                return self.FAST_MODEL
            if self.PRIMARY_MODEL in allowed_models:
                return self.PRIMARY_MODEL
            return allowed_models[0]

        # EXTENDED_REASONING and BALANCED both prefer the largest capable model.
        if self.PRIMARY_MODEL in allowed_models:
            return self.PRIMARY_MODEL
        if self.FAST_MODEL in allowed_models:
            return self.FAST_MODEL
        return allowed_models[0]


# Load registry data at import time
CerebrasModelProvider._ensure_registry()
