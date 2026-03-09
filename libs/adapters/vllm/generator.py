"""vLLM generator adapter stub.

Satisfies the ``Generator`` protocol without importing any
external vLLM or HTTP client libraries at module level.
"""

from __future__ import annotations

from libs.adapters.config import VllmConfig
from libs.contracts.generation import GeneratedAnswer
from libs.generation.models import GenerationRequest

_NOT_IMPLEMENTED_MSG = (
    "Install vllm client and implement VllmGenerator to use vLLM for generation"
)


class VllmGenerator:
    """Stub adapter for the vLLM inference server.

    Satisfies the ``Generator`` protocol.  All data methods raise
    ``NotImplementedError`` until a real implementation is provided.
    """

    def __init__(self, config: VllmConfig) -> None:
        self._config = config
        self._generator_id = f"vllm:{config.model_id}"

    # -- Protocol property ---------------------------------------------------

    @property
    def generator_id(self) -> str:
        return self._generator_id

    # -- Lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        """Connect to the vLLM server.  TODO: implement."""

    def close(self) -> None:
        """Close the vLLM connection.  TODO: implement."""

    def health_check(self) -> bool:
        """Return *False* — not connected."""
        return False

    # -- Generator protocol --------------------------------------------------

    def generate(self, request: GenerationRequest) -> GeneratedAnswer:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
