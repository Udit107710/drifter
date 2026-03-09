"""Pipeline orchestrators.

Thin composition layers that wire libraries into coherent pipelines:
- IngestionOrchestrator: source → parse → chunk → embed → index
- QueryOrchestrator: normalize → retrieve → rerank → build context → generate

Orchestrators call libraries sequentially, passing typed contracts between them.
They own degraded-mode logic (fallbacks, timeouts, circuit breakers).
They depend on library protocols and contracts — never on concrete adapters.
"""

from orchestrators.bootstrap import ServiceRegistry, create_registry
from orchestrators.ingestion import IngestionOrchestrator, IngestionPipelineResult
from orchestrators.query import QueryOrchestrator, QueryResult

__all__ = [
    "IngestionOrchestrator",
    "IngestionPipelineResult",
    "QueryOrchestrator",
    "QueryResult",
    "ServiceRegistry",
    "create_registry",
]
