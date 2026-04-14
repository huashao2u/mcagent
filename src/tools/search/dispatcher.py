from __future__ import annotations

from typing import Any

from src.tools.search.brave_backend import BraveBackend
from src.tools.search.exa_backend import ExaBackend
from src.tools.search.local_retriever import LocalRetriever
from src.tools.search.mock_retriever import MockRetriever
from src.tools.search.serper_backend import SerperBackend
from src.tools.search.tavily_backend import TavilyBackend


ONLINE_BACKENDS = {
    "serper": SerperBackend,
    "brave": BraveBackend,
    "tavily": TavilyBackend,
    "exa": ExaBackend,
}


def build_search_backend(search_config: dict[str, Any] | None = None):
    cfg = search_config or {}
    backend_name = str(cfg.get("backend", "mock_retriever"))
    top_k = int(cfg.get("top_k", 3))
    if backend_name == "mock_retriever":
        return MockRetriever(top_k=top_k)
    if backend_name == "local_retriever":
        return LocalRetriever(top_k=top_k)
    if backend_name in ONLINE_BACKENDS:
        if not bool(cfg.get("enable_online_backend", False)):
            raise RuntimeError(f"Online backend `{backend_name}` is disabled by config.")
        return ONLINE_BACKENDS[backend_name]()
    raise KeyError(f"Unsupported search backend: {backend_name}")
