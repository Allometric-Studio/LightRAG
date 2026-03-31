"""
Per-request token usage tracking for LightRAG API.

Uses a contextvars.ContextVar so that LLM and embedding wrapper functions
can record usage without any changes to the core LightRAG pipeline code.
"""

import threading
from contextvars import ContextVar

_current_tracker: ContextVar["TokenTracker | None"] = ContextVar(
    "lightrag_token_tracker", default=None
)


def get_current_tracker() -> "TokenTracker | None":
    return _current_tracker.get()


def set_current_tracker(tracker: "TokenTracker | None") -> None:
    _current_tracker.set(tracker)


class TokenTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._llm_calls = 0
        self._embedding_calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._embedding_tokens = 0
        self.llm_model: str | None = None
        self.embedding_model: str | None = None

    def add_llm_usage(self, counts: dict) -> None:
        with self._lock:
            self._llm_calls += 1
            self._prompt_tokens += counts.get("prompt_tokens", 0)
            self._completion_tokens += counts.get("completion_tokens", 0)

    def add_embedding_usage(self, counts: dict) -> None:
        with self._lock:
            self._embedding_calls += 1
            self._embedding_tokens += counts.get("total_tokens", 0) or counts.get(
                "prompt_tokens", 0
            )

    def add_usage(self, counts: dict) -> None:
        """Generic add_usage for compatibility with existing token_tracker interface.
        Distinguishes LLM vs embedding by presence of completion_tokens."""
        if counts.get("completion_tokens", 0) > 0:
            self.add_llm_usage(counts)
        else:
            self.add_embedding_usage(counts)

    def get_totals(self) -> dict:
        with self._lock:
            result = {
                "llm_calls": self._llm_calls,
                "embedding_calls": self._embedding_calls,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "embedding_tokens": self._embedding_tokens,
            }
            if self.llm_model:
                result["llm_model"] = self.llm_model
            if self.embedding_model:
                result["embedding_model"] = self.embedding_model
            return result